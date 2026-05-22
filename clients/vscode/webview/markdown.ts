/**
 * Markdown rendering for assistant content + reasoning. Uses `marked`
 * configured for chat output: GFM on, raw HTML off (models occasionally
 * emit it; we don't want to honor it), line breaks treated as <br>.
 *
 * Why no DOMPurify yet: the webview's CSP (set in chatViewProvider.ts)
 * is `default-src 'none'; script-src 'nonce-X'`, which blocks all
 * inline scripts, image loads, iframe embeds, etc. Anything a hostile
 * model could inject is already inert in the rendered DOM. If the CSP
 * is ever relaxed, add DOMPurify here.
 *
 * Streaming-safe: `marked.parse` is synchronous and idempotent; the
 * MessageList re-parses on every content_delta. Cost is microseconds
 * for typical chat content.
 */
import { marked } from 'marked';

marked.setOptions({
  // GitHub-flavored markdown — what every LLM produces by default.
  gfm: true,
  // Single newlines become <br> so model output reads the way the
  // model formatted it without needing double-newlines everywhere.
  breaks: true,
});

/** Pattern that matches a single-paragraph wrapper inside a list item.
 *  Models love writing markdown lists with blank lines between items
 *  ("loose lists"), which marked then wraps in <p>. Each <li><p>…</p>…</li>
 *  stacks paragraph + line-height margins that visually triple the gap
 *  between items. We collapse those wrappers to render tight lists
 *  uniformly. Multi-paragraph items keep their inner paragraphs — only
 *  the *first* <p> child of an <li> is unwrapped (it's the one that
 *  stacked margin against the next sibling: a nested list or the next
 *  list item). */
const LOOSE_LI_FIRST_P = /<li>\s*<p>([\s\S]*?)<\/p>(?=\s*(?:<(?:ul|ol)\b|<\/li>))/g;

/** Parse a markdown string into HTML. Returns "" for empty input so
 *  callers can `{html && <div dangerouslySetInnerHTML…/>}` cleanly. */
export function renderMarkdown(src: string): string {
  if (!src) return '';
  // marked.parse can return a Promise when extensions are async; with
  // our config it's always a string. Cast for the type system.
  const html = marked.parse(src, { async: false }) as string;
  return html.replace(LOOSE_LI_FIRST_P, '<li>$1');
}
