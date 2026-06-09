import { useCallback, useLayoutEffect, useRef, useState } from 'preact/hooks';
import { send } from '../host';
import { renderMarkdown } from '../markdown';
import type { Activity, Entry } from '../state';

interface Props {
  entries: Entry[];
  activity: Activity;
}

// Scroll is "pinned to bottom" when the user is within this many pixels of
// the bottom. Streaming content auto-scrolls only while pinned.
const PIN_THRESHOLD = 80;

/**
 * Build a fingerprint that changes whenever any rendered text grows or a
 * new entry arrives. Used as the dep for the auto-scroll effect — without
 * it, useLayoutEffect with no deps would still run too eagerly during
 * React's reconciliation, but with it we trigger exactly when content
 * has changed in a way that affects scroll height.
 */
function contentSignature(entries: Entry[]): string {
  let sig = `${entries.length}`;
  for (const e of entries) {
    if (e.kind === 'user') sig += `|u${e.content.length}`;
    else if (e.kind === 'assistant') sig += `|a${e.content.length}+${e.reasoning.length}`;
    else if (e.kind === 'tool') sig += `|t${e.result ? '!' : '?'}${e.args.length}`;
    else if (e.kind === 'approval') sig += `|p${e.resolved ? '!' : '?'}`;
    else if (e.kind === 'error') sig += `|e`;
    else if (e.kind === 'swarm') {
      // Swarm activity grows when steps transition status — encode each
      // step's status so auto-scroll fires when a worker_done arrives.
      sig += `|s${e.depth}.${e.verifying ? 'v' : ''}.` + e.steps.map((s) => s.status[0]).join('');
    } else if (e.kind === 'compaction') {
      sig += `|c${e.phase[0]}`;
    }
  }

  return sig;
}

export function MessageList({ entries, activity }: Props) {
  const ref = useRef<HTMLDivElement | null>(null);
  // Default to pinned. The user must scroll up explicitly to break it.
  const pinnedRef = useRef(true);
  const [unreadGrowth, setUnreadGrowth] = useState(false);
  // Bump-counter so scroll-driven pin toggles cause a re-render — refs
  // alone don't trigger Preact's reconciliation.
  const [, setForceTick] = useState(0);
  // Tracks whether we're currently scrolling programmatically — so the
  // resulting scroll event doesn't accidentally toggle pinned.
  const programmaticScrollRef = useRef(false);

  const scrollToBottom = useCallback((smooth = false) => {
    const el = ref.current;
    if (!el) return;
    programmaticScrollRef.current = true;
    el.scrollTo({ top: el.scrollHeight, behavior: smooth ? 'smooth' : 'auto' });
    // Reset the flag on the next tick — by then the scroll event has fired.
    requestAnimationFrame(() => {
      programmaticScrollRef.current = false;
    });
  }, []);

  // Latest user id — submit forces re-pin and scroll.
  const lastUserId = (() => {
    for (let i = entries.length - 1; i >= 0; i--) {
      if (entries[i].kind === 'user') return entries[i].id;
    }

    return undefined;
  })();

  // useLayoutEffect runs after DOM commit, before paint — so scrollHeight
  // is correct and the user never sees the un-scrolled state flash.
  useLayoutEffect(() => {
    if (lastUserId) {
      pinnedRef.current = true;
      setUnreadGrowth(false);
      scrollToBottom();
    }
  }, [lastUserId, scrollToBottom]);

  const sig = contentSignature(entries);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (pinnedRef.current) {
      scrollToBottom();
      if (unreadGrowth) setUnreadGrowth(false);
    } else {
      // Content changed while unpinned — surface the badge.
      setUnreadGrowth(true);
    }
    // We deliberately depend on `sig` only — pinnedRef is a ref, not state.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sig]);

  const onScroll = () => {
    if (programmaticScrollRef.current) return;
    const el = ref.current;
    if (!el) return;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    const nowPinned = distance < PIN_THRESHOLD;
    if (nowPinned !== pinnedRef.current) {
      pinnedRef.current = nowPinned;
      // Force a re-render so the jump button shows/hides correctly.
      if (nowPinned && unreadGrowth) {
        setUnreadGrowth(false);
      } else if (!nowPinned) {
        // re-render via state poke — toggling unreadGrowth would be wrong here.
        setForceTick((t) => t + 1);
      }
    }
  };

  const pinned = pinnedRef.current;

  return (
    <div class="messages-wrap">
      <div class="messages" ref={ref} onScroll={onScroll}>
        {entries.map((e) => (
          <EntryView key={e.id} entry={e} activity={activity} />
        ))}
      </div>
      {!pinned && unreadGrowth && (
        <button
          class="jump-to-latest"
          onClick={() => {
            pinnedRef.current = true;
            setUnreadGrowth(false);
            setForceTick((t) => t + 1);
            scrollToBottom(true);
          }}
        >
          ↓ New messages
        </button>
      )}
    </div>
  );
}

function EntryView({ entry, activity }: { entry: Entry; activity: Activity }) {
  switch (entry.kind) {
    case 'user':
      return (
        <div class="msg user">
          <div class="role">You</div>
          <div class="body">{entry.content}</div>
        </div>
      );
    case 'assistant': {
      const reasoningStreaming = entry.open && activity.kind === 'thinking';
      const contentStreaming = entry.open && activity.kind === 'generating';

      return (
        <>
          {entry.reasoning && (
            <div class={`msg reasoning${reasoningStreaming ? ' streaming' : ''}`}>
              <div class="role">
                Thinking
                {reasoningStreaming && <span class="pulse" />}
              </div>
              <div
                class="body markdown"
                // CSP `default-src 'none'; script-src 'nonce-X'` blocks
                // anything dangerous a model could inject (scripts, img
                // loads, iframes) so the parsed HTML is safe to mount.
                // See webview/markdown.ts.
                dangerouslySetInnerHTML={{ __html: renderMarkdown(entry.reasoning) }}
              />
            </div>
          )}
          {entry.content && (
            <div class={`msg assistant${contentStreaming ? ' streaming' : ''}`}>
              <div class="role">Tanrenai</div>
              <div class="body markdown">
                <span dangerouslySetInnerHTML={{ __html: renderMarkdown(entry.content) }} />
                {contentStreaming && <span class="caret" />}
              </div>
            </div>
          )}
        </>
      );
    }
    case 'tool':
      return <ToolCard entry={entry} />;
    case 'approval':
      return <ApprovalCard entry={entry} />;
    case 'swarm':
      return <SwarmActivityCard entry={entry} />;
    case 'error':
      return (
        <div class="msg reasoning" style="color: var(--vscode-charts-red);">
          <div class="role">Error</div>
          <div class="body">{entry.text}</div>
        </div>
      );
    case 'compaction':
      return <CompactionRow entry={entry} />;
  }
}

function CompactionRow({ entry }: { entry: Extract<Entry, { kind: 'compaction' }> }) {
  let label: string;
  let cls = 'compaction';
  if (entry.phase === 'start') {
    label = 'Compacting older messages…';
    cls += ' compaction-running';
  } else if (entry.phase === 'done') {
    label = `Compacted ${entry.messages ?? 0} message${entry.messages === 1 ? '' : 's'} into summary`;
  } else if (entry.phase === 'noop') {
    label = 'Nothing to compact — context not full enough yet';
  } else {
    label = `Compact failed${entry.error ? `: ${entry.error}` : ''}`;
    cls += ' compaction-error';
  }
  return (
    <div class={cls} role="status">
      <span class="compaction-rule" aria-hidden="true" />
      <span class="compaction-label">{label}</span>
      <span class="compaction-rule" aria-hidden="true" />
    </div>
  );
}

function SwarmActivityCard({ entry }: { entry: Extract<Entry, { kind: 'swarm' }> }) {
  const total = entry.steps.length;
  const done = entry.steps.filter((s) => s.status === 'done').length;
  // Pluralisation matters here — "1 of 1 step" looks broken. Keep
  // pluralisation simple; agent plans never go below 1 step in practice.
  const stepWord = total === 1 ? 'step' : 'steps';
  return (
    <div class="swarm">
      <div class="swarm-head">
        <span class="bracket">〔</span>
        <span>Swarm</span>
        {entry.depth > 0 && <span class="swarm-depth">depth {entry.depth}</span>}
        <span class="swarm-progress">
          {done}/{total} {stepWord}
        </span>
        {entry.verifying && <span class="swarm-verify">verifying…</span>}
        <span class="bracket" style="margin-left:auto">〕</span>
      </div>
      {entry.architectSpec && (
        // Architect spec is usually multi-line markdown; render as
        // pre-wrapped text so structure survives without pulling in a
        // full markdown renderer just for this surface.
        <details class="swarm-architect">
          <summary>Architecture spec</summary>
          <pre>{entry.architectSpec}</pre>
        </details>
      )}
      <ol class="swarm-steps">
        {entry.steps.map((s) => (
          <li key={s.index} class={`swarm-step ${s.status}`}>
            <span class="swarm-step-marker" aria-hidden="true">
              {stepGlyph(s.status)}
            </span>
            <span class="swarm-step-body">
              <span class="swarm-step-desc">{s.description}</span>
              {s.status === 'done' && s.result && (
                <span class="swarm-step-result">{s.result}</span>
              )}
              {s.status === 'error' && s.error && (
                <span class="swarm-step-error">{s.error}</span>
              )}
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}

/** Single-character status glyph for a step. Unicode shapes match the
 *  rest of the editorial-forge aesthetic — flat, no boxed-tick icons. */
function stepGlyph(status: string): string {
  switch (status) {
    case 'done':
      return '✓';
    case 'error':
      return '✗';
    case 'running':
      return '◐';
    default:
      return '·';
  }
}

function ApprovalCard({ entry }: { entry: Extract<Entry, { kind: 'approval' }> }) {
  const argPreview = entry.args.length > 240 ? entry.args.slice(0, 240) + '…' : entry.args;
  const decide = (action: 'allow' | 'deny' | 'always') => {
    send({ type: 'approval_decision', id: entry.id, action });
  };

  return (
    <div class={`approval${entry.resolved ? ' resolved' : ''}`}>
      <div class="approval-head">
        <span class="bracket">〔</span>
        <span>Approval</span>
        <span class="approval-title">
          run <code>{entry.name}</code>
        </span>
        <span class="bracket" style="margin-left:auto">〕</span>
      </div>
      <div class="approval-args">{argPreview}</div>
      {!entry.resolved ? (
        <div class="approval-actions">
          <button onClick={() => decide('allow')}>Allow once</button>
          <button onClick={() => decide('always')}>Always</button>
          <button onClick={() => decide('deny')}>Deny</button>
        </div>
      ) : (
        <div class="approval-resolved-label">Resolved</div>
      )}
    </div>
  );
}

function ToolCard({ entry }: { entry: Extract<Entry, { kind: 'tool' }> }) {
  const failed = !!(entry.result && !entry.result.ok);
  const running = !entry.result;
  const argPreview = entry.args.length > 240 ? entry.args.slice(0, 240) + '…' : entry.args;

  return (
    <div class={`tool${failed ? ' error' : ''}${running ? ' running' : ''}`}>
      <div class="name">
        {running && <span class="tool-spinner" />}
        <code>{entry.name}</code>
        {entry.intercepted && <span class="tag">editor</span>}
      </div>
      <div class="args">{argPreview}</div>
      <details>
        <summary>{entry.result ? (entry.result.ok ? 'Result' : 'Error') : 'Running…'}</summary>
        <div class="result">{entry.result?.content ?? 'running…'}</div>
      </details>
    </div>
  );
}
