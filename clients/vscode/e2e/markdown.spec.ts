import { test, expect, connectShell } from './fixtures';

/**
 * Assistant content + reasoning are parsed as GitHub-flavored markdown
 * (see webview/markdown.ts). These tests pin which constructs actually
 * render — a regression that swaps in a markdown lib with a different
 * config (e.g. no gfm, no breaks) would surface here.
 */
test.describe('markdown rendering', () => {
  test.beforeEach(async ({ webview }) => {
    await connectShell(webview);
  });

  /** Helper: stream one assistant content delta, end the turn. Returns
   *  a locator for the rendered .body.markdown surface. */
  async function renderAssistant(
    webview: { push: (m: unknown) => Promise<void>; page: import('@playwright/test').Page },
    src: string,
  ) {
    await webview.push({ type: 'turn_start' });
    await webview.push({ type: 'message_start', role: 'assistant', id: 'a1' });
    await webview.push({ type: 'message_delta', id: 'a1', text: src, channel: 'content' });
    await webview.push({ type: 'message_end', id: 'a1' });
    await webview.push({ type: 'turn_end', ok: true });
    return webview.page.locator('.msg.assistant .body.markdown');
  }

  test('renders headings as semantic h1/h2/h3 (not literal #)', async ({ webview }) => {
    const body = await renderAssistant(
      webview,
      '# Title\n## Subhead\n### Smaller\n\nparagraph',
    );
    await expect(body.locator('h1')).toHaveText('Title');
    await expect(body.locator('h2')).toHaveText('Subhead');
    await expect(body.locator('h3')).toHaveText('Smaller');
    // Raw `#` markers must not appear in the rendered text.
    await expect(body).not.toContainText('# Title');
  });

  test('renders fenced code blocks as <pre><code>', async ({ webview }) => {
    const body = await renderAssistant(
      webview,
      'try this:\n\n```ts\nconst x = 1;\nfunction f() { return x; }\n```\n',
    );
    const pre = body.locator('pre');
    await expect(pre).toBeVisible();
    // Newlines inside the code block are preserved (white-space: pre).
    await expect(pre.locator('code')).toContainText('const x = 1;');
    await expect(pre.locator('code')).toContainText('function f()');
  });

  test('renders inline code distinctly from prose', async ({ webview }) => {
    const body = await renderAssistant(webview, 'call `fooBar()` to start');
    // Inline code lives directly under a paragraph (not under pre).
    const inline = body.locator('p code');
    await expect(inline).toHaveText('fooBar()');
  });

  test('renders bullet and numbered lists', async ({ webview }) => {
    const body = await renderAssistant(
      webview,
      '- first\n- second\n- third\n\n1. one\n2. two\n',
    );
    await expect(body.locator('ul li')).toHaveCount(3);
    await expect(body.locator('ol li')).toHaveCount(2);
    await expect(body.locator('ul li').first()).toHaveText('first');
    await expect(body.locator('ol li').nth(1)).toHaveText('two');
  });

  test('bold and italic emit <strong>/<em>', async ({ webview }) => {
    const body = await renderAssistant(webview, 'this is **bold** and *italic*');
    await expect(body.locator('strong')).toHaveText('bold');
    await expect(body.locator('em')).toHaveText('italic');
  });

  test('links render as anchors with the right href', async ({ webview }) => {
    const body = await renderAssistant(webview, 'see [the docs](https://example.com/docs)');
    const link = body.locator('a');
    await expect(link).toHaveText('the docs');
    await expect(link).toHaveAttribute('href', 'https://example.com/docs');
  });

  test('reasoning content is also rendered as markdown', async ({ webview }) => {
    // Reasoning often contains the same markdown patterns as content;
    // both go through renderMarkdown.
    await webview.push({ type: 'turn_start' });
    await webview.push({ type: 'message_start', role: 'assistant', id: 'a1' });
    await webview.push({
      type: 'message_delta',
      id: 'a1',
      text: 'let me think:\n- option A\n- option B',
      channel: 'reasoning',
    });
    await webview.push({ type: 'message_end', id: 'a1' });

    const reasoning = webview.page.locator('.msg.reasoning .body.markdown');
    await expect(reasoning.locator('li')).toHaveCount(2);
  });

  test('script tags in model output do not execute (CSP blocks)', async ({ webview }) => {
    // Defense in depth — even if the model writes <script>, the CSP on
    // the real webview blocks inline scripts. Our dev shell uses a
    // localStorage-backed stub and doesn't replicate VS Code's CSP, so
    // a script tag WOULD execute here if we let marked emit it. With
    // marked configured to NOT pass through raw HTML, the tag is
    // escaped at parse time — assert that.
    let alerted = false;
    webview.page.on('dialog', (d) => {
      alerted = true;
      void d.dismiss();
    });
    const body = await renderAssistant(
      webview,
      'hello\n\n<script>alert("xss")</script>\n\nworld',
    );
    // The raw text appears (escaped) but no script ran.
    await expect(body).toContainText('hello');
    await expect(body).toContainText('world');
    expect(alerted).toBe(false);
  });

  test('streaming markdown re-renders progressively', async ({ webview }) => {
    // marked is sync; the reducer re-parses on every delta. The header
    // should appear as soon as the # line completes, even while later
    // content is still arriving.
    await webview.push({ type: 'turn_start' });
    await webview.push({ type: 'message_start', role: 'assistant', id: 'a1' });
    await webview.push({ type: 'message_delta', id: 'a1', text: '# part one\n', channel: 'content' });
    await expect(webview.page.locator('.msg.assistant h1')).toHaveText('part one');

    await webview.push({ type: 'message_delta', id: 'a1', text: 'now some prose.', channel: 'content' });
    await expect(webview.page.locator('.msg.assistant p')).toContainText('now some prose');
  });
});
