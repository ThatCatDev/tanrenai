import { test, expect, connectShell } from './fixtures';

/**
 * Markdown edge cases beyond markdown.spec.ts:
 *   - Nested lists (the common LLM-output shape)
 *   - Tables (GFM, regressions easy to miss without explicit test)
 *   - HTML special characters escape correctly (no XSS bleed through)
 *   - Long lines in code blocks scroll horizontally rather than wrap
 *   - Code blocks inside list items
 *   - Empty content doesn't crash
 *   - Inline markdown emphasis next to punctuation
 */
test.describe('markdown edge cases', () => {
  test.beforeEach(async ({ webview }) => {
    await connectShell(webview);
  });

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

  test('nested lists render with the right hierarchy', async ({ webview }) => {
    const body = await renderAssistant(
      webview,
      `1. outer one
   - inner A
   - inner B
2. outer two
   - inner C
`,
    );
    // Outer list = ol; inside the second outer li there's a nested ul.
    await expect(body.locator('ol > li')).toHaveCount(2);
    await expect(body.locator('ol > li:nth-child(1) > ul > li')).toHaveCount(2);
    await expect(body.locator('ol > li:nth-child(2) > ul > li')).toHaveCount(1);
    await expect(body.locator('ol > li:nth-child(2) > ul > li')).toContainText('inner C');
  });

  test('GFM tables render as proper <table>', async ({ webview }) => {
    const body = await renderAssistant(
      webview,
      `| Header 1 | Header 2 |
|----------|----------|
| cell a   | cell b   |
| cell c   | cell d   |
`,
    );
    await expect(body.locator('table')).toBeVisible();
    await expect(body.locator('th')).toHaveCount(2);
    await expect(body.locator('tbody tr')).toHaveCount(2);
    await expect(body.locator('tbody tr:nth-child(2) td:nth-child(2)')).toHaveText('cell d');
  });

  test('dangerous HTML in model output is inert (no scripts, no iframes)', async ({
    webview,
  }) => {
    // marked passes raw HTML through by design — the safety net is the
    // VS Code webview's CSP (`default-src 'none'; script-src 'nonce-X'`)
    // which neutralises everything that could execute. In production:
    //   - <script> blocked by script-src
    //   - <img onerror=…> blocked by img-src (no source allowed)
    //   - <iframe> blocked by frame-src
    // The dev shell lacks that CSP, so this test verifies the
    // *semantic* safety: even when injected HTML is parsed into the
    // DOM, nothing dangerous runs. The dialog handler catches alerts;
    // navigation tracking catches iframe loads.
    let alerted = false;
    let navigated = false;
    webview.page.on('dialog', (d) => {
      alerted = true;
      void d.dismiss();
    });
    webview.page.on('framenavigated', () => {
      navigated = true;
    });

    const body = await renderAssistant(
      webview,
      [
        'before',
        '',
        '<script>alert("xss-inline")</script>',
        '<img src=x onerror="alert(\'xss-img\')">',
        '<iframe src="https://example.com"></iframe>',
        '',
        'after',
      ].join('\n'),
    );

    // Surrounding prose still renders normally.
    await expect(body).toContainText('before');
    await expect(body).toContainText('after');
    // None of the injected vectors fired.
    expect(alerted).toBe(false);
    expect(navigated).toBe(false);
  });

  test('code blocks inside list items render and scroll', async ({ webview }) => {
    // Common LLM output pattern: a numbered step with code under it.
    // Both the list structure and the <pre> must survive parsing.
    const body = await renderAssistant(
      webview,
      `1. Run the command:

   \`\`\`sh
   npm install
   \`\`\`

2. Then build:

   \`\`\`sh
   npm run build
   \`\`\`
`,
    );
    await expect(body.locator('ol > li')).toHaveCount(2);
    await expect(body.locator('ol > li pre')).toHaveCount(2);
    await expect(body.locator('ol > li:nth-child(1) pre code')).toContainText('npm install');
    await expect(body.locator('ol > li:nth-child(2) pre code')).toContainText('npm run build');
  });

  test('long code lines do not break the column width', async ({ webview }) => {
    // Code blocks must allow horizontal scroll instead of forcing the
    // entire chat surface wider — that would push the composer off
    // screen in a narrow sidebar.
    const longLine = 'const veryLongIdentifier = ' + 'foo.'.repeat(60) + 'end;';
    await renderAssistant(webview, '```ts\n' + longLine + '\n```');
    const pre = webview.page.locator('.body.markdown pre');
    // overflow-x: auto is the visible cue that horizontal scroll is on.
    await expect(pre).toHaveCSS('overflow-x', 'auto');
  });

  test('inline emphasis is honored adjacent to punctuation', async ({ webview }) => {
    // GFM corner case: **bold**, *italic*, and `code` rendering when
    // adjacent to commas / parens / periods. Models often produce
    // exactly this shape ("the **foo()** function returns…").
    const body = await renderAssistant(
      webview,
      'the **bold**, *italic*, and `code` are all here.',
    );
    await expect(body.locator('strong')).toHaveText('bold');
    await expect(body.locator('em')).toHaveText('italic');
    await expect(body.locator('code')).toHaveText('code');
  });

  test('empty assistant content renders nothing without crashing', async ({ webview }) => {
    // A turn that produced no content (tool-only, e.g.) shouldn't
    // render an empty assistant bubble. The assistant entry only
    // renders if entry.content is truthy.
    await webview.push({ type: 'turn_start' });
    await webview.push({ type: 'message_start', role: 'assistant', id: 'a1' });
    await webview.push({ type: 'message_end', id: 'a1' });
    await webview.push({ type: 'turn_end', ok: true });
    // No assistant bubble — there was no content channel delta.
    await expect(webview.page.locator('.msg.assistant')).toHaveCount(0);
  });

  test('reasoning + content channel transition keeps both bubbles', async ({ webview }) => {
    // The agent emits reasoning first, then content. Both should
    // remain visible after the turn ends — the user wants to be able
    // to scroll back to "why did it do that".
    await webview.pushSequence([
      { type: 'turn_start' },
      { type: 'message_start', role: 'assistant', id: 'a1' },
      { type: 'message_delta', id: 'a1', text: 'analysing…', channel: 'reasoning' },
      { type: 'message_delta', id: 'a1', text: '# answer\n\nhere it is', channel: 'content' },
      { type: 'message_end', id: 'a1' },
      { type: 'turn_end', ok: true },
    ]);
    await expect(webview.page.locator('.msg.reasoning')).toBeVisible();
    await expect(webview.page.locator('.msg.assistant h1')).toHaveText('answer');
  });
});
