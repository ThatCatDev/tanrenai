import { test, expect, connectShell } from './fixtures';

/**
 * Tool result disclosure — collapsed by default, click to expand. The
 * disclosure is the only way to see what a tool actually returned;
 * a regression that broke the toggle would hide all tool output
 * silently.
 */
test.describe('tool result expansion', () => {
  test.beforeEach(async ({ webview }) => {
    await connectShell(webview);
  });

  test('result is collapsed by default; clicking the summary expands it', async ({
    webview,
  }) => {
    await webview.pushSequence([
      { type: 'turn_start' },
      {
        type: 'tool_call',
        id: 't1',
        name: 'file_read',
        arguments: '{"path":"foo.ts"}',
        intercepted: false,
      },
      { type: 'tool_result', id: 't1', ok: true, content: 'export function foo() {}' },
    ]);

    const details = webview.page.locator('.tool details');
    await expect(details).not.toHaveAttribute('open', '');
    // Body is not visible until expanded.
    await expect(details.locator('pre, code')).toBeHidden();

    await details.locator('summary').click();
    await expect(details).toHaveAttribute('open', '');
    await expect(details).toContainText('export function foo()');
  });

  test('summary label says "Running…" before the result lands', async ({ webview }) => {
    await webview.push({ type: 'turn_start' });
    await webview.push({
      type: 'tool_call',
      id: 't1',
      name: 'shell_exec',
      arguments: '{"cmd":"sleep 1"}',
      intercepted: false,
    });
    // No tool_result yet — disclosure should hint that the tool is in
    // flight; the bare "Result" label would imply it's already done.
    await expect(webview.page.locator('.tool summary')).toHaveText('Running…');
  });

  test('summary label flips to "Error" on a failure result', async ({ webview }) => {
    await webview.pushSequence([
      { type: 'turn_start' },
      {
        type: 'tool_call',
        id: 't1',
        name: 'shell_exec',
        arguments: '{"cmd":"bad"}',
        intercepted: false,
      },
      { type: 'tool_result', id: 't1', ok: false, content: 'command not found' },
    ]);
    await expect(webview.page.locator('.tool summary')).toHaveText('Error');
  });
});
