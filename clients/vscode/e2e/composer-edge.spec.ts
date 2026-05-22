import { test, expect, connectShell } from './fixtures';

/**
 * Composer edge cases beyond composer.spec.ts:
 *   - Shift+Enter inserts a newline rather than submitting
 *   - Composer state during a running turn (Cancel replaces Send)
 *   - Multi-line input round-trips correctly
 *   - Empty composer with only attachments still enables Send
 */
test.describe('composer edge cases', () => {
  test.beforeEach(async ({ webview }) => {
    await connectShell(webview, { mode: 'agent' });
  });

  test('Shift+Enter inserts a newline instead of submitting', async ({ webview }) => {
    const input = webview.page.getByPlaceholder(/Ask Tanrenai/);
    await input.focus();
    await input.fill('line one');
    await webview.clearSent();
    await input.press('Shift+Enter');
    // After Shift+Enter the textarea contains a newline; no send went out.
    await input.pressSequentially('line two');
    expect(await input.inputValue()).toBe('line one\nline two');
    expect(await webview.sentMessages()).not.toContainEqual(
      expect.objectContaining({ type: 'send' }),
    );
  });

  test('multi-line input is sent verbatim with embedded newlines', async ({ webview }) => {
    const input = webview.page.getByPlaceholder(/Ask Tanrenai/);
    await input.fill('first line\nsecond line\nthird line');
    await webview.clearSent();
    await input.press('Enter');
    await expect.poll(() => webview.sentMessages()).toContainEqual({
      type: 'send',
      content: 'first line\nsecond line\nthird line',
    });
  });

  test('Send remains disabled with empty input even mid-turn', async ({ webview }) => {
    // Cancel is what replaces Send while a turn runs — the Send button
    // shouldn't briefly appear (and shouldn't be enabled) in any state
    // where the composer is empty during a turn.
    await webview.push({ type: 'turn_start' });
    await expect(webview.page.getByRole('button', { name: /^send$/i })).toHaveCount(0);
    await expect(webview.page.getByRole('button', { name: /^cancel$/i })).toBeVisible();
  });

  test('attachments alone can satisfy Send (no text needed)', async ({ webview }) => {
    // A user attaching just an editor selection + clicking Send is a
    // valid flow ("review this selection"). Send must enable on
    // attachment alone — a regression that gated only on `text.length`
    // would force users to type a placeholder word.
    await webview.push({
      type: 'attach_selection',
      selection: {
        label: 'a.ts:1-1',
        path: 'a.ts',
        languageId: 'typescript',
        startLine: 1,
        endLine: 1,
        text: 'export const x = 1;',
      },
    });
    await expect(webview.page.getByRole('button', { name: 'Send' })).toBeEnabled();
  });

  test('Send carries attachments in the outbound payload', async ({ webview }) => {
    const sel = {
      label: 'a.ts:5-10',
      path: 'a.ts',
      languageId: 'typescript',
      startLine: 5,
      endLine: 10,
      text: 'function foo() {}',
    };
    await webview.push({ type: 'attach_selection', selection: sel });
    await webview.page.getByPlaceholder(/Ask Tanrenai/).fill('explain this');
    await webview.clearSent();
    await webview.page.getByRole('button', { name: 'Send' }).click();

    const sent = await webview.sentMessages();
    const send = (sent as Array<{ type: string }>).find((m) => m.type === 'send') as
      | { content: string; attachments?: typeof sel[] }
      | undefined;
    expect(send).toBeDefined();
    expect(send?.content).toBe('explain this');
    expect(send?.attachments).toEqual([sel]);
  });

  test('Send clears the composer after firing', async ({ webview }) => {
    const input = webview.page.getByPlaceholder(/Ask Tanrenai/);
    await input.fill('one shot');
    await webview.page.getByRole('button', { name: 'Send' }).click();
    // Leaving stale text in the composer after sending would invite
    // accidental double-sends. The composer must clear.
    await expect(input).toHaveValue('');
  });
});
