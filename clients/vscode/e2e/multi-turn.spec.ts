import { test, expect, connectShell } from './fixtures';

/**
 * Multi-turn conversation contracts — the realistic usage pattern of
 * "send a message, get a response, send another, get another". Each
 * turn must accumulate cleanly into the chat without smearing entries
 * across turns, and per-turn counters (iteration, token_rate) reset.
 */
test.describe('multi-turn conversations', () => {
  test.beforeEach(async ({ webview }) => {
    await connectShell(webview, { mode: 'agent' });
  });

  test('entries accumulate across multiple turns', async ({ webview }) => {
    await webview.pushSequence([
      { type: 'message_start', role: 'user', id: 'u1' },
      { type: 'message_delta', id: 'u1', text: 'first question' },
      { type: 'message_end', id: 'u1' },
      { type: 'turn_start' },
      { type: 'message_start', role: 'assistant', id: 'a1' },
      { type: 'message_delta', id: 'a1', text: 'first answer', channel: 'content' },
      { type: 'message_end', id: 'a1' },
      { type: 'turn_end', ok: true },

      { type: 'message_start', role: 'user', id: 'u2' },
      { type: 'message_delta', id: 'u2', text: 'second question' },
      { type: 'message_end', id: 'u2' },
      { type: 'turn_start' },
      { type: 'message_start', role: 'assistant', id: 'a2' },
      { type: 'message_delta', id: 'a2', text: 'second answer', channel: 'content' },
      { type: 'message_end', id: 'a2' },
      { type: 'turn_end', ok: true },
    ]);

    await expect(webview.page.getByText('first question')).toBeVisible();
    await expect(webview.page.getByText('first answer')).toBeVisible();
    await expect(webview.page.getByText('second question')).toBeVisible();
    await expect(webview.page.getByText('second answer')).toBeVisible();
    // Two user bubbles and two assistant bubbles — total 4 message entries.
    await expect(webview.page.locator('.msg')).toHaveCount(4);
  });

  test('iteration counter resets to 0 on a new turn', async ({ webview }) => {
    await webview.pushSequence([
      { type: 'turn_start' },
      { type: 'iteration_start', iteration: 3, maxIterations: 8 },
    ]);
    await expect(webview.page.getByText(/iter 3\/8/)).toBeVisible();
    await webview.push({ type: 'turn_end', ok: true });

    // New turn: iteration counter should not still say 3/8.
    await webview.push({ type: 'turn_start' });
    await expect(webview.page.getByText(/iter 3\/8/)).toHaveCount(0);
  });

  test('clear_chat empties entries but keeps the connected shell', async ({ webview }) => {
    await webview.pushSequence([
      { type: 'message_start', role: 'user', id: 'u1' },
      { type: 'message_delta', id: 'u1', text: 'hello' },
      { type: 'message_end', id: 'u1' },
    ]);
    await expect(webview.page.getByText('hello')).toBeVisible();

    await webview.push({ type: 'clear_chat' });
    await expect(webview.page.locator('.msg')).toHaveCount(0);
    // Composer and header still mounted — only entries cleared.
    await expect(webview.page.getByPlaceholder(/Ask Tanrenai/)).toBeVisible();
  });

  test('tool call → result → second tool call sequence renders both', async ({ webview }) => {
    await webview.pushSequence([
      { type: 'turn_start' },
      {
        type: 'tool_call',
        id: 't1',
        name: 'file_read',
        arguments: '{"path":"a.ts"}',
        intercepted: false,
      },
      { type: 'tool_result', id: 't1', ok: true, content: 'a' },
      {
        type: 'tool_call',
        id: 't2',
        name: 'file_read',
        arguments: '{"path":"b.ts"}',
        intercepted: false,
      },
      { type: 'tool_result', id: 't2', ok: true, content: 'b' },
    ]);

    // Two tool entries, each with its own result. A regression that
    // overwrote the first when the second landed would fail here.
    await expect(webview.page.locator('.tool')).toHaveCount(2);
    await expect(webview.page.getByText('{"path":"a.ts"}')).toBeVisible();
    await expect(webview.page.getByText('{"path":"b.ts"}')).toBeVisible();
  });
});
