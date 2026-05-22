import { test, expect, connectShell } from './fixtures';

/**
 * Activity bar surfaces what the agent is doing right now — the
 * always-visible signal that tells the user the agent hasn't hung.
 * `deriveActivity` in state.ts decides which kind to show based on
 * entry/tool/approval state; tests here pin each kind's visible
 * representation so a regression in derivation order is loud.
 */
test.describe('activity bar', () => {
  test.beforeEach(async ({ webview }) => {
    await connectShell(webview, { mode: 'agent' });
  });

  test('idle when no turn is running', async ({ webview }) => {
    // Nothing in flight → bar should be silent (the layout still
    // reserves the row to avoid jumping; "idle" just means no text).
    // The exact rendering of idle is "no activity row visible".
    await expect(webview.page.locator('.activity-bar')).toHaveCount(0);
  });

  test('thinking shows during an open assistant bubble with no content yet', async ({
    webview,
  }) => {
    await webview.push({ type: 'turn_start' });
    await webview.push({ type: 'message_start', role: 'assistant', id: 'a1' });
    // No content delta yet — agent is in prompt-eval / first-token wait.
    await expect(webview.page.getByText(/thinking/i)).toBeVisible();
  });

  test('generating shows once content starts streaming', async ({ webview }) => {
    await webview.push({ type: 'turn_start' });
    await webview.push({ type: 'message_start', role: 'assistant', id: 'a1' });
    await webview.push({ type: 'message_delta', id: 'a1', text: 'word', channel: 'content' });
    await expect(webview.page.getByText(/generating/i)).toBeVisible();
  });

  test('tool kind appears while a tool is running', async ({ webview }) => {
    await webview.push({ type: 'turn_start' });
    await webview.push({
      type: 'tool_call',
      id: 't1',
      name: 'shell_exec',
      arguments: '{"cmd":"ls"}',
      intercepted: false,
    });
    // The activity bar names the in-flight tool — important for long
    // shell commands where this is the only progress signal.
    await expect(webview.page.getByText(/shell_exec/).first()).toBeVisible();
  });

  test('awaiting_approval beats everything else in the activity bar', async ({ webview }) => {
    // Approval is blocking. Even if a tool just started, the bar must
    // surface the approval — otherwise the user thinks the model is
    // working when actually it's waiting on a click.
    await webview.push({ type: 'turn_start' });
    await webview.push({
      type: 'tool_call',
      id: 't1',
      name: 'shell_exec',
      arguments: '{"cmd":"x"}',
      intercepted: false,
    });
    await webview.push({
      type: 'approval_required',
      id: 'a1',
      name: 'shell_exec',
      arguments: '{"cmd":"y"}',
    });
    await expect(webview.page.getByText(/awaiting approval/)).toBeVisible();
  });

  test('iteration counter advances on iteration_start', async ({ webview }) => {
    await webview.push({ type: 'turn_start' });
    await webview.push({ type: 'iteration_start', iteration: 1, maxIterations: 8 });
    await expect(webview.page.getByText(/iter 1\/8/)).toBeVisible();

    await webview.push({ type: 'iteration_start', iteration: 2, maxIterations: 8 });
    await expect(webview.page.getByText(/iter 2\/8/)).toBeVisible();
  });

  test('returns to idle after turn_end', async ({ webview }) => {
    await webview.push({ type: 'turn_start' });
    await webview.push({ type: 'message_start', role: 'assistant', id: 'a1' });
    await webview.push({ type: 'message_delta', id: 'a1', text: 'done.', channel: 'content' });
    await expect(webview.page.getByText(/generating/i)).toBeVisible();

    await webview.push({ type: 'message_end', id: 'a1' });
    await webview.push({ type: 'turn_end', ok: true });
    // The activity bar should be absent again — turnRunning=false →
    // deriveActivity returns idle → ActivityBar renders nothing.
    await expect(webview.page.locator('.activity-bar')).toHaveCount(0);
  });
});
