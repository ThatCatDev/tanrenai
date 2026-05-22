import { test, expect, connectShell } from './fixtures';

/**
 * Tool-flow coverage beyond the happy path in tools.spec.ts:
 *  - tool_call_streaming (model is incrementally streaming args)
 *  - tool_result with ok=false (error styling)
 *  - intercepted tool_call_request (different from non-intercepted tool_call)
 *  - approval "Always" action
 */
test.describe('tool flows — extended', () => {
  test.beforeEach(async ({ webview }) => {
    await connectShell(webview, { mode: 'agent' });
  });

  test('tool_call_streaming surfaces in the activity bar before tool_call lands', async ({
    webview,
  }) => {
    // The model can take a few hundred ms to fully emit its tool-call
    // JSON args. During that window the user sees "preparing X (N chars)"
    // in the activity bar — without it, the UI looks frozen.
    await webview.push({ type: 'turn_start' });
    await webview.push({
      type: 'tool_call_streaming',
      index: 0,
      name: 'file_write',
      argsDelta: '{"path":"src/big.ts",',
    });
    await webview.push({
      type: 'tool_call_streaming',
      index: 0,
      name: 'file_write',
      argsDelta: '"content":"…"',
    });
    await expect(webview.page.getByText(/preparing/i)).toBeVisible();
    await expect(webview.page.getByText(/file_write/).first()).toBeVisible();
  });

  test('finalised tool_call clears the matching streaming entry', async ({ webview }) => {
    // Once tool_call lands the reducer drops the streaming placeholder
    // — otherwise users would see both the "preparing" hint and the
    // finalised card at the same time.
    await webview.push({ type: 'turn_start' });
    await webview.push({
      type: 'tool_call_streaming',
      index: 0,
      name: 'file_read',
      argsDelta: '{"path":',
    });
    await expect(webview.page.getByText(/preparing/i)).toBeVisible();
    await webview.push({
      type: 'tool_call',
      id: 't1',
      name: 'file_read',
      arguments: '{"path":"x.ts"}',
      intercepted: false,
    });
    await expect(webview.page.getByText(/preparing/i)).toHaveCount(0);
  });

  test('tool_result with ok=false styles the card as error', async ({ webview }) => {
    await webview.pushSequence([
      { type: 'turn_start' },
      {
        type: 'tool_call',
        id: 't1',
        name: 'shell_exec',
        arguments: '{"cmd":"bad"}',
        intercepted: false,
      },
      { type: 'tool_result', id: 't1', ok: false, content: 'exit 1: command not found' },
    ]);

    // The .tool.error class is the visual cue that the call failed —
    // separate styling from "running" or "ok".
    const tool = webview.page.locator('.tool').first();
    await expect(tool).toHaveClass(/error/);
    // Disclosure label says "Error" instead of "Result".
    await expect(tool.locator('summary')).toHaveText('Error');
  });

  test('intercepted tool renders with an "editor" tag', async ({ webview }) => {
    // Intercepted tools (file_write, patch_file) round-trip through the
    // VS Code editor for diff approval. The controller translates the
    // RPC `tool_call_request` into a view-level `tool_call` with
    // `intercepted: true`; the webview tags those so users know the
    // work is happening in the editor, not just terminal.
    await webview.push({ type: 'turn_start' });
    await webview.push({
      type: 'tool_call',
      id: 'r1',
      name: 'file_write',
      arguments: '{"path":"src/foo.ts","content":"x"}',
      intercepted: true,
    });
    const tool = webview.page.locator('.tool').first();
    await expect(tool.getByText('editor')).toBeVisible();
  });

  test('approval "Always" action posts {action:"always"}', async ({ webview }) => {
    // Allow-always persists the permission via the controller's
    // persistAlwaysAllow path — a different action than one-shot
    // allow. Wire format matters because the Go side dispatches on it.
    await webview.push({ type: 'turn_start' });
    await webview.push({
      type: 'approval_required',
      id: 'a1',
      name: 'shell_exec',
      arguments: '{"cmd":"ls"}',
    });
    await webview.clearSent();
    await webview.page.getByRole('button', { name: /always/i }).click();
    await expect.poll(() => webview.sentMessages()).toContainEqual({
      type: 'approval_decision',
      id: 'a1',
      action: 'always',
    });
  });

  test('approval_resolved marks the prompt as resolved (greyed out)', async ({ webview }) => {
    await webview.push({ type: 'turn_start' });
    await webview.push({
      type: 'approval_required',
      id: 'a1',
      name: 'shell_exec',
      arguments: '{"cmd":"ls"}',
    });
    await webview.push({ type: 'approval_resolved', id: 'a1' });
    const approval = webview.page.locator('.approval').first();
    await expect(approval).toHaveClass(/resolved/);
    // Action buttons should be gone — the decision is in.
    await expect(approval.getByRole('button', { name: /allow once/i })).toHaveCount(0);
  });
});
