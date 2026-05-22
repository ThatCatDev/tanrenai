import { test, expect, connectShell } from './fixtures';

/**
 * Swarm mode UX — these tests document what currently renders, including
 * the known v1 limitation: agentrpc.go folds swarm events (plan, worker
 * start/done, verify) into plain `content_delta` strings. A richer
 * protocol that emits structured `swarm_*` events for the webview to
 * render as cards is a separate piece of work tracked in TODO below.
 *
 * Tests here are characterization-style: they pin the current behavior
 * so a future swarm-UI rework can deliberately break them and replace
 * them. If you're refactoring swarm rendering, plan to rewrite this
 * file alongside the new component.
 */
test.describe('swarm mode (v1 — flat content)', () => {
  test.beforeEach(async ({ webview }) => {
    await connectShell(webview, { mode: 'swarm' });
  });

  test('placeholder + footer indicate swarm mode', async ({ webview }) => {
    await expect(webview.page.getByPlaceholder(/Brief the swarm/)).toBeVisible();
    await expect(webview.page.locator('.footer-mode')).toHaveText('swarm');
  });

  test('plan + worker output renders as flat assistant content (current v1 behavior)', async ({
    webview,
  }) => {
    // TODO(swarm-ui): once agentrpc.go emits structured swarm_plan /
    // swarm_worker_* events, replace this test with one that asserts a
    // step list, not a text blob.
    await webview.pushSequence([
      { type: 'turn_start' },
      { type: 'iteration_start', iteration: 1, maxIterations: 8 },
      { type: 'message_start', role: 'assistant', id: 's1' },
      {
        type: 'message_delta',
        id: 's1',
        text: '[swarm plan d=0]\n  1. Step one\n  2. Step two\n',
        channel: 'content',
      },
      {
        type: 'message_delta',
        id: 's1',
        text: '[swarm worker d=0 1] Step one done\n',
        channel: 'content',
      },
      { type: 'message_end', id: 's1' },
      { type: 'turn_end', ok: true },
    ]);

    // Until the structured-events refactor lands, the rendered surface
    // is whatever string the controller concatenated. We assert the
    // distinctive prefix is visible — if it disappears (good!) the
    // refactor has shipped and this test should be updated.
    await expect(webview.page.getByText(/swarm plan/)).toBeVisible();
    await expect(webview.page.getByText(/swarm worker d=0 1/)).toBeVisible();
  });
});
