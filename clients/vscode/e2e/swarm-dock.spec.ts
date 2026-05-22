import { test, expect, connectShell } from './fixtures';

/**
 * SwarmDock — always-visible status pinned above the composer. Surfaces
 * progress + current step without forcing the user to scroll back to
 * the in-message SwarmActivityCard. Collapsed by default; click to
 * expand the full step list.
 */
test.describe('swarm dock', () => {
  test.beforeEach(async ({ webview }) => {
    await connectShell(webview, { mode: 'swarm' });
  });

  test('absent when there is no swarm activity yet', async ({ webview }) => {
    await expect(webview.page.locator('.swarm-dock')).toHaveCount(0);
  });

  test('appears as soon as a swarm_plan lands; shows progress + first step', async ({
    webview,
  }) => {
    await webview.pushSequence([
      { type: 'turn_start' },
      {
        type: 'swarm_plan',
        depth: 0,
        steps: [
          { index: 1, description: 'first step' },
          { index: 2, description: 'second step' },
          { index: 3, description: 'third step' },
        ],
      },
    ]);
    const dock = webview.page.locator('.swarm-dock');
    await expect(dock).toBeVisible();
    await expect(dock.locator('.swarm-dock-count')).toHaveText('0/3');
    // No running step yet → falls back to the first pending step.
    await expect(dock.locator('.swarm-dock-desc')).toHaveText('first step');
  });

  test('current step updates to the running worker', async ({ webview }) => {
    await webview.pushSequence([
      { type: 'turn_start' },
      {
        type: 'swarm_plan',
        depth: 0,
        steps: [
          { index: 1, description: 'alpha' },
          { index: 2, description: 'beta' },
        ],
      },
      { type: 'swarm_worker_start', depth: 0, stepIndex: 2, description: 'beta' },
    ]);
    // The dock should reflect "running beta", not "next alpha", because
    // the user wants to see the worker that's actually live.
    const dock = webview.page.locator('.swarm-dock');
    await expect(dock.locator('.swarm-dock-desc')).toHaveText('beta');
    await expect(dock).toHaveClass(/running/);
  });

  test('progress counter advances on each worker_done', async ({ webview }) => {
    await webview.pushSequence([
      { type: 'turn_start' },
      {
        type: 'swarm_plan',
        depth: 0,
        steps: [
          { index: 1, description: 'a' },
          { index: 2, description: 'b' },
          { index: 3, description: 'c' },
        ],
      },
    ]);
    const count = webview.page.locator('.swarm-dock-count');
    await expect(count).toHaveText('0/3');

    await webview.push({
      type: 'swarm_worker_done',
      depth: 0,
      stepIndex: 1,
      status: 'done',
    });
    await expect(count).toHaveText('1/3');

    await webview.push({
      type: 'swarm_worker_done',
      depth: 0,
      stepIndex: 2,
      status: 'done',
    });
    await expect(count).toHaveText('2/3');
  });

  test('expand reveals the full step list with the current step highlighted', async ({
    webview,
  }) => {
    await webview.pushSequence([
      { type: 'turn_start' },
      {
        type: 'swarm_plan',
        depth: 0,
        steps: [
          { index: 1, description: 'one' },
          { index: 2, description: 'two' },
          { index: 3, description: 'three' },
        ],
      },
      { type: 'swarm_worker_done', depth: 0, stepIndex: 1, status: 'done' },
      { type: 'swarm_worker_start', depth: 0, stepIndex: 2, description: 'two' },
    ]);

    // Steps list is hidden by default.
    await expect(webview.page.locator('.swarm-dock-steps')).toHaveCount(0);

    await webview.page.locator('.swarm-dock-summary').click();
    const steps = webview.page.locator('.swarm-dock-step');
    await expect(steps).toHaveCount(3);
    // Step 2 (the running one) carries the is-current class so styling
    // can highlight it without other markup changes.
    await expect(steps.nth(1)).toHaveClass(/is-current/);
    // Step 1 done, step 3 still pending.
    await expect(steps.nth(0)).toHaveClass(/done/);
    await expect(steps.nth(2)).toHaveClass(/pending/);

    // Clicking again collapses.
    await webview.page.locator('.swarm-dock-summary').click();
    await expect(webview.page.locator('.swarm-dock-steps')).toHaveCount(0);
  });

  test('depth>0 surfaces the depth label so nested swarms are visible', async ({
    webview,
  }) => {
    // The outer orchestrator stays visible in messages, but the dock
    // tracks the deepest currently-running swarm — that's where the
    // worker's active.
    await webview.pushSequence([
      { type: 'turn_start' },
      {
        type: 'swarm_plan',
        depth: 0,
        steps: [{ index: 1, description: 'outer' }],
      },
      {
        type: 'swarm_plan',
        depth: 1,
        steps: [
          { index: 1, description: 'inner one' },
          { index: 2, description: 'inner two' },
        ],
      },
      { type: 'swarm_worker_start', depth: 1, stepIndex: 1, description: 'inner one' },
    ]);
    const dock = webview.page.locator('.swarm-dock');
    await expect(dock.locator('.swarm-dock-depth')).toHaveText('d1');
    await expect(dock.locator('.swarm-dock-desc')).toHaveText('inner one');
  });

  test('persists across turn_end so users see the final result', async ({ webview }) => {
    // After the turn ends, the dock should still be there showing the
    // completed state — gives the user a moment to read what just
    // finished before the next turn wipes it.
    await webview.pushSequence([
      { type: 'turn_start' },
      {
        type: 'swarm_plan',
        depth: 0,
        steps: [{ index: 1, description: 'only step' }],
      },
      { type: 'swarm_worker_done', depth: 0, stepIndex: 1, status: 'done' },
      { type: 'turn_end', ok: true },
    ]);
    const dock = webview.page.locator('.swarm-dock');
    await expect(dock).toBeVisible();
    await expect(dock).toHaveClass(/done/);
    await expect(dock.locator('.swarm-dock-count')).toHaveText('1/1');
  });

  test('verifying indicator surfaces on swarm_verify', async ({ webview }) => {
    await webview.pushSequence([
      { type: 'turn_start' },
      {
        type: 'swarm_plan',
        depth: 0,
        steps: [{ index: 1, description: 'a' }],
      },
      { type: 'swarm_worker_done', depth: 0, stepIndex: 1, status: 'done' },
      { type: 'swarm_verify', depth: 0 },
    ]);
    await expect(webview.page.locator('.swarm-dock-verify')).toBeVisible();
  });

  test('long step descriptions ellipsize in the collapsed row', async ({ webview }) => {
    // Narrow sidebar — long step descriptions must clip, not push the
    // composer off screen.
    const longDesc =
      'this is a very long step description that should be ellipsized inside the collapsed dock row to keep the composer visible at the bottom of the panel';
    await webview.pushSequence([
      { type: 'turn_start' },
      {
        type: 'swarm_plan',
        depth: 0,
        steps: [{ index: 1, description: longDesc }],
      },
      { type: 'swarm_worker_start', depth: 0, stepIndex: 1, description: longDesc },
    ]);
    const desc = webview.page.locator('.swarm-dock-desc');
    await expect(desc).toHaveCSS('text-overflow', 'ellipsis');
    await expect(desc).toHaveCSS('white-space', 'nowrap');
  });
});
