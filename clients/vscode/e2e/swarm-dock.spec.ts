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

  test('ancestor breadcrumb appears when the active swarm is a sub-decomposition', async ({
    webview,
  }) => {
    // Common confusion: "the count went from 11 to 2". That's a depth
    // switch — the agent decomposed step N of the outer 11-step plan
    // into an inner 2-step plan. The breadcrumb keeps the outer plan
    // in view so users don't lose the overall progress when the dock
    // focuses on the sub-swarm.
    const outer = Array.from({ length: 11 }, (_, i) => ({
      index: i + 1,
      description: `outer ${i + 1}`,
    }));
    await webview.pushSequence([
      { type: 'turn_start' },
      { type: 'swarm_plan', depth: 0, steps: outer },
      { type: 'swarm_worker_done', depth: 0, stepIndex: 1, status: 'done' },
      { type: 'swarm_worker_done', depth: 0, stepIndex: 2, status: 'done' },
      { type: 'swarm_worker_done', depth: 0, stepIndex: 3, status: 'done' },
      { type: 'swarm_worker_start', depth: 0, stepIndex: 4, description: 'outer 4' },
      // Now step 4 spawns a child swarm with 2 steps.
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

    // Breadcrumb shows the depth-0 row above the main dock content.
    const crumb = webview.page.locator('.swarm-dock-crumb');
    await expect(crumb).toHaveCount(1);
    await expect(crumb.locator('.swarm-dock-crumb-depth')).toHaveText('d0');
    await expect(crumb.locator('.swarm-dock-crumb-count')).toHaveText('3/11');
    // The step the parent is "in" — the running depth-0 step that
    // spawned the sub-swarm.
    await expect(crumb.locator('.swarm-dock-crumb-step')).toContainText('outer 4');

    // Main row is the depth-1 active swarm.
    const dock = webview.page.locator('.swarm-dock');
    await expect(dock.locator('.swarm-dock-count')).toHaveText('0/2');
    await expect(dock.locator('.swarm-dock-depth')).toHaveText('d1');
    await expect(dock.locator('.swarm-dock-desc')).toHaveText('inner one');
  });

  test('clicking an ancestor crumb pins the dock to that depth', async ({ webview }) => {
    // Drill-down: a sub-decomposition (depth 1) is live, but the user
    // wants to see the outer plan's full step list. Click the d0 crumb
    // → dock pins to depth 0, expands, lists all outer steps.
    const outer = Array.from({ length: 11 }, (_, i) => ({
      index: i + 1,
      description: `outer ${i + 1}`,
    }));
    await webview.pushSequence([
      { type: 'turn_start' },
      { type: 'swarm_plan', depth: 0, steps: outer },
      { type: 'swarm_worker_done', depth: 0, stepIndex: 1, status: 'done' },
      { type: 'swarm_worker_done', depth: 0, stepIndex: 2, status: 'done' },
      { type: 'swarm_worker_done', depth: 0, stepIndex: 3, status: 'done' },
      { type: 'swarm_worker_start', depth: 0, stepIndex: 4, description: 'outer 4' },
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

    // Default: dock shows the live (depth-1) status.
    const dock = webview.page.locator('.swarm-dock');
    await expect(dock.locator('.swarm-dock-count')).toHaveText('0/2');

    // Click the d0 crumb. Now the dock pins to depth 0 — count
    // updates, "viewing" verb appears, full outer step list is shown
    // expanded (the user explicitly drilled in, they want all 11).
    await webview.page.locator('.swarm-dock-crumb-btn').click();
    await expect(dock).toHaveClass(/pinned/);
    await expect(dock.locator('.swarm-dock-count')).toHaveText('3/11');
    await expect(dock.locator('.swarm-dock-verb')).toHaveText('viewing');
    await expect(webview.page.locator('.swarm-dock-step')).toHaveCount(11);
    // Crumb itself is marked focused.
    await expect(webview.page.locator('.swarm-dock-crumb')).toHaveClass(/is-focused/);
  });

  test('clicking the main row while pinned returns to live status', async ({ webview }) => {
    await webview.pushSequence([
      { type: 'turn_start' },
      {
        type: 'swarm_plan',
        depth: 0,
        steps: [{ index: 1, description: 'outer' }],
      },
      { type: 'swarm_worker_start', depth: 0, stepIndex: 1, description: 'outer' },
      {
        type: 'swarm_plan',
        depth: 1,
        steps: [{ index: 1, description: 'inner' }],
      },
      { type: 'swarm_worker_start', depth: 1, stepIndex: 1, description: 'inner' },
    ]);

    // Pin to d0.
    await webview.page.locator('.swarm-dock-crumb-btn').click();
    const dock = webview.page.locator('.swarm-dock');
    await expect(dock).toHaveClass(/pinned/);
    await expect(dock.locator('.swarm-dock-desc')).toHaveText('outer');

    // Click the main summary row — un-pin and return to following the
    // live (deepest) swarm.
    await webview.page.locator('.swarm-dock-summary').click();
    await expect(dock).not.toHaveClass(/pinned/);
    await expect(dock.locator('.swarm-dock-desc')).toHaveText('inner');
  });

  test('depth 5 hierarchy: dock renders all 5 ancestor crumbs', async ({ webview }) => {
    // Nested swarms can in principle go arbitrarily deep. The dock
    // doesn't crash, and each crumb is clickable.
    await webview.push({ type: 'turn_start' });
    for (let d = 0; d <= 5; d++) {
      await webview.push({
        type: 'swarm_plan',
        depth: d,
        steps: [
          { index: 1, description: `d${d} step a` },
          { index: 2, description: `d${d} step b` },
        ],
      });
    }
    await webview.push({
      type: 'swarm_worker_start',
      depth: 5,
      stepIndex: 1,
      description: 'd5 step a',
    });

    // Active is d5; the breadcrumb should hold all 5 ancestors d0..d4.
    await expect(webview.page.locator('.swarm-dock-crumb')).toHaveCount(5);
    const crumbs = webview.page.locator('.swarm-dock-crumb-depth');
    for (let d = 0; d < 5; d++) {
      await expect(crumbs.nth(d)).toHaveText(`d${d}`);
    }
    // The current depth chip on the main row reflects d5.
    await expect(webview.page.locator('.swarm-dock-depth')).toHaveText('d5');
  });

  test('no breadcrumb when the active swarm is at depth 0', async ({ webview }) => {
    // The breadcrumb is only meaningful when there's a parent — a
    // single-line "d0 0/3" above an identical d0 main row would be
    // pure noise.
    await webview.pushSequence([
      { type: 'turn_start' },
      {
        type: 'swarm_plan',
        depth: 0,
        steps: [{ index: 1, description: 'a' }],
      },
    ]);
    await expect(webview.page.locator('.swarm-dock-breadcrumb')).toHaveCount(0);
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
