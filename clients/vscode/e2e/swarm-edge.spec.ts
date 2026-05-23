import { test, expect, connectShell } from './fixtures';

/**
 * Swarm edge cases beyond swarm.spec.ts:
 *   - Replan: a second swarm_plan at the same depth replaces the
 *     step list (not appends)
 *   - worker_done for an unknown step appends gracefully (race-safe)
 *   - architect spec re-emission replaces, not appends
 *   - multiple workers running at once (parallel work items)
 */
test.describe('swarm edge cases', () => {
  test.beforeEach(async ({ webview }) => {
    await connectShell(webview, { mode: 'swarm' });
  });

  test('replan replaces the step list rather than appending', async ({ webview }) => {
    await webview.push({ type: 'turn_start' });
    await webview.push({
      type: 'swarm_plan',
      depth: 0,
      steps: [
        { index: 1, description: 'original A' },
        { index: 2, description: 'original B' },
      ],
    });
    await expect(webview.page.locator('.swarm-step')).toHaveCount(2);

    // Agent decided to replan — same depth, different steps. The
    // upsert keyed by depth should replace, not stack a second card.
    await webview.push({
      type: 'swarm_plan',
      depth: 0,
      steps: [
        { index: 1, description: 'revised A' },
        { index: 2, description: 'revised B' },
        { index: 3, description: 'new step C' },
      ],
    });
    await expect(webview.page.locator('.swarm')).toHaveCount(1);
    await expect(webview.page.locator('.swarm-step')).toHaveCount(3);
    // Scope to the in-message card so we don't pick up the dock's
    // mirror of the same text — both show the current plan.
    await expect(webview.page.locator('.swarm').getByText('revised A')).toBeVisible();
    await expect(webview.page.locator('.swarm').getByText('original A')).toHaveCount(0);
  });

  test('worker_done for an unknown step appends a new row', async ({ webview }) => {
    // Race: worker_done can land before swarm_plan in some orderings
    // (the agent emits plan + immediately fires worker_done for a
    // pre-scheduled step). The reducer must accept it and create the
    // row rather than drop the event.
    await webview.push({ type: 'turn_start' });
    await webview.push({
      type: 'swarm_worker_done',
      depth: 0,
      stepIndex: 1,
      status: 'done',
      result: 'pre-staged',
    });
    await expect(webview.page.locator('.swarm-step')).toHaveCount(1);
    await expect(webview.page.locator('.swarm-step')).toContainText('pre-staged');
  });

  test('architect spec re-emission replaces, not duplicates', async ({ webview }) => {
    await webview.push({ type: 'turn_start' });
    await webview.push({ type: 'swarm_architect', depth: 0, spec: 'first version' });
    await webview.push({ type: 'swarm_architect', depth: 0, spec: 'revised version' });
    // Only one architect disclosure, with the latest content.
    await expect(webview.page.locator('.swarm-architect')).toHaveCount(1);
    await webview.page.locator('.swarm-architect summary').click();
    await expect(webview.page.locator('.swarm-architect pre')).toContainText('revised version');
    await expect(webview.page.locator('.swarm-architect pre')).not.toContainText('first version');
  });

  test('multiple workers running concurrently both show "running"', async ({ webview }) => {
    // The orchestrator can spawn workers for independent steps in
    // parallel. Both should display the running marker simultaneously
    // — no "only one step can be active" assumption.
    await webview.push({ type: 'turn_start' });
    await webview.push({
      type: 'swarm_plan',
      depth: 0,
      steps: [
        { index: 1, description: 'a' },
        { index: 2, description: 'b' },
        { index: 3, description: 'c' },
      ],
    });
    await webview.push({ type: 'swarm_worker_start', depth: 0, stepIndex: 1, description: 'a' });
    await webview.push({ type: 'swarm_worker_start', depth: 0, stepIndex: 2, description: 'b' });

    const steps = webview.page.locator('.swarm-step');
    await expect(steps.nth(0)).toHaveClass(/running/);
    await expect(steps.nth(1)).toHaveClass(/running/);
    await expect(steps.nth(2)).toHaveClass(/pending/);
  });

  test('worker_done with unknown status passes through and is visible', async ({ webview }) => {
    // The agent can emit statuses we don't style explicitly (skipped,
    // cancelled, etc.). The reducer must accept any string and the
    // text must be visible — even if uncoloured.
    await webview.push({ type: 'turn_start' });
    await webview.push({
      type: 'swarm_plan',
      depth: 0,
      steps: [{ index: 1, description: 'maybe' }],
    });
    await webview.push({
      type: 'swarm_worker_done',
      depth: 0,
      stepIndex: 1,
      status: 'skipped',
    });
    // The status is reflected on the step's class so future styling
    // can hook it without further reducer changes.
    await expect(webview.page.locator('.swarm-step.skipped')).toBeVisible();
  });
});
