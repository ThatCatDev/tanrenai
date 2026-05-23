import { test, expect, connectShell } from './fixtures';

/**
 * Swarm mode UX. v2 emits structured swarm_plan / swarm_worker_start /
 * swarm_worker_done / swarm_verify events that the webview reduces into
 * a SwarmActivity entry per depth, rendered as a step list with live
 * status. These tests pin that contract — a regression that flattens
 * events back into content_delta strings (the v1 behavior) would fail
 * every assertion here.
 */
test.describe('swarm mode', () => {
  test.beforeEach(async ({ webview }) => {
    await connectShell(webview, { mode: 'swarm' });
  });

  test('placeholder + footer indicate swarm mode', async ({ webview }) => {
    await expect(webview.page.getByPlaceholder(/Brief the swarm/)).toBeVisible();
    await expect(webview.page.locator('.footer-mode')).toHaveText('swarm');
  });

  test('swarm_plan renders a step list with pending markers', async ({ webview }) => {
    await webview.push({ type: 'turn_start' });
    await webview.push({
      type: 'swarm_plan',
      depth: 0,
      steps: [
        { index: 1, description: 'Add /healthz route' },
        { index: 2, description: 'Add response model' },
        { index: 3, description: 'Write tests' },
      ],
    });

    // Three step rows, all in the pending state — the marker glyph is
    // the visible cue users use to scan progress.
    const steps = webview.page.locator('.swarm-step');
    await expect(steps).toHaveCount(3);
    await expect(steps.nth(0)).toContainText('Add /healthz route');
    await expect(steps.nth(2)).toContainText('Write tests');
    for (let i = 0; i < 3; i++) {
      await expect(steps.nth(i)).toHaveClass(/pending/);
    }
    // Header shows total + step word.
    await expect(webview.page.locator('.swarm-progress')).toHaveText('0/3 steps');
  });

  test('worker_start moves the step to running; worker_done marks it done', async ({
    webview,
  }) => {
    await webview.pushSequence([
      { type: 'turn_start' },
      {
        type: 'swarm_plan',
        depth: 0,
        steps: [
          { index: 1, description: 'first' },
          { index: 2, description: 'second' },
        ],
      },
      { type: 'swarm_worker_start', depth: 0, stepIndex: 1, description: 'first' },
    ]);

    const steps = webview.page.locator('.swarm-step');
    await expect(steps.nth(0)).toHaveClass(/running/);
    await expect(steps.nth(1)).toHaveClass(/pending/);

    await webview.push({
      type: 'swarm_worker_done',
      depth: 0,
      stepIndex: 1,
      status: 'done',
      result: 'wrote 32 bytes',
    });

    await expect(steps.nth(0)).toHaveClass(/done/);
    // Result text appears underneath the description.
    await expect(steps.nth(0).locator('.swarm-step-result')).toHaveText('wrote 32 bytes');
    // Progress counter advances — this is the at-a-glance scan surface.
    await expect(webview.page.locator('.swarm-progress')).toHaveText('1/2 steps');
  });

  test('worker_done with status=error surfaces the error message', async ({ webview }) => {
    await webview.pushSequence([
      { type: 'turn_start' },
      {
        type: 'swarm_plan',
        depth: 0,
        steps: [{ index: 1, description: 'risky thing' }],
      },
      { type: 'swarm_worker_start', depth: 0, stepIndex: 1, description: 'risky thing' },
      {
        type: 'swarm_worker_done',
        depth: 0,
        stepIndex: 1,
        status: 'error',
        error: 'subprocess exited 137',
      },
    ]);

    const step = webview.page.locator('.swarm-step').first();
    await expect(step).toHaveClass(/error/);
    await expect(step.locator('.swarm-step-error')).toHaveText('subprocess exited 137');
  });

  test('swarm_verify surfaces a verifying indicator without blocking', async ({ webview }) => {
    await webview.pushSequence([
      { type: 'turn_start' },
      {
        type: 'swarm_plan',
        depth: 0,
        steps: [{ index: 1, description: 'a' }],
      },
      {
        type: 'swarm_worker_done',
        depth: 0,
        stepIndex: 1,
        status: 'done',
      },
      { type: 'swarm_verify', depth: 0 },
    ]);

    await expect(webview.page.locator('.swarm-verify')).toBeVisible();
  });

  test('depth>0 renders as a separate card with a depth label', async ({ webview }) => {
    // Nested swarms (worker spawns a sub-orchestrator) get their own
    // depth — each depth lands as its own activity card so users can
    // see the hierarchy rather than one big merged list.
    await webview.pushSequence([
      { type: 'turn_start' },
      {
        type: 'swarm_plan',
        depth: 0,
        steps: [{ index: 1, description: 'outer task' }],
      },
      {
        type: 'swarm_plan',
        depth: 1,
        steps: [
          { index: 1, description: 'sub-task A' },
          { index: 2, description: 'sub-task B' },
        ],
      },
    ]);

    await expect(webview.page.locator('.swarm')).toHaveCount(2);
    // Only the depth>0 card surfaces the depth label — depth 0 is the
    // default and would be visual noise to label.
    await expect(webview.page.locator('.swarm-depth')).toHaveCount(1);
    await expect(webview.page.locator('.swarm-depth')).toHaveText('depth 1');
  });

  test('architect spec lives behind a collapsed disclosure', async ({ webview }) => {
    // Spec can be hundreds of lines of markdown — defaulting to expanded
    // would dominate the activity card and bury the steps below.
    await webview.pushSequence([
      { type: 'turn_start' },
      {
        type: 'swarm_architect',
        depth: 0,
        spec: '# Architecture\n- one\n- two',
      },
      {
        type: 'swarm_plan',
        depth: 0,
        steps: [{ index: 1, description: 'do it' }],
      },
    ]);

    const details = webview.page.locator('.swarm-architect');
    await expect(details).toBeVisible();
    // Spec text is hidden until the user opens the <details>.
    await expect(details.locator('pre')).toBeHidden();
    await details.locator('summary').click();
    await expect(details.locator('pre')).toContainText('Architecture');
  });
});
