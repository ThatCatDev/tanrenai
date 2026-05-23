import { test, expect, connectShell } from './fixtures';

test.describe('streaming a turn', () => {
  test.beforeEach(async ({ webview }) => {
    await connectShell(webview, { mode: 'agent' });
  });

  test('renders user + reasoning + content into separate bubbles', async ({ webview }) => {
    await webview.pushSequence([
      { type: 'message_start', role: 'user', id: 'u1' },
      { type: 'message_delta', id: 'u1', text: 'explain X' },
      { type: 'message_end', id: 'u1' },
      { type: 'turn_start' },
      { type: 'message_start', role: 'assistant', id: 'a1' },
      { type: 'message_delta', id: 'a1', text: 'thinking about it', channel: 'reasoning' },
      { type: 'message_delta', id: 'a1', text: 'X is foo and bar.', channel: 'content' },
      { type: 'message_end', id: 'a1' },
      { type: 'turn_end', ok: true },
    ]);

    // User bubble, reasoning bubble, content bubble — distinct entries
    // with distinct headers ("You", "Thinking", "Tanrenai") and
    // distinct content. A regression that collapsed channels would fail
    // the reasoning-text assertion.
    await expect(webview.page.getByText('explain X')).toBeVisible();
    await expect(webview.page.getByText('Thinking', { exact: true })).toBeVisible();
    await expect(webview.page.getByText('thinking about it')).toBeVisible();
    await expect(webview.page.getByText('X is foo and bar.')).toBeVisible();
  });

  test('footer t/s readout appears after token_rate event', async ({ webview }) => {
    // Token rate is the meter users watch when comparing model speeds /
    // diagnosing memory pressure. Footer must surface it AS SOON AS a
    // token_rate event arrives — not wait for turn_end.
    await webview.push({ type: 'turn_start' });
    await webview.push({ type: 'message_start', role: 'assistant', id: 'a1' });
    await webview.push({ type: 'message_delta', id: 'a1', text: 'hi ', channel: 'content' });
    await webview.push({ type: 'token_rate', tokens: 8, tps: 17.4 });

    // Footer renders the value rounded — `tps.toFixed(0)` → "17".
    await expect(webview.page.locator('.footer-rate')).toHaveText('17 t/s');
    // Tooltip surfaces the raw token count so users can sanity-check the
    // sample size — short responses with absurd rates get hidden by the
    // tracker's ≥2-token / ≥100ms guard, so any visible rate is valid.
    await expect(webview.page.locator('.footer-rate')).toHaveAttribute(
      'title',
      /8 tokens generated/,
    );
  });

  test('footer t/s clears on the next turn_start', async ({ webview }) => {
    // Per-turn isolation: a stale "67 t/s" sitting in the footer while a
    // new turn is mid-prompt-eval would mislead the user about what's
    // happening NOW. Reducer clears tokenRate to null on turn_start.
    await webview.push({ type: 'turn_start' });
    await webview.push({ type: 'token_rate', tokens: 30, tps: 67.2 });
    await expect(webview.page.locator('.footer-rate')).toBeVisible();
    await webview.push({ type: 'turn_end', ok: true });
    // The final value stays visible after turn_end — that's intentional;
    // users want to see the result of the turn they just watched.
    await expect(webview.page.locator('.footer-rate')).toBeVisible();
    // But a fresh turn must clear it before the new rate populates.
    await webview.push({ type: 'turn_start' });
    await expect(webview.page.locator('.footer-rate')).toHaveCount(0);
  });

  test('error in the turn renders an error entry', async ({ webview }) => {
    await webview.push({ type: 'turn_start' });
    await webview.push({ type: 'turn_end', ok: false, reason: 'model load failed' });
    await expect(webview.page.getByText('model load failed')).toBeVisible();
  });
});
