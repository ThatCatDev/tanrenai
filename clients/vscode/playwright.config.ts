import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright config for the webview UX tests. Drives the browser-hosted
 * dev shell (`webview:dev`) — same Preact bundle that ships in the VS
 * Code extension, but mounted in plain Chromium with a stub for
 * `acquireVsCodeApi` so we can inject `WebviewOutbound` messages and
 * assert what the user sees. See webview/dev.html.
 *
 * Run: `npm run test:e2e` (auto-spawns the dev server via `webServer`).
 *      `npm run test:e2e -- --ui` for interactive mode.
 */
export default defineConfig({
  testDir: './e2e',
  timeout: 15_000,
  // Match the sidebar's effective layout width so wrap/spacing behave
  // like a real VS Code sidebar rather than a full desktop browser.
  use: {
    baseURL: 'http://127.0.0.1:5173',
    viewport: { width: 560, height: 900 },
    trace: 'on-first-retry',
  },
  // Force a single worker for now — the dev shell exposes a shared
  // `window.__sentMessages` global per page, and tests sometimes inspect
  // it. Parallel pages don't share state, but keeping it single-worker
  // also makes flaky-timing failures less noisy while we tune the suite.
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['list'], ['github']] : 'list',
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  ],
  webServer: {
    command: 'npm run webview:dev',
    url: 'http://127.0.0.1:5173',
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
    stdout: 'pipe',
    stderr: 'pipe',
  },
});
