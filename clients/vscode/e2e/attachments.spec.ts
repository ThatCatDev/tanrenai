import { test, expect, connectShell } from './fixtures';

/**
 * Selection attachments — adding code snippets from the active editor
 * to the next message. The "+ Sel" button posts attach_request so the
 * extension can read the active editor's selection; attach_selection
 * comes back the other way with the resolved snippet. Available-selection
 * is a live hint above the composer that lets users one-click attach
 * what they currently have selected.
 *
 * Image attachments use paste/drag and are tricky to drive in a browser
 * without the real ClipboardItem/DataTransfer plumbing — covered at
 * unit level (see App.test.tsx) rather than here.
 */
test.describe('selection attachments', () => {
  test.beforeEach(async ({ webview }) => {
    await connectShell(webview);
  });

  test('"+ Sel" button posts {type:"attach_request"}', async ({ webview }) => {
    await webview.clearSent();
    await webview.page.getByRole('button', { name: /\+ sel/i }).click();
    await expect.poll(() => webview.sentMessages()).toContainEqual({ type: 'attach_request' });
  });

  test('attach_selection adds a chip above the composer', async ({ webview }) => {
    const selection = {
      label: 'src/foo.ts:12-34',
      path: 'src/foo.ts',
      languageId: 'typescript',
      startLine: 12,
      endLine: 34,
      text: 'export function foo() {}',
    };
    await webview.push({ type: 'attach_selection', selection });
    // The chip is the visible cue that the next message will carry the
    // attachment — without it, users wouldn't know their selection is
    // queued.
    await expect(webview.page.getByText('src/foo.ts:12-34')).toBeVisible();
  });

  test('clicking the chip remove control clears that one attachment', async ({ webview }) => {
    const sel = (path: string) => ({
      label: `${path}:1-1`,
      path,
      languageId: 'typescript',
      startLine: 1,
      endLine: 1,
      text: 'x',
    });
    await webview.push({ type: 'attach_selection', selection: sel('a.ts') });
    await webview.push({ type: 'attach_selection', selection: sel('b.ts') });
    await expect(webview.page.getByText('a.ts:1-1')).toBeVisible();
    await expect(webview.page.getByText('b.ts:1-1')).toBeVisible();

    // Each chip is a `.chip`; the remove control is the inner `.chip-x`
    // button. Filter by text so we target the right one when there are
    // multiple chips in the composer.
    const removeOnA = webview.page
      .locator('.chip', { hasText: 'a.ts:1-1' })
      .locator('.chip-x');
    await removeOnA.click();
    await expect(webview.page.getByText('a.ts:1-1')).toHaveCount(0);
    await expect(webview.page.getByText('b.ts:1-1')).toBeVisible();
  });

  test('available_selection chip is shown when the editor has a selection', async ({
    webview,
  }) => {
    // available_selection is a host→webview hint about the *current*
    // editor selection — clicking it attaches without an extra round
    // trip. Distinct from the "+ Sel" button because it doesn't require
    // the user to drop focus on the chat first.
    await webview.push({
      type: 'available_selection',
      selection: {
        label: 'src/x.ts:5-7',
        path: 'src/x.ts',
        languageId: 'typescript',
        startLine: 5,
        endLine: 7,
        text: 'pick me',
      },
    });
    await expect(webview.page.getByText('src/x.ts:5-7')).toBeVisible();
  });

  test('available_selection hint disappears when set to null', async ({ webview }) => {
    await webview.push({
      type: 'available_selection',
      selection: {
        label: 'src/x.ts:5-7',
        path: 'src/x.ts',
        languageId: 'typescript',
        startLine: 5,
        endLine: 7,
        text: 'pick me',
      },
    });
    await expect(webview.page.getByText('src/x.ts:5-7')).toBeVisible();
    await webview.push({ type: 'available_selection', selection: null });
    await expect(webview.page.getByText('src/x.ts:5-7')).toHaveCount(0);
  });
});
