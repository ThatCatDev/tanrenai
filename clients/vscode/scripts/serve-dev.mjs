// Standalone dev server for the Preact webview. Builds the same bundle
// VS Code ships, but serves it from a plain HTML page with a stub for
// `acquireVsCodeApi` so it can run in a regular browser — fast iteration
// loop and a fixed URL Playwright can attach to without launching the
// Electron host.
//
// Usage: npm run webview:dev → http://127.0.0.1:5173

import * as esbuild from 'esbuild';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const outDir = path.join(root, 'dist-dev');

const HOST = process.env.WEBVIEW_DEV_HOST || '127.0.0.1';
const PORT = Number(process.env.WEBVIEW_DEV_PORT || 5173);

fs.mkdirSync(outDir, { recursive: true });

// Mirror static files into the served dir. Called once at startup and
// re-called on every change to either source so the page stays current
// without a server restart.
function copyStatic() {
  fs.copyFileSync(path.join(root, 'webview/dev.html'), path.join(outDir, 'index.html'));
  fs.copyFileSync(path.join(root, 'media/chat.css'), path.join(outDir, 'chat.css'));
}
copyStatic();

// Watch the static files for changes too — without this, edits to
// dev.html or chat.css would only show up after a server restart.
// fs.watch is event-driven but unreliable on macOS (misses some edits
// silently); fs.watchFile is polling but reliable. Use the latter at a
// modest interval — these files are tiny so the poll cost is nothing,
// and the alternative is "wait, my edit didn't apply" debugging.
const watched = [
  { src: path.join(root, 'webview/dev.html'), dst: path.join(outDir, 'index.html') },
  { src: path.join(root, 'media/chat.css'), dst: path.join(outDir, 'chat.css') },
];
for (const { src } of watched) {
  fs.watchFile(src, { interval: 300 }, copyStatic);
}

const ctx = await esbuild.context({
  entryPoints: [path.join(root, 'webview/main.tsx')],
  bundle: true,
  format: 'iife',
  platform: 'browser',
  target: 'es2020',
  outfile: path.join(outDir, 'webview.js'),
  sourcemap: true,
  // Keep it readable so debugger stepping isn't useless.
  minify: false,
  jsx: 'automatic',
  jsxImportSource: 'preact',
  logLevel: 'info',
});

await ctx.watch();

const { host, port } = await ctx.serve({
  servedir: outDir,
  host: HOST,
  port: PORT,
});

const url = `http://${host}:${port}`;
console.log('');
console.log(`tanrenai webview dev shell:  ${url}`);
console.log('  Auto-connects on load; append ?nostate=1 to skip.');
console.log('  Inject:   window.__pushMsg({ type: "turn_start" })');
console.log('  Observe:  window.__sentMessages');
console.log('  Reload:   Cmd/Ctrl-R (sources rebuild on save)');
console.log('');
