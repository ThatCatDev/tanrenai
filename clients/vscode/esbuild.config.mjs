import * as esbuild from 'esbuild';

const watch = process.argv.includes('--watch');

// Two bundles:
//   1. extension.js — runs in the VS Code extension host (Node).
//   2. webview.js   — runs in the sidebar webview iframe (browser, Preact).
const configs = [
  {
    entryPoints: ['src/extension.ts'],
    bundle: true,
    format: 'cjs',
    platform: 'node',
    target: 'node20',
    outfile: 'dist/extension.js',
    external: ['vscode'],
    sourcemap: true,
    minify: false,
    logLevel: 'info',
  },
  {
    entryPoints: ['webview/main.tsx'],
    bundle: true,
    format: 'iife',
    platform: 'browser',
    target: 'es2020',
    outfile: 'dist/webview.js',
    sourcemap: true,
    minify: !watch,
    jsx: 'automatic',
    jsxImportSource: 'preact',
    logLevel: 'info',
  },
];

const contexts = await Promise.all(configs.map((c) => esbuild.context(c)));

if (watch) {
  await Promise.all(contexts.map((c) => c.watch()));
  console.log('[esbuild] watching extension + webview…');
} else {
  await Promise.all(contexts.map((c) => c.rebuild()));
  await Promise.all(contexts.map((c) => c.dispose()));
}
