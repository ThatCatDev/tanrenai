import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { resolveCliPath } from '../../src/rpc/cliPath';

// The bundled binary lives at dist/bin/<process.platform>-<process.arch>/.
const targetDir = `${process.platform}-${process.arch}`;
const binName = process.platform === 'win32' ? 'tanrenai.exe' : 'tanrenai';

describe('resolveCliPath', () => {
  let root: string;

  beforeEach(() => {
    root = fs.mkdtempSync(path.join(os.tmpdir(), 'tanrenai-clipath-'));
  });

  afterEach(() => {
    fs.rmSync(root, { recursive: true, force: true });
  });

  it('returns the explicit override when set, ignoring everything else', () => {
    expect(resolveCliPath(root, '/custom/tanrenai')).toBe('/custom/tanrenai');
  });

  it('falls back to PATH lookup when no binary is bundled', () => {
    expect(resolveCliPath(root, '')).toBe('tanrenai');
  });

  it('prefers the bundled binary over PATH when present', () => {
    const dir = path.join(root, 'dist', 'bin', targetDir);
    fs.mkdirSync(dir, { recursive: true });
    const bundled = path.join(dir, binName);
    fs.writeFileSync(bundled, '#!/bin/sh\n');

    expect(resolveCliPath(root, '')).toBe(bundled);
  });

  it('restores the executable bit on the bundled binary (non-Windows)', () => {
    if (process.platform === 'win32') return; // exec bit is a no-op on Windows
    const dir = path.join(root, 'dist', 'bin', targetDir);
    fs.mkdirSync(dir, { recursive: true });
    const bundled = path.join(dir, binName);
    // Simulate a VSIX extraction that dropped the +x bit.
    fs.writeFileSync(bundled, '#!/bin/sh\n', { mode: 0o644 });

    resolveCliPath(root, '');

    // owner-execute bit should now be set
    expect(fs.statSync(bundled).mode & 0o100).toBe(0o100);
  });
});
