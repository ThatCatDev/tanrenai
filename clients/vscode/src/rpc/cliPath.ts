import * as fs from 'node:fs';
import * as path from 'node:path';

/**
 * Resolves the path to the tanrenai CLI binary in priority order:
 *   1. Explicit override (from settings).
 *   2. Bundled binary at <extensionRoot>/dist/bin/<platform>-<arch>/tanrenai[.exe].
 *   3. `tanrenai` from PATH (Node will resolve this when we spawn).
 */
export function resolveCliPath(extensionRoot: string, override: string): string {
  if (override) {
    return override;
  }
  const platform = process.platform; // "linux" | "darwin" | "win32"
  const arch = process.arch; // "x64" | "arm64" | ...
  const ext = platform === 'win32' ? '.exe' : '';
  const bundled = path.join(extensionRoot, 'dist', 'bin', `${platform}-${arch}`, `tanrenai${ext}`);
  if (fs.existsSync(bundled)) {
    // A .vsix is a zip, and extraction can drop the executable bit on
    // Unix — spawning would then fail with EACCES. Restore it (best-effort).
    if (platform !== 'win32') {
      try {
        fs.chmodSync(bundled, 0o755);
      } catch {
        /* non-fatal: fall through and try to spawn it anyway */
      }
    }

    return bundled;
  }

  return 'tanrenai';
}
