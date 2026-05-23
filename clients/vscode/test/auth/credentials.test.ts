import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

// credentials.ts resolves the data dir from $TANRENAI_DATA_DIR — point it
// at a per-test tmpdir so the tests never touch the user's real file.
let originalDataDir: string | undefined;
let tmpdir: string;

beforeEach(async () => {
  originalDataDir = process.env.TANRENAI_DATA_DIR;
  tmpdir = await fs.mkdtemp(path.join(os.tmpdir(), 'tanrenai-creds-'));
  process.env.TANRENAI_DATA_DIR = tmpdir;
});

afterEach(async () => {
  if (originalDataDir === undefined) {
    delete process.env.TANRENAI_DATA_DIR;
  } else {
    process.env.TANRENAI_DATA_DIR = originalDataDir;
  }
  await fs.rm(tmpdir, { recursive: true, force: true });
});

describe('credentials', () => {
  it('returns null when no credentials file exists', async () => {
    const { loadCredentials } = await import('../../src/auth/credentials');
    expect(await loadCredentials()).toBeNull();
  });

  it('round-trips save → load', async () => {
    const { saveCredentials, loadCredentials } = await import('../../src/auth/credentials');
    const creds = {
      server_url: 'https://api.example.com',
      access_token: 'tok',
      refresh_token: 'ref',
      expires_at: '2030-01-01T00:00:00Z',
    };
    await saveCredentials(creds);
    expect(await loadCredentials()).toEqual(creds);
  });

  it('writes the file with restrictive permissions', async () => {
    const { saveCredentials, credentialsPath } = await import('../../src/auth/credentials');
    await saveCredentials({ server_url: 'x', access_token: 't' });
    const stat = await fs.stat(credentialsPath());
    // 0o600 is rw for owner only; on Windows mode bits aren't meaningful,
    // so this assertion only runs on POSIX.
    if (process.platform !== 'win32') {
      expect(stat.mode & 0o777).toBe(0o600);
    }
  });

  it('deleteCredentials removes the file and is safe to call when missing', async () => {
    const { saveCredentials, deleteCredentials, loadCredentials } = await import(
      '../../src/auth/credentials'
    );
    await saveCredentials({ server_url: 'x', access_token: 't' });
    await deleteCredentials();
    expect(await loadCredentials()).toBeNull();
    // Calling again on an absent file should not throw.
    await expect(deleteCredentials()).resolves.toBeUndefined();
  });

  it('isExpired respects the leeway margin', async () => {
    const { isExpired } = await import('../../src/auth/credentials');
    const now = Date.now();
    expect(
      isExpired({
        server_url: 'x',
        access_token: 't',
        expires_at: new Date(now + 30_000).toISOString(),
      }),
    ).toBe(true); // within default 60s leeway
    expect(
      isExpired({
        server_url: 'x',
        access_token: 't',
        expires_at: new Date(now + 5 * 60_000).toISOString(),
      }),
    ).toBe(false);
  });

  it('isExpired returns false when expires_at is missing or unparseable', async () => {
    const { isExpired } = await import('../../src/auth/credentials');
    expect(isExpired({ server_url: 'x', access_token: 't' })).toBe(false);
    expect(isExpired({ server_url: 'x', access_token: 't', expires_at: 'not-a-date' })).toBe(false);
  });
});
