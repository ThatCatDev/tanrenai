import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';

/**
 * Wire-compatible with the Go CLI's credentials.json (clients/cli/cmd/credentials.go).
 * JSON field names are snake_case so a file written by either side can be read by the other.
 */
export interface Credentials {
  server_url: string;
  access_token: string;
  refresh_token?: string;
  expires_at?: string; // ISO-8601 timestamp; matches Go's time.Time JSON encoding
}

/** Resolves the data directory the CLI uses for credentials and models. */
export function tanrenaiDataDir(): string {
  const override = process.env.TANRENAI_DATA_DIR;
  if (override) {
    return override;
  }
  if (process.platform === 'win32') {
    const localAppData = process.env.LOCALAPPDATA;
    if (localAppData) {
      return path.join(localAppData, 'tanrenai');
    }
    // Fallback when LOCALAPPDATA is unset (rare).
    return path.join(os.homedir(), 'AppData', 'Local', 'tanrenai');
  }

  return path.join(os.homedir(), '.local', 'share', 'tanrenai');
}

export function credentialsPath(): string {
  return path.join(tanrenaiDataDir(), 'credentials.json');
}

export async function loadCredentials(): Promise<Credentials | null> {
  try {
    const data = await fs.readFile(credentialsPath(), 'utf8');

    return JSON.parse(data) as Credentials;
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
      return null;
    }
    throw err;
  }
}

export async function saveCredentials(creds: Credentials): Promise<void> {
  const dir = path.dirname(credentialsPath());
  await fs.mkdir(dir, { recursive: true, mode: 0o700 });
  await fs.writeFile(credentialsPath(), JSON.stringify(creds, null, 2), { mode: 0o600 });
}

export async function deleteCredentials(): Promise<void> {
  try {
    await fs.unlink(credentialsPath());
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== 'ENOENT') {
      throw err;
    }
  }
}

/** Returns true if credentials.expires_at has passed (or is within `marginMs` of passing). */
export function isExpired(creds: Credentials, marginMs = 60_000): boolean {
  if (!creds.expires_at) {
    return false;
  }
  const expiresMs = new Date(creds.expires_at).getTime();
  if (Number.isNaN(expiresMs)) {
    return false;
  }

  return expiresMs - Date.now() < marginMs;
}
