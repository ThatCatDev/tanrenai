// Thin HTTP client for the few platform endpoints the extension calls
// directly (separate from the agent-rpc subprocess). Used for instance
// control (stop / destroy / status) so the user can manage runaway GPU
// spawns even when agent-rpc itself isn't running.

import { loadCredentials } from './auth/credentials';

const USER_AGENT = 'tanrenai-vscode/0.1';

async function authedFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const creds = await loadCredentials();
  if (!creds?.server_url || !creds?.access_token) {
    throw new Error('Not signed in');
  }
  const url = creds.server_url.replace(/\/$/, '') + path;
  const headers = new Headers(init.headers);
  headers.set('Authorization', `Bearer ${creds.access_token}`);
  headers.set('User-Agent', USER_AGENT);
  if (init.body && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }

  return fetch(url, { ...init, headers });
}

export async function instanceStatus(): Promise<Record<string, unknown>> {
  const res = await authedFetch('/api/instance/status');
  if (!res.ok) {
    throw new Error(`instance/status returned ${res.status}: ${(await res.text()).slice(0, 200)}`);
  }

  return (await res.json()) as Record<string, unknown>;
}

export async function instanceStop(): Promise<void> {
  const res = await authedFetch('/api/instance/stop', { method: 'POST' });
  if (!res.ok) {
    throw new Error(`instance/stop returned ${res.status}: ${(await res.text()).slice(0, 200)}`);
  }
}

export async function instanceDestroy(): Promise<void> {
  const res = await authedFetch('/api/instance/destroy', { method: 'POST' });
  if (!res.ok) {
    throw new Error(`instance/destroy returned ${res.status}: ${(await res.text()).slice(0, 200)}`);
  }
}
