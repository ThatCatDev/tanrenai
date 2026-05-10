import * as http from 'node:http';
import * as vscode from 'vscode';
import { Credentials, saveCredentials } from './credentials';

// Mirrors clients/cli/cmd/login.go:callbackPort. The frontend redirects here
// with the access token as a query string after the user completes sign-in.
const CALLBACK_PORT = 18293;
const LOGIN_TIMEOUT_MS = 5 * 60 * 1000;

export interface LoginOptions {
  /** Base URL of the tanrenai web frontend, e.g. https://dev.tanrenai.com */
  webUrl: string;
  /** Backend URL the credentials will record. */
  serverUrl: string;
  /** Optional logger so the controller can surface what's happening. */
  log?: (msg: string) => void;
}

/**
 * Run the browser-based login flow:
 *   1. Open a localhost HTTP server on CALLBACK_PORT.
 *   2. Open the user's browser to {webUrl}/cli-login?callback=...
 *   3. Wait for the redirect carrying access_token (and optionally
 *      refresh_token, expires_at).
 *   4. Persist the credentials to disk in the format the CLI uses.
 */
export async function runLoginFlow(opts: LoginOptions): Promise<Credentials> {
  const callbackUrl = `http://localhost:${CALLBACK_PORT}/callback`;
  const loginUrl =
    opts.webUrl.replace(/\/$/, '') +
    `/cli-login?callback=${encodeURIComponent(callbackUrl)}`;
  const log = opts.log ?? (() => {});

  return new Promise<Credentials>((resolve, reject) => {
    let timeoutHandle: NodeJS.Timeout | undefined;

    const server = http.createServer((req, res) => {
      log(`callback hit: ${req.method} ${req.url}`);
      const url = new URL(req.url ?? '/', `http://localhost:${CALLBACK_PORT}`);
      if (url.pathname !== '/callback') {
        log(`  → 404 (wrong path: ${url.pathname})`);
        res.statusCode = 404;
        res.end('Not found');

        return;
      }

      const errParam = url.searchParams.get('error');
      if (errParam) {
        const desc = url.searchParams.get('error_description') ?? '';
        log(`  → error from web: ${errParam} ${desc}`);
        res.statusCode = 400;
        res.end(`Login failed: ${errParam} ${desc}`);
        cleanup();
        reject(new Error(`Login failed: ${errParam} ${desc}`));

        return;
      }

      const accessToken = url.searchParams.get('access_token');
      log(`  → params: ${[...url.searchParams.keys()].join(',') || '(none)'}`);
      if (!accessToken) {
        // OAuth "implicit" flows put the token in a URL fragment, which the
        // browser doesn't send to the server. Serve a tiny page that
        // re-submits with the fragment turned into a query string. The
        // second hit lands here with proper params and resolves the flow.
        log('  → no access_token in query — serving fragment-rescue page');
        res.statusCode = 200;
        res.setHeader('Content-Type', 'text/html; charset=utf-8');
        res.end(`<!doctype html><meta charset=utf-8><title>Tanrenai</title>
<body style="font-family:system-ui;padding:3rem;background:#0f1419;color:#e6e6e6">
<h2>Finishing sign-in…</h2>
<p id=msg>If this doesn't auto-redirect, paste the callback URL into the
terminal where you ran <code>tanrenai login</code>.</p>
<script>
  if (window.location.hash && window.location.hash.length > 1) {
    var qs = window.location.hash.replace(/^#/, '?');
    window.location.replace(window.location.pathname + qs);
  } else {
    document.getElementById('msg').textContent =
      'No access_token in the URL. The web app did not include sign-in tokens — please retry.';
  }
</script></body>`);

        return;
      }

      const refreshToken = url.searchParams.get('refresh_token') ?? undefined;
      const expiresAt = parseExpiry(url.searchParams);

      const creds: Credentials = {
        server_url: opts.serverUrl,
        access_token: accessToken,
        refresh_token: refreshToken,
        expires_at: expiresAt,
      };

      saveCredentials(creds)
        .then(() => {
          res.setHeader('Content-Type', 'text/html; charset=utf-8');
          res.end(
            `<!doctype html><meta charset=utf-8><title>Tanrenai login</title>` +
              `<body style="font-family:system-ui;padding:3rem;background:#0f1419;color:#e6e6e6">` +
              `<h2>Signed in.</h2><p>You can close this tab.</p></body>`,
          );
          cleanup();
          resolve(creds);
        })
        .catch((err) => {
          res.statusCode = 500;
          res.end('Could not persist credentials: ' + (err as Error).message);
          cleanup();
          reject(err);
        });
    });

    function cleanup(): void {
      if (timeoutHandle) {
        clearTimeout(timeoutHandle);
      }
      server.close();
    }

    server.on('error', (err) => {
      cleanup();
      reject(err);
    });

    server.listen(CALLBACK_PORT, '127.0.0.1', () => {
      log(`listening on 127.0.0.1:${CALLBACK_PORT}, waiting for callback`);
      log(`opening browser → ${loginUrl}`);
      timeoutHandle = setTimeout(() => {
        log('timed out after 5 min — the web app likely did not redirect');
        cleanup();
        reject(new Error('Login timed out — no callback received in 5 minutes'));
      }, LOGIN_TIMEOUT_MS);

      vscode.env.openExternal(vscode.Uri.parse(loginUrl)).then(undefined, () => {
        log('vscode.env.openExternal failed — falling back to manual notification');
        void vscode.window.showInformationMessage(
          `Open this URL to sign in: ${loginUrl}`,
        );
      });
    });
  });
}

function parseExpiry(params: URLSearchParams): string | undefined {
  // Prefer absolute timestamp; fall back to expires_in seconds-from-now.
  const expiresAt = params.get('expires_at');
  if (expiresAt) {
    const seconds = Number.parseInt(expiresAt, 10);
    if (!Number.isNaN(seconds)) {
      return new Date(seconds * 1000).toISOString();
    }
  }
  const expiresIn = params.get('expires_in');
  if (expiresIn) {
    const seconds = Number.parseInt(expiresIn, 10);
    if (!Number.isNaN(seconds)) {
      return new Date(Date.now() + seconds * 1000).toISOString();
    }
  }

  return undefined;
}
