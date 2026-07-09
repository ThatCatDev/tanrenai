/**
 * Classification helpers for CLI subprocess failures, kept free of vscode
 * imports so they can be unit-tested. The controller uses these to turn the
 * raw startup-failure soup (generic "exited before ready" + stderr tail)
 * into something actionable.
 */

/** Shown in the chat view when the stored credentials are dead. */
export const SESSION_EXPIRED_MESSAGE =
  'Your session has expired. Sign in to Tanrenai again to reconnect.';

/**
 * Does this error text indicate dead/expired credentials (as opposed to a
 * backend or network problem)? Matches the CLI's own phrasing: the
 * refresh-failure warning ("Invalid Refresh Token: Already Used"), the
 * session-expired error ("session expired — run `tanrenai login`"), and the
 * post-retry 401 from model load ("invalid or expired token (status 401)").
 */
export function looksLikeAuthError(s: string | undefined): boolean {
  if (!s) {
    return false;
  }
  const lower = s.toLowerCase();

  return (
    lower.includes('session expired') ||
    lower.includes('session has expired') ||
    lower.includes('invalid refresh token') ||
    lower.includes('invalid or expired token') ||
    lower.includes('authentication failed') ||
    lower.includes('status 401')
  );
}
