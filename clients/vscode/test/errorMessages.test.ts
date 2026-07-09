import { describe, expect, it } from 'vitest';
import { looksLikeAuthError } from '../src/errorMessages';

describe('looksLikeAuthError', () => {
  it('matches the CLI refresh-failure warning', () => {
    expect(
      looksLikeAuthError(
        'warning: token refresh failed: refresh failed (400): Invalid Refresh Token: Already Used (continuing with existing token)',
      ),
    ).toBe(true);
  });

  it('matches the post-retry 401 from model load', () => {
    expect(
      looksLikeAuthError(
        'Error: failed to load model (is the backend running?): invalid or expired token (status 401)',
      ),
    ).toBe(true);
  });

  it('matches the CLI session-expired error', () => {
    expect(
      looksLikeAuthError('setup failed: authentication failed — your session has expired; run `tanrenai login` to sign in again'),
    ).toBe(true);
  });

  it('does not match backend/network failures', () => {
    expect(looksLikeAuthError('failed to load model (is the backend running?): server unavailable')).toBe(false);
    expect(looksLikeAuthError('connect ECONNREFUSED 127.0.0.1:8080')).toBe(false);
    expect(looksLikeAuthError('tanrenai agent-rpc exited (exit code 1) before ready')).toBe(false);
  });

  it('handles undefined and empty input', () => {
    expect(looksLikeAuthError(undefined)).toBe(false);
    expect(looksLikeAuthError('')).toBe(false);
  });
});
