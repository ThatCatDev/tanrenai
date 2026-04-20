package cmd

import (
	"log/slog"

	"github.com/ThatCatDev/tanrenai/shared/apiclient"
)

// newAuthedClient constructs an apiclient.Client with the bearer token
// already set and the 401-retry refresh hook wired to the on-disk
// credentials file. Use this instead of raw apiclient.New whenever the
// client will talk to the remote platform — local-mode callers don't
// need auto-refresh and may bypass this.
//
// The refresh hook unconditionally exchanges the stored refresh token
// for a fresh access token and writes the rotated pair back to
// credentials.json. If that fails (network, revoked token, ...) the
// apiclient transport surfaces the original 401 to the caller.
func newAuthedClient(baseURL, token string) *apiclient.Client {
	c := apiclient.New(baseURL)
	if token != "" {
		c.SetAuthToken(token)
	}
	c.SetRefreshFunc(func() (string, error) {
		creds, err := loadCredentials()
		if err != nil {
			return "", err
		}
		fresh, err := refreshCredentials(creds)
		if err != nil {
			slog.Debug("mid-session token refresh failed", "error", err)
			return "", err
		}
		return fresh.AccessToken, nil
	})
	return c
}
