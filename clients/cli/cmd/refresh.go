package cmd

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// refreshLeeway is how far ahead of expiry we refresh, to avoid losing
// access mid-request.
const refreshLeeway = time.Minute

// refreshResponse mirrors what POST /api/auth/refresh returns — the
// platform's pass-through of Supabase's refresh endpoint.
type refreshResponse struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	ExpiresAt    int64  `json:"expires_at"`
	ExpiresIn    int    `json:"expires_in"`
	ErrorDesc    string `json:"error_description"`
	Msg          string `json:"msg"`
}

// maybeRefreshCredentials looks at the stored credentials and, if the
// access token is expired (or within refreshLeeway of expiring), POSTs
// the refresh token to the platform's /api/auth/refresh and writes the
// result back to disk. Returns the (possibly updated) credentials.
//
// If no credentials are stored, returns (nil, nil).
// If refresh fails, returns the original creds plus an error — callers
// can choose to continue with the stale token (it may still work) or
// bail.
func maybeRefreshCredentials(creds *Credentials) (*Credentials, error) {
	if creds == nil {
		return nil, nil
	}
	if creds.RefreshToken == "" {
		return creds, nil
	}
	if !creds.ExpiresAt.IsZero() && time.Now().Before(creds.ExpiresAt.Add(-refreshLeeway)) {
		return creds, nil
	}
	return refreshCredentials(creds)
}

// refreshCredentials unconditionally exchanges the stored refresh token
// for a new access token via the platform's /api/auth/refresh proxy and
// writes the result back to disk. Called from the proactive startup path
// (via maybeRefreshCredentials, when near expiry) and reactively from
// the apiclient's 401-retry transport during long-running sessions.
func refreshCredentials(creds *Credentials) (*Credentials, error) {
	if creds == nil || creds.RefreshToken == "" {
		return creds, fmt.Errorf("no refresh token available")
	}
	if creds.ServerURL == "" {
		return creds, fmt.Errorf("credentials missing server URL, can't refresh")
	}

	body, _ := json.Marshal(map[string]string{"refresh_token": creds.RefreshToken})
	req, err := http.NewRequest(http.MethodPost, strings.TrimRight(creds.ServerURL, "/")+"/api/auth/refresh", bytes.NewReader(body))
	if err != nil {
		return creds, err
	}
	req.Header.Set("Content-Type", "application/json")
	// Non-default UA so Cloudflare et al. don't reject the request as a
	// bot before it reaches the platform.
	req.Header.Set("User-Agent", "tanrenai-cli/0.1")

	client := &http.Client{Timeout: 20 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return creds, fmt.Errorf("refresh request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, _ := io.ReadAll(resp.Body)

	var rr refreshResponse
	_ = json.Unmarshal(respBody, &rr)

	if resp.StatusCode != http.StatusOK {
		msg := rr.ErrorDesc
		if msg == "" {
			msg = rr.Msg
		}
		if msg == "" {
			msg = string(respBody)
		}
		return creds, fmt.Errorf("refresh failed (%d): %s", resp.StatusCode, msg)
	}

	if rr.AccessToken == "" {
		return creds, fmt.Errorf("refresh response missing access_token")
	}

	var expiresAt time.Time
	if rr.ExpiresAt > 0 {
		expiresAt = time.Unix(rr.ExpiresAt, 0)
	} else if rr.ExpiresIn > 0 {
		expiresAt = time.Now().Add(time.Duration(rr.ExpiresIn) * time.Second)
	}

	newRefresh := rr.RefreshToken
	if newRefresh == "" {
		newRefresh = creds.RefreshToken
	}

	fresh := &Credentials{
		ServerURL:    creds.ServerURL,
		AccessToken:  rr.AccessToken,
		RefreshToken: newRefresh,
		ExpiresAt:    expiresAt,
	}
	if err := saveCredentials(fresh); err != nil {
		return creds, fmt.Errorf("save refreshed credentials: %w", err)
	}
	return fresh, nil
}
