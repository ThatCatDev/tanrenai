package cmd

import (
	"encoding/json"
	"os"
	"path/filepath"
	"time"
)

// Credentials holds the stored authentication tokens.
type Credentials struct {
	ServerURL    string    `json:"server_url"`
	AccessToken  string    `json:"access_token"`
	RefreshToken string    `json:"refresh_token,omitempty"`
	ExpiresAt    time.Time `json:"expires_at,omitempty"`
}

func credentialsPath() string {
	dir := tanrenaiDataDir()
	return filepath.Join(dir, "credentials.json")
}

// loadCredentials reads stored credentials from disk.
func loadCredentials() (*Credentials, error) {
	data, err := os.ReadFile(credentialsPath())
	if err != nil {
		return nil, err
	}
	var creds Credentials
	if err := json.Unmarshal(data, &creds); err != nil {
		return nil, err
	}
	return &creds, nil
}

// saveCredentials writes credentials to disk with restricted permissions.
// The write is atomic (temp file + rename) so a concurrent reader never
// sees a truncated file and a crash mid-write never clobbers the previous
// (still valid) pair.
func saveCredentials(creds *Credentials) error {
	path := credentialsPath()
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(creds, "", "  ")
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".credentials-*.json")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath) // no-op after a successful rename
	if err := tmp.Chmod(0600); err != nil {
		_ = tmp.Close()
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
}

// lockCredentials takes an exclusive advisory lock serializing the
// credential refresh read-modify-write across processes (CLI sessions,
// editor-spawned agent-rpc instances). The platform rotates the refresh
// token on every use, so two processes racing to refresh the same token
// means the loser gets "Invalid Refresh Token: Already Used" — and can get
// the whole token family revoked. Blocks until the lock is available;
// returns an unlock func. Best effort: callers may proceed unlocked on
// error (the lock reduces a race, it doesn't guard correctness of a
// single process).
func lockCredentials() (func(), error) {
	dir := filepath.Dir(credentialsPath())
	if err := os.MkdirAll(dir, 0700); err != nil {
		return nil, err
	}
	f, err := os.OpenFile(filepath.Join(dir, "credentials.lock"), os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		return nil, err
	}
	if err := flockExclusive(f); err != nil {
		_ = f.Close()
		return nil, err
	}
	return func() {
		_ = flockUnlock(f)
		_ = f.Close()
	}, nil
}

// deleteCredentials removes the credentials file.
func deleteCredentials() error {
	return os.Remove(credentialsPath())
}
