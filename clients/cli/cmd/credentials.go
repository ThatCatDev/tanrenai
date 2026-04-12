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
func saveCredentials(creds *Credentials) error {
	dir := filepath.Dir(credentialsPath())
	if err := os.MkdirAll(dir, 0700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(creds, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(credentialsPath(), data, 0600)
}

// deleteCredentials removes the credentials file.
func deleteCredentials() error {
	return os.Remove(credentialsPath())
}
