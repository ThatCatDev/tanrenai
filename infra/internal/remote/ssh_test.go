package remote

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestIsAuthError(t *testing.T) {
	tests := []struct {
		err  error
		want bool
	}{
		{nil, false},
		{errors.New("some random error"), false},
		{errors.New("connection refused"), false},
		{errors.New("unable to authenticate, attempted methods [none publickey]"), true},
		{errors.New("ssh: handshake failed: ssh: unable to authenticate"), true},
		{errors.New("ssh: no supported methods remain"), true},
		{errors.New("handshake failed: connection reset"), true},
	}

	for _, tc := range tests {
		got := isAuthError(tc.err)
		if got != tc.want {
			errMsg := "<nil>"
			if tc.err != nil {
				errMsg = tc.err.Error()
			}
			t.Errorf("isAuthError(%q) = %v, want %v", errMsg, got, tc.want)
		}
	}
}

func TestIsAuthErrorNilIsNotAuth(t *testing.T) {
	if isAuthError(nil) {
		t.Error("isAuthError(nil) should return false")
	}
}

func TestLoadSSHKeyNotFound(t *testing.T) {
	// Override HOME to a temp dir with no .ssh keys
	tmpDir := t.TempDir()
	orig := os.Getenv("HOME")
	os.Setenv("HOME", tmpDir)
	defer os.Setenv("HOME", orig)

	_, err := loadSSHKey()
	if err == nil {
		t.Error("expected error when no SSH keys exist")
	}
	if err.Error() == "" {
		t.Error("error message should not be empty")
	}
}

func TestLoadSSHKeyFindsEd25519(t *testing.T) {
	tmpDir := t.TempDir()
	sshDir := filepath.Join(tmpDir, ".ssh")
	if err := os.MkdirAll(sshDir, 0700); err != nil {
		t.Fatalf("failed to create .ssh dir: %v", err)
	}

	keyData := []byte("fake-ssh-key-data")
	keyPath := filepath.Join(sshDir, "id_ed25519")
	if err := os.WriteFile(keyPath, keyData, 0600); err != nil {
		t.Fatalf("failed to write key file: %v", err)
	}

	orig := os.Getenv("HOME")
	os.Setenv("HOME", tmpDir)
	defer os.Setenv("HOME", orig)

	data, err := loadSSHKey()
	if err != nil {
		t.Fatalf("loadSSHKey() error: %v", err)
	}
	if string(data) != "fake-ssh-key-data" {
		t.Errorf("loadSSHKey() = %q, want \"fake-ssh-key-data\"", string(data))
	}
}

func TestLoadSSHKeyFindsRSAWhenNoEd25519(t *testing.T) {
	tmpDir := t.TempDir()
	sshDir := filepath.Join(tmpDir, ".ssh")
	if err := os.MkdirAll(sshDir, 0700); err != nil {
		t.Fatalf("failed to create .ssh dir: %v", err)
	}

	keyData := []byte("fake-rsa-key-data")
	keyPath := filepath.Join(sshDir, "id_rsa")
	if err := os.WriteFile(keyPath, keyData, 0600); err != nil {
		t.Fatalf("failed to write key file: %v", err)
	}

	orig := os.Getenv("HOME")
	os.Setenv("HOME", tmpDir)
	defer os.Setenv("HOME", orig)

	data, err := loadSSHKey()
	if err != nil {
		t.Fatalf("loadSSHKey() error: %v", err)
	}
	if string(data) != "fake-rsa-key-data" {
		t.Errorf("loadSSHKey() = %q, want \"fake-rsa-key-data\"", string(data))
	}
}

func TestLoadSSHKeyPrefersEd25519OverRSA(t *testing.T) {
	tmpDir := t.TempDir()
	sshDir := filepath.Join(tmpDir, ".ssh")
	if err := os.MkdirAll(sshDir, 0700); err != nil {
		t.Fatalf("failed to create .ssh dir: %v", err)
	}

	// Write both keys
	if err := os.WriteFile(filepath.Join(sshDir, "id_ed25519"), []byte("ed25519-key"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(sshDir, "id_rsa"), []byte("rsa-key"), 0600); err != nil {
		t.Fatal(err)
	}

	orig := os.Getenv("HOME")
	os.Setenv("HOME", tmpDir)
	defer os.Setenv("HOME", orig)

	data, err := loadSSHKey()
	if err != nil {
		t.Fatalf("loadSSHKey() error: %v", err)
	}
	// Should prefer ed25519 (it's first in the list)
	if string(data) != "ed25519-key" {
		t.Errorf("loadSSHKey() = %q, want ed25519 key", string(data))
	}
}
