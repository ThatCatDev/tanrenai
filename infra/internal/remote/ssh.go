package remote

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net"
	"os"
	"strings"
	"time"

	"golang.org/x/crypto/ssh"
)

// SSHClient wraps an SSH connection to a remote host.
type SSHClient struct {
	client *ssh.Client
	host   string
}

// Connect establishes an SSH connection. It tries the user's default SSH key.
func Connect(ctx context.Context, host string, port int, user string) (*SSHClient, error) {
	key, err := loadSSHKey()
	if err != nil {
		return nil, fmt.Errorf("load SSH key: %w", err)
	}

	signer, err := ssh.ParsePrivateKey(key)
	if err != nil {
		return nil, fmt.Errorf("parse SSH key: %w", err)
	}

	config := &ssh.ClientConfig{
		User: user,
		Auth: []ssh.AuthMethod{
			ssh.PublicKeys(signer),
		},
		HostKeyCallback: ssh.InsecureIgnoreHostKey(),
		Timeout:         30 * time.Second,
	}

	addr := fmt.Sprintf("%s:%d", host, port)
	slog.Info("connecting via SSH", "addr", addr, "user", user)

	var client *ssh.Client
	deadline, ok := ctx.Deadline()
	if !ok {
		deadline = time.Now().Add(5 * time.Minute)
	}

	for time.Now().Before(deadline) {
		client, err = ssh.Dial("tcp", addr, config)
		if err == nil {
			break
		}

		// Don't retry auth failures — key is wrong, retrying won't help
		if isAuthError(err) {
			return nil, fmt.Errorf("SSH auth failed (check your key is added to vast.ai): %w", err)
		}

		slog.Debug("SSH not ready, retrying", "err", err)
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(5 * time.Second):
		}
	}
	if client == nil {
		return nil, fmt.Errorf("SSH connection failed after retries: %w", err)
	}

	return &SSHClient{client: client, host: host}, nil
}

// Run executes a command over SSH and streams output to the given writer.
func (s *SSHClient) Run(ctx context.Context, cmd string, output io.Writer) error {
	session, err := s.client.NewSession()
	if err != nil {
		return fmt.Errorf("create SSH session: %w", err)
	}
	defer session.Close()

	session.Stdout = output
	session.Stderr = output

	done := make(chan error, 1)
	go func() {
		done <- session.Run(cmd)
	}()

	select {
	case <-ctx.Done():
		_ = session.Signal(ssh.SIGTERM)
		return ctx.Err()
	case err := <-done:
		return err
	}
}

// Close closes the SSH connection.
func (s *SSHClient) Close() error {
	return s.client.Close()
}

// Host returns the remote host address.
func (s *SSHClient) Host() string {
	return s.host
}

// WaitForSSH polls until SSH is reachable on the given host:port.
func WaitForSSH(ctx context.Context, host string, port int) error {
	addr := fmt.Sprintf("%s:%d", host, port)
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			conn, err := net.DialTimeout("tcp", addr, 5*time.Second)
			if err == nil {
				conn.Close()
				return nil
			}
			slog.Debug("waiting for SSH", "addr", addr, "err", err)
		}
	}
}

func loadSSHKey() ([]byte, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}

	// Try common key files in order
	for _, name := range []string{"id_ed25519", "id_rsa"} {
		path := home + "/.ssh/" + name
		data, err := os.ReadFile(path)
		if err == nil {
			return data, nil
		}
	}

	return nil, fmt.Errorf("no SSH key found in ~/.ssh/ (tried id_ed25519, id_rsa)")
}

func isAuthError(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	return strings.Contains(msg, "unable to authenticate") ||
		strings.Contains(msg, "no supported methods remain") ||
		strings.Contains(msg, "handshake failed")
}
