package remote

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"net"
	"testing"
	"time"

	"golang.org/x/crypto/ssh"
)

// TestConnectToTestServer verifies the exported Connect() function works with our test SSH server.
// This requires the user to have an SSH key in ~/.ssh/ (id_ed25519 or id_rsa), since Connect()
// calls loadSSHKey() internally. The test SSH server accepts any public key.
func TestConnectToTestServer(t *testing.T) {
	// First check if any SSH key exists — if not, skip this test
	key, err := loadSSHKey()
	if err != nil {
		t.Skip("no SSH key found, skipping Connect test:", err)
	}
	if len(key) == 0 {
		t.Skip("empty SSH key, skipping Connect test")
	}

	// Start a test SSH server that accepts any public key
	_, hostPrivKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("generate host key: %v", err)
	}
	hostSigner, err := ssh.NewSignerFromKey(hostPrivKey)
	if err != nil {
		t.Fatalf("create host signer: %v", err)
	}

	srvCfg := &ssh.ServerConfig{
		PublicKeyCallback: func(_ ssh.ConnMetadata, _ ssh.PublicKey) (*ssh.Permissions, error) {
			return nil, nil // accept any key
		},
	}
	srvCfg.AddHostKey(hostSigner)

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer func() { _ = ln.Close() }()

	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go func(c net.Conn) {
				sconn, chans, reqs, err := ssh.NewServerConn(c, srvCfg)
				if err != nil {
					return
				}
				defer func() { _ = sconn.Close() }()
				go ssh.DiscardRequests(reqs)
				for newChan := range chans {
					_ = newChan.Reject(ssh.UnknownChannelType, "no channels")
				}
			}(conn)
		}
	}()

	host := "127.0.0.1"
	port := ln.Addr().(*net.TCPAddr).Port

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	sshClient, err := Connect(ctx, host, port, "root")
	if err != nil {
		t.Fatalf("Connect() error: %v", err)
	}
	defer func() { _ = sshClient.Close() }()

	if sshClient.Host() != host {
		t.Errorf("Host() = %q, want %q", sshClient.Host(), host)
	}
}

// TestConnectContextCancelledDuringRetry verifies Connect's retry loop exits when context is done.
// We start a TCP listener that accepts connections but doesn't speak SSH (causing auth failure retry loop),
// then cancel the context. This exercises the ctx.Done() branch in the retry loop.
func TestConnectContextCancelledDuringRetry(t *testing.T) {
	// Open a TCP listener that accepts but immediately closes (simulates SSH not ready)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer func() { _ = ln.Close() }()

	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			// Write garbage to cause SSH connection to fail
			_, _ = conn.Write([]byte("not-ssh\r\n"))
			_ = conn.Close()
		}
	}()

	port := ln.Addr().(*net.TCPAddr).Port

	// Use a very short deadline so the loop exits quickly
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	_, err = Connect(ctx, "127.0.0.1", port, "root")
	if err == nil {
		t.Error("expected error when connection can't be established")
	}
}

// TestConnectAuthFailure verifies Connect returns an auth error when the server rejects the key.
func TestConnectAuthFailure(t *testing.T) {
	// Check we have an SSH key to use
	_, err := loadSSHKey()
	if err != nil {
		t.Skip("no SSH key found, skipping Connect auth failure test")
	}

	// Start a server that always rejects auth
	_, hostPrivKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("generate host key: %v", err)
	}
	hostSigner, err := ssh.NewSignerFromKey(hostPrivKey)
	if err != nil {
		t.Fatalf("create host signer: %v", err)
	}

	srvCfg := &ssh.ServerConfig{
		PublicKeyCallback: func(_ ssh.ConnMetadata, _ ssh.PublicKey) (*ssh.Permissions, error) {
			return nil, ssh.ErrNoAuth // reject all keys
		},
	}
	srvCfg.AddHostKey(hostSigner)

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer func() { _ = ln.Close() }()

	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go func(c net.Conn) {
				sconn, _, _, err := ssh.NewServerConn(c, srvCfg)
				if err == nil && sconn != nil {
					_ = sconn.Close()
				}
			}(conn)
		}
	}()

	port := ln.Addr().(*net.TCPAddr).Port
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, err = Connect(ctx, "127.0.0.1", port, "root")
	if err == nil {
		t.Error("expected auth error when server rejects key")
	}
}
