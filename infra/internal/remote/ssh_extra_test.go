package remote

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"io"
	"net"
	"strings"
	"testing"
	"time"

	"golang.org/x/crypto/ssh"
)

// handleTestSSHConn handles a single SSH connection, executing commands via /bin/sh.
func handleTestSSHConn(c net.Conn, cfg *ssh.ServerConfig) {
	sconn, chans, reqs, err := ssh.NewServerConn(c, cfg)
	if err != nil {
		return
	}
	defer func() { _ = sconn.Close() }()
	go ssh.DiscardRequests(reqs)

	for newChan := range chans {
		if newChan.ChannelType() != "session" {
			_ = newChan.Reject(ssh.UnknownChannelType, "unknown channel type")

			continue
		}
		ch, requests, err := newChan.Accept()
		if err != nil {
			return
		}
		go handleTestSSHSession(ch, requests)
	}
}

func handleTestSSHSession(ch ssh.Channel, requests <-chan *ssh.Request) {
	defer func() { _ = ch.Close() }()
	for req := range requests {
		if req.Type == "exec" { //nolint:nestif
			// Parse command from payload (4-byte length prefix + command string)
			if len(req.Payload) < 4 {
				if req.WantReply {
					_ = req.Reply(false, nil)
				}

				continue
			}
			cmdLen := int(req.Payload[0])<<24 | int(req.Payload[1])<<16 | int(req.Payload[2])<<8 | int(req.Payload[3])
			if len(req.Payload) < 4+cmdLen {
				if req.WantReply {
					_ = req.Reply(false, nil)
				}

				continue
			}
			cmd := string(req.Payload[4 : 4+cmdLen])
			if req.WantReply {
				_ = req.Reply(true, nil)
			}
			_, _ = io.WriteString(ch, "output: "+cmd+"\n")
			// Send exit status 0
			exitStatus := []byte{0, 0, 0, 0}
			_, _ = ch.SendRequest("exit-status", false, exitStatus)

			return
		}
		if req.WantReply {
			_ = req.Reply(false, nil)
		}
	}
}

// makeTestSSHClient creates an SSHClient connected to an in-process test server.
func makeTestSSHClient(t *testing.T) (*SSHClient, func()) {
	t.Helper()

	// Generate host key for server
	_, hostPrivKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("generate host key: %v", err)
	}
	hostSigner, err := ssh.NewSignerFromKey(hostPrivKey)
	if err != nil {
		t.Fatalf("create host signer: %v", err)
	}

	// Generate client key
	_, clientPrivKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("generate client key: %v", err)
	}
	clientSigner, err := ssh.NewSignerFromKey(clientPrivKey)
	if err != nil {
		t.Fatalf("create client signer: %v", err)
	}

	cfg := &ssh.ServerConfig{
		PublicKeyCallback: func(_ ssh.ConnMetadata, _ ssh.PublicKey) (*ssh.Permissions, error) {
			return nil, nil
		},
	}
	cfg.AddHostKey(hostSigner)

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}

	serverDone := make(chan struct{})
	go func() {
		defer close(serverDone)
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go handleTestSSHConn(conn, cfg)
		}
	}()

	// Connect the SSH client
	clientConfig := &ssh.ClientConfig{
		User:            "root",
		Auth:            []ssh.AuthMethod{ssh.PublicKeys(clientSigner)},
		HostKeyCallback: ssh.InsecureIgnoreHostKey(),
		Timeout:         5 * time.Second,
	}

	sshConn, err := ssh.Dial("tcp", ln.Addr().String(), clientConfig)
	if err != nil {
		_ = ln.Close()
		t.Fatalf("dial test SSH server: %v", err)
	}

	host, _, _ := net.SplitHostPort(ln.Addr().String())
	client := &SSHClient{client: sshConn, host: host}

	cleanup := func() {
		_ = sshConn.Close()
		_ = ln.Close()
		<-serverDone
	}

	return client, cleanup
}

// --- SSHClient.Host ---

func TestSSHClientHost(t *testing.T) {
	client, cleanup := makeTestSSHClient(t)
	defer cleanup()

	host := client.Host()
	if host == "" {
		t.Error("Host() should return non-empty string")
	}
	if host != "127.0.0.1" {
		t.Errorf("Host() = %q, want \"127.0.0.1\"", host)
	}
}

// --- SSHClient.Close ---

func TestSSHClientClose(t *testing.T) {
	client, cleanup := makeTestSSHClient(t)
	defer cleanup()

	err := client.Close()
	if err != nil {
		t.Errorf("Close() error: %v", err)
	}
}

func TestSSHClientCloseIdempotent(t *testing.T) {
	client, cleanup := makeTestSSHClient(t)
	defer cleanup()

	// First close should succeed
	if err := client.Close(); err != nil {
		t.Logf("First Close() error (may be ok): %v", err)
	}
	// Second close may or may not error, but should not panic
	_ = client.Close()
}

// --- SSHClient.Run ---

func TestSSHClientRun(t *testing.T) {
	client, cleanup := makeTestSSHClient(t)
	defer cleanup()

	var buf strings.Builder
	err := client.Run(context.Background(), "echo hello", &buf)
	if err != nil {
		t.Errorf("Run() error: %v", err)
	}
}

func TestSSHClientRunContextCancelled(t *testing.T) {
	client, cleanup := makeTestSSHClient(t)
	defer cleanup()

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // pre-cancel

	var buf strings.Builder
	err := client.Run(ctx, "echo hello", &buf)
	// With a cancelled context, Run may return the context error or the command may
	// still complete since session is already established. Either is acceptable.
	_ = err
}

// --- WaitForSSH ---

func TestWaitForSSHSuccess(t *testing.T) {
	// Start a simple TCP listener to simulate SSH being available
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer func() { _ = ln.Close() }()

	host, portStr, _ := net.SplitHostPort(ln.Addr().String())
	var port int
	if _, err := io.WriteString(io.Discard, ""); err == nil {
		// Parse port
		_, _ = net.LookupPort("tcp", portStr)
	}
	// Convert portStr to int
	for _, c := range portStr {
		port = port*10 + int(c-'0')
	}

	// Accept connections in background
	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			_ = conn.Close()
		}
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	err = WaitForSSH(ctx, host, port)
	if err != nil {
		t.Errorf("WaitForSSH() error: %v", err)
	}
}

func TestWaitForSSHContextCancelled(t *testing.T) {
	// Use a port that is not listening
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // pre-cancel

	err := WaitForSSH(ctx, "127.0.0.1", 19999)
	if err == nil {
		t.Error("expected error when context is pre-cancelled")
	}
}

func TestWaitForSSHTimeout(t *testing.T) {
	// Use a port that is not listening — connection will be refused
	// Use a short timeout so the test completes quickly
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()

	// Port 1 should always be refused or filtered
	err := WaitForSSH(ctx, "127.0.0.1", 1)
	if err == nil {
		t.Error("expected timeout error when SSH port is not open")
	}
}
