package deploy

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/binary"
	"encoding/json"
	"encoding/pem"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	gossh "golang.org/x/crypto/ssh"

	"github.com/ThatCatDev/tanrenai/infra/internal/config"
	"github.com/ThatCatDev/tanrenai/infra/internal/network"
	"github.com/ThatCatDev/tanrenai/infra/internal/vastai"
)

// failingSSHServer is an in-process SSH server that returns exit code 1 for all commands.
type failingSSHServer struct {
	ln   net.Listener
	done chan struct{}
}

func startFailingSSHServer(t *testing.T) (*failingSSHServer, string, int) {
	t.Helper()

	_, hostPrivKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("generate host key: %v", err)
	}
	hostSigner, err := gossh.NewSignerFromKey(hostPrivKey)
	if err != nil {
		t.Fatalf("create host signer: %v", err)
	}

	cfg := &gossh.ServerConfig{
		PublicKeyCallback: func(_ gossh.ConnMetadata, _ gossh.PublicKey) (*gossh.Permissions, error) {
			return nil, nil
		},
	}
	cfg.AddHostKey(hostSigner)

	// Generate a client key pair and write it to a temp dir so that
	// remote.Connect() → loadSSHKey() finds a valid key on CI runners.
	_, clientPrivKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("generate client key: %v", err)
	}
	pemBlock, err := gossh.MarshalPrivateKey(clientPrivKey, "")
	if err != nil {
		t.Fatalf("marshal client private key: %v", err)
	}
	pemBytes := pem.EncodeToMemory(pemBlock)

	tmpDir := t.TempDir()
	sshDir := filepath.Join(tmpDir, ".ssh")
	if err := os.MkdirAll(sshDir, 0o700); err != nil {
		t.Fatalf("mkdir .ssh: %v", err)
	}
	keyPath := filepath.Join(sshDir, "id_ed25519")
	if err := os.WriteFile(keyPath, pemBytes, 0o600); err != nil {
		t.Fatalf("write client key: %v", err)
	}
	t.Setenv("HOME", tmpDir)

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}

	s := &failingSSHServer{
		ln:   ln,
		done: make(chan struct{}),
	}

	go func() {
		defer close(s.done)
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go s.handleConn(conn, cfg)
		}
	}()

	host := "127.0.0.1"
	port := ln.Addr().(*net.TCPAddr).Port

	t.Cleanup(func() {
		_ = ln.Close()
		<-s.done
	})

	return s, host, port
}

func (s *failingSSHServer) handleConn(c net.Conn, cfg *gossh.ServerConfig) {
	sconn, chans, reqs, err := gossh.NewServerConn(c, cfg)
	if err != nil {
		return
	}
	defer func() { _ = sconn.Close() }()
	go gossh.DiscardRequests(reqs)

	for newChan := range chans {
		if newChan.ChannelType() != "session" {
			_ = newChan.Reject(gossh.UnknownChannelType, "unknown channel type")

			continue
		}
		ch, requests, err := newChan.Accept()
		if err != nil {
			return
		}
		go s.handleFailingSession(ch, requests)
	}
}

func (s *failingSSHServer) handleFailingSession(ch gossh.Channel, requests <-chan *gossh.Request) {
	defer func() { _ = ch.Close() }()
	for req := range requests {
		if req.Type == "exec" {
			if req.WantReply {
				_ = req.Reply(true, nil)
			}
			_, _ = io.WriteString(ch, "command failed: permission denied\n")
			// Send exit status 1 (failure)
			exitStatus := make([]byte, 4)
			binary.BigEndian.PutUint32(exitStatus, 1)
			_, _ = ch.SendRequest("exit-status", false, exitStatus)

			return
		}
		if req.WantReply {
			_ = req.Reply(false, nil)
		}
	}
}

// TestDeployRunStageFails tests that Run returns an error when a setup stage fails (non-verbose mode).
func TestDeployRunStageFails(t *testing.T) {
	_, sshHost, sshPort := startFailingSSHServer(t)

	inst := vastai.Instance{
		ID:        570,
		Status:    "running",
		GPUName:   "RTX 4090",
		CostPerHr: 0.5,
		SSHHost:   sshHost,
		SSHPort:   sshPort,
	}

	vastaiSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"instances": []vastai.Instance{inst},
		})
	}))
	defer vastaiSrv.Close()

	client := vastai.NewClient("test-key")
	overrideVastaiClientTransport(client, vastaiSrv)

	cfg := config.Config{
		VastaiAPIKey:   "test-key",
		VastaiInstance: "570",
		Network:        "none",
		GPUPort:        11435,
	}

	var buf bytes.Buffer
	d := New(client, network.NewNoneProvider(), cfg, &buf, false) // verbose=false

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	_, err := d.Run(ctx)
	if err == nil {
		t.Error("expected error when setup stage fails")
	}
	if !strings.Contains(err.Error(), "setup stage") {
		t.Errorf("error should mention setup stage, got: %v", err)
	}
}

// TestDeployRunStageFailsVerbose tests the verbose mode failure path (lines 136-149 in deployer.go).
func TestDeployRunStageFailsVerbose(t *testing.T) {
	_, sshHost, sshPort := startFailingSSHServer(t)

	inst := vastai.Instance{
		ID:        571,
		Status:    "running",
		GPUName:   "A100",
		CostPerHr: 1.0,
		SSHHost:   sshHost,
		SSHPort:   sshPort,
	}

	vastaiSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"instances": []vastai.Instance{inst},
		})
	}))
	defer vastaiSrv.Close()

	client := vastai.NewClient("test-key")
	overrideVastaiClientTransport(client, vastaiSrv)

	cfg := config.Config{
		VastaiAPIKey:   "test-key",
		VastaiInstance: "571",
		Network:        "none",
		GPUPort:        11435,
	}

	var buf bytes.Buffer
	d := New(client, network.NewNoneProvider(), cfg, &buf, true) // verbose=true

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	_, err := d.Run(ctx)
	if err == nil {
		t.Error("expected error when setup stage fails in verbose mode")
	}
	if !strings.Contains(err.Error(), "setup stage") {
		t.Errorf("error should mention setup stage, got: %v", err)
	}
}
