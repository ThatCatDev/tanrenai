package deploy

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"encoding/pem"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	gossh "golang.org/x/crypto/ssh"

	"github.com/ThatCatDev/tanrenai/infra/internal/config"
	"github.com/ThatCatDev/tanrenai/infra/internal/network"
	"github.com/ThatCatDev/tanrenai/infra/internal/vastai"
)

// deploySSHServer is a minimal in-process SSH server for deploy integration tests.
type deploySSHServer struct {
	ln   net.Listener
	done chan struct{}
}

func startDeploySSHServer(t *testing.T) (*deploySSHServer, string, int) {
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

	s := &deploySSHServer{
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
	_, portStr, _ := net.SplitHostPort(ln.Addr().String())
	port := 0
	for _, c := range portStr {
		port = port*10 + int(c-'0')
	}

	t.Cleanup(func() {
		_ = ln.Close()
		<-s.done
	})

	return s, host, port
}

func (s *deploySSHServer) handleConn(c net.Conn, cfg *gossh.ServerConfig) {
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
		go s.handleSession(ch, requests)
	}
}

func (s *deploySSHServer) handleSession(ch gossh.Channel, requests <-chan *gossh.Request) {
	defer func() { _ = ch.Close() }()
	for req := range requests {
		if req.Type == "exec" {
			if req.WantReply {
				_ = req.Reply(true, nil)
			}
			_, _ = io.WriteString(ch, "ok\n")
			exitStatus := []byte{0, 0, 0, 0}
			_, _ = ch.SendRequest("exit-status", false, exitStatus)

			return
		}
		if req.WantReply {
			_ = req.Reply(false, nil)
		}
	}
}

// TestDeployRunWithNoneNetwork tests the full Run pipeline using none network
// and an in-process SSH server, stubbing out vast.ai and health check.
func TestDeployRunWithNoneNetwork(t *testing.T) {
	// 1. Start in-process SSH server
	_, sshHost, sshPort := startDeploySSHServer(t)

	// 2. Start health check HTTP server
	healthSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
		}
	}))
	defer healthSrv.Close()

	// Parse the health server port for the GPU URL
	healthPort := 0
	_, portStr, _ := net.SplitHostPort(healthSrv.Listener.Addr().String())
	for _, c := range portStr {
		healthPort = healthPort*10 + int(c-'0')
	}

	// 3. Set up mock vast.ai server
	// - GetInstance returns a running instance pointing at our SSH server
	// - When deployer polls for the instance to have SSHHost/SSHPort set, it succeeds immediately
	inst := vastai.Instance{
		ID:        555,
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
		VastaiInstance: "555",
		Network:        "none",
		GPUPort:        healthPort, // health check hits our mock server
		Model:          "",         // no model pull
		MinGPURAM:      24,
		MaxCostPerHr:   1.0,
		DiskGB:         50,
	}

	var buf bytes.Buffer
	d := New(client, network.NewNoneProvider(), cfg, &buf, false)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	result, err := d.Run(ctx)
	if err != nil {
		t.Fatalf("Run() error: %v\nOutput:\n%s", err, buf.String())
	}
	if result == nil {
		t.Fatal("Run() returned nil result")
	}
	if result.InstanceID != 555 {
		t.Errorf("result.InstanceID = %d, want 555", result.InstanceID)
	}
	if result.GPUName != "RTX 4090" {
		t.Errorf("result.GPUName = %q, want \"RTX 4090\"", result.GPUName)
	}
}
