package cmd

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"github.com/ThatCatDev/tanrenai/shared/apiclient"
	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
)

// sessionMode picks whether `tanrenai run` uses an embedded local GPU or the
// hosted platform. The decision tree is enforced in resolveSessionMode.
type sessionMode int

const (
	sessionModeLocal sessionMode = iota
	sessionModeRemote
)

// resolveSessionMode applies the approved policy:
//
//  1. --local flag always forces local (escape hatch, even if logged in).
//  2. Otherwise, if credentials are present, use the remote platform.
//  3. Otherwise, fall back to local with a one-line hint.
func resolveSessionMode(p runParams, log *startupLog) sessionMode {
	if p.local {
		return sessionModeLocal
	}
	if authToken != "" {
		return sessionModeRemote
	}
	log.Info("Not logged in — using local GPU. Run `tanrenai login` to use the hosted service.")
	return sessionModeLocal
}

// isModelURI reports whether the user supplied a pullable URI/URL rather
// than a bare model name. hf://, https://, and http:// are all supported
// by the platform's /api/pull proxy.
func isModelURI(s string) bool {
	return strings.HasPrefix(s, "hf://") ||
		strings.HasPrefix(s, "https://") ||
		strings.HasPrefix(s, "http://")
}

// pullModelForRemote streams a download through the platform's /api/pull
// endpoint (which in turn triggers EnsureRunning on the platform). Returns
// the bare model name the GPU will resolve on a subsequent /api/load call.
func pullModelForRemote(ctx context.Context, client *apiclient.Client, uri string, log *startupLog) (string, error) {
	log.Info("Pulling model from " + uri + "...")

	ch, err := client.PullModel(ctx, uri)
	if err != nil {
		return "", err
	}

	var lastPath string
	var lastPct int
	for ev := range ch {
		if ev.Err != nil {
			return "", ev.Err
		}
		switch ev.Event.Status {
		case "resolving":
			if ev.Event.TotalFiles > 1 {
				log.Info(fmt.Sprintf("Downloading %d files...", ev.Event.TotalFiles))
			}
		case "downloading":
			// Log every ~10% to avoid spamming the TUI.
			if ev.Event.Percent >= lastPct+10 {
				lastPct = ev.Event.Percent
				log.Info(fmt.Sprintf("Downloading... %d%%", ev.Event.Percent))
			}
		case "downloaded":
			if ev.Event.Path != "" {
				lastPath = ev.Event.Path
			}
		case "error":
			msg := ev.Event.Error
			if msg == "" {
				msg = "pull failed"
			}
			return "", fmt.Errorf("%s", msg)
		}
	}

	if lastPath == "" {
		return "", fmt.Errorf("pull completed without returning a file path")
	}
	return modelNameFromDiskPath(lastPath), nil
}

// modelNameFromDiskPath strips the directory and extension to get the model
// identifier the GPU's /api/load expects (e.g.
// /data/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf -> Qwen2.5-7B-Instruct-Q4_K_M).
func modelNameFromDiskPath(p string) string {
	base := filepath.Base(p)
	return strings.TrimSuffix(base, filepath.Ext(base))
}

// loadModelWithProgress performs `client.LoadModel(ctx, model)` and, when in
// remote mode, concurrently polls `/api/instance/status` every 5s to surface
// provisioning progress to the TUI. It retries on the platform's transient
// "503 gpu_unavailable / still provisioning" responses, which happen while
// the Vast.ai instance is booting or the GPU server is coming up.
func loadModelWithProgress(ctx context.Context, client *apiclient.Client, mode sessionMode, model string, log *startupLog) (*api.LoadResponse, error) {
	if mode != sessionModeRemote {
		return client.LoadModel(ctx, model)
	}

	done := make(chan struct{})
	go pollProvisionStatus(ctx, client, log, done)
	defer close(done)

	// Cold Vast.ai provision takes 5-15 min. The platform returns 503 with
	// "still provisioning" during the window when EnsureRunning has kicked
	// off work but the instance isn't ready yet. Retry with backoff.
	const (
		maxAttempts = 120 // 120 * 10s = 20 min ceiling
		retryDelay  = 10 * time.Second
	)
	for attempt := 1; ; attempt++ {
		resp, err := client.LoadModel(ctx, model)
		if err == nil {
			return resp, nil
		}
		if !isProvisioningInProgress(err) || attempt >= maxAttempts {
			return nil, err
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(retryDelay):
		}
	}
}

// isProvisioningInProgress reports whether the error indicates the platform
// has started provisioning but the GPU isn't ready yet. It's transient — the
// caller should wait and retry rather than abort.
func isProvisioningInProgress(err error) bool {
	var se *apiclient.StatusError
	if !errors.As(err, &se) {
		return false
	}
	if se.Code != 503 {
		return false
	}
	body := strings.ToLower(se.Body)
	return strings.Contains(body, "provisioning") ||
		strings.Contains(body, "booting") ||
		strings.Contains(body, "gpu_unavailable")
}

func pollProvisionStatus(ctx context.Context, client *apiclient.Client, log *startupLog, done <-chan struct{}) {
	t := time.NewTicker(5 * time.Second)
	defer t.Stop()

	var lastMsg string
	for {
		select {
		case <-done:
			return
		case <-ctx.Done():
			return
		case <-t.C:
			st, err := client.InstanceStatus(ctx)
			if err != nil || st == nil {
				continue
			}
			msg := humanizeStatus(st.Status)
			if msg != "" && msg != lastMsg {
				log.Info(msg)
				lastMsg = msg
			}
		}
	}
}

// humanizeStatus turns the coarse instance states the platform reports
// into user-friendly TUI lines. The platform's `/api/instance/status`
// also emits a `provision_state` field; once the shared api.InstanceStatus
// struct grows that field we can show finer-grained progress.
func humanizeStatus(status string) string {
	switch status {
	case "none":
		return "GPU: provisioning (searching for offer)..."
	case "pending", "provisioning":
		return "GPU: provisioning..."
	case "running":
		return "" // don't spam once it's up
	case "destroying":
		return "GPU: shutting down..."
	}
	return ""
}
