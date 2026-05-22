package cmd

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ThatCatDev/tanrenai/shared/apiclient"
	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
)

func TestResolvePullArgs(t *testing.T) {
	tests := []struct {
		name        string
		arg         string
		saveAs      string
		wantURL     string
		wantSaveAs  string
		wantErrFrag string
	}{
		// Bare names get expanded to canonical hf:// URIs and the on-disk
		// basename is pinned to the user-typed identifier so a subsequent
		// /api/load by that same name finds the file.
		{
			name:       "bare name resolves and pins saveAs",
			arg:        "Qwen3.6-35B-A3B-MTP-Q8_0",
			saveAs:     "",
			wantURL:    "hf://unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Q8_0",
			wantSaveAs: "Qwen3.6-35B-A3B-MTP-Q8_0",
		},
		{
			name:       "bare name with UD dynamic quant",
			arg:        "Qwen3.5-122B-A10B-UD-Q4_K_XL",
			saveAs:     "",
			wantURL:    "hf://unsloth/Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL",
			wantSaveAs: "Qwen3.5-122B-A10B-UD-Q4_K_XL",
		},
		{
			name:       "explicit --name wins over derived bare name",
			arg:        "Qwen3.6-35B-A3B-Q4_K_M",
			saveAs:     "myalias",
			wantURL:    "hf://unsloth/Qwen3.6-35B-A3B-GGUF/Q4_K_M",
			wantSaveAs: "myalias",
		},
		// URIs flow through unchanged. The GPU server (tanrenai-gpu ≥
		// v1.4.0) derives the on-disk basename from canonical hf:// URIs
		// in its PullHandler, so the CLI doesn't need to.
		{
			name:       "hf URI flows through with empty saveAs",
			arg:        "hf://unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Q8_0",
			saveAs:     "",
			wantURL:    "hf://unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Q8_0",
			wantSaveAs: "",
		},
		{
			name:       "explicit --name with hf URI overrides server derivation",
			arg:        "hf://unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Q8_0",
			saveAs:     "myalias",
			wantURL:    "hf://unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Q8_0",
			wantSaveAs: "myalias",
		},
		{
			name:       "https URL flows through unchanged",
			arg:        "https://huggingface.co/x/resolve/main/y.gguf",
			saveAs:     "",
			wantURL:    "https://huggingface.co/x/resolve/main/y.gguf",
			wantSaveAs: "",
		},
		{
			name:        "bare name without quant suffix errors",
			arg:         "Qwen2.5-7B-Instruct",
			saveAs:      "",
			wantErrFrag: "could not resolve",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			url, saveAs, err := resolvePullArgs(tc.arg, tc.saveAs)
			if tc.wantErrFrag != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tc.wantErrFrag)
				}
				if !strings.Contains(err.Error(), tc.wantErrFrag) {
					t.Errorf("error = %q, want fragment %q", err.Error(), tc.wantErrFrag)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if url != tc.wantURL {
				t.Errorf("url = %q, want %q", url, tc.wantURL)
			}
			if saveAs != tc.wantSaveAs {
				t.Errorf("saveAs = %q, want %q", saveAs, tc.wantSaveAs)
			}
		})
	}
}

// TestPullModel_BareNamePinsBackendName pins the wire-level invariant for
// the bare-name UX path: pulling `Qwen3.6-35B-A3B-MTP-Q8_0` sends the
// expanded hf:// URI plus a Name field matching the user-typed identifier,
// so /api/load by that same name later finds the freshly-pulled file.
//
// The URI path is no longer tested here because tanrenai-gpu ≥ v1.4.0
// derives the basename server-side via naming.DeriveBareNameFromURI — the
// invariant lives in that repo's pull handler tests.
func TestPullModel_BareNamePinsBackendName(t *testing.T) {
	var gotReq api.PullRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		_, _ = w.Write([]byte(`data: {"status":"downloaded","path":"/m.gguf"}` + "\n\n"))
		flusher.Flush()
	}))
	defer srv.Close()

	url, saveAs, err := resolvePullArgs("Qwen3.6-35B-A3B-MTP-Q8_0", "")
	if err != nil {
		t.Fatalf("resolvePullArgs: %v", err)
	}

	c := apiclient.New(srv.URL)
	ch, err := c.PullModel(t.Context(), url, saveAs)
	if err != nil {
		t.Fatalf("PullModel: %v", err)
	}
	for range ch {
	}

	if gotReq.URL != "hf://unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Q8_0" {
		t.Errorf("backend received URL %q, want the expanded hf:// URI", gotReq.URL)
	}
	if gotReq.Name != "Qwen3.6-35B-A3B-MTP-Q8_0" {
		t.Errorf("backend received Name %q, want %q (bare name pinned for /api/load round-trip)", gotReq.Name, "Qwen3.6-35B-A3B-MTP-Q8_0")
	}
}
