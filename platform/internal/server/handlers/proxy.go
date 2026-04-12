package handlers

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"strings"

	"github.com/ThatCatDev/tanrenai/platform/internal/auth"
	"github.com/ThatCatDev/tanrenai/platform/internal/instance"
)

// ProxyHandler proxies GPU requests to the user's active instance.
type ProxyHandler struct {
	Manager *instance.Manager
}

// ChatCompletions proxies POST /v1/chat/completions to the user's GPU instance.
func (h *ProxyHandler) ChatCompletions(w http.ResponseWriter, r *http.Request) {
	user := auth.UserFromContext(r.Context())
	if user == nil {
		writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "not authenticated"})
		return
	}

	// Extract model from request body to determine instance requirements
	modelSize := extractModelSize(r)

	gpuURL, err := h.Manager.EnsureRunning(r.Context(), user, modelSize)
	if err != nil {
		slog.Warn("GPU not available", "error", err, "user", user.Email)
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error":   "gpu_unavailable",
			"message": err.Error(),
		})
		return
	}

	h.Manager.RecordActivity(r.Context(), user.ID)
	h.proxyRequest(w, r, gpuURL+"/v1/chat/completions")
}

// Tokenize proxies POST /tokenize.
func (h *ProxyHandler) Tokenize(w http.ResponseWriter, r *http.Request) {
	h.proxyToGPU(w, r, "/tokenize")
}

// ListModels proxies GET /v1/models.
func (h *ProxyHandler) ListModels(w http.ResponseWriter, r *http.Request) {
	h.proxyToGPU(w, r, "/v1/models")
}

// LoadModel proxies POST /api/load.
func (h *ProxyHandler) LoadModel(w http.ResponseWriter, r *http.Request) {
	h.proxyToGPU(w, r, "/api/load")
}

// PullModel proxies POST /api/pull.
func (h *ProxyHandler) PullModel(w http.ResponseWriter, r *http.Request) {
	h.proxyToGPU(w, r, "/api/pull")
}

func (h *ProxyHandler) proxyToGPU(w http.ResponseWriter, r *http.Request, path string) {
	user := auth.UserFromContext(r.Context())
	if user == nil {
		writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "not authenticated"})
		return
	}

	gpuURL, err := h.Manager.EnsureRunning(r.Context(), user, "")
	if err != nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error":   "gpu_unavailable",
			"message": err.Error(),
		})
		return
	}

	h.Manager.RecordActivity(r.Context(), user.ID)
	h.proxyRequest(w, r, gpuURL+path)
}

func (h *ProxyHandler) proxyRequest(w http.ResponseWriter, r *http.Request, targetURL string) {
	proxyReq, err := http.NewRequestWithContext(r.Context(), r.Method, targetURL, r.Body)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to create proxy request"})
		return
	}

	proxyReq.Header.Set("Content-Type", r.Header.Get("Content-Type"))

	resp, err := http.DefaultClient.Do(proxyReq)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": "GPU server unreachable"})
		return
	}
	defer func() { _ = resp.Body.Close() }()

	// Copy response headers
	for k, v := range resp.Header {
		for _, vv := range v {
			w.Header().Add(k, vv)
		}
	}
	w.WriteHeader(resp.StatusCode)

	// Stream the response (important for SSE chat completions)
	if f, ok := w.(http.Flusher); ok {
		buf := make([]byte, 4096)
		for {
			n, err := resp.Body.Read(buf)
			if n > 0 {
				_, _ = w.Write(buf[:n])
				f.Flush()
			}
			if err != nil {
				break
			}
		}
	} else {
		_, _ = io.Copy(w, resp.Body)
	}
}

func extractModelSize(r *http.Request) string {
	// Try to peek at the model field without consuming the body
	// This is a best-effort extraction for auto-provisioning hints
	if r.Body == nil {
		return ""
	}

	// Read body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return ""
	}
	// Restore body for proxy
	r.Body = io.NopCloser(strings.NewReader(string(body)))

	var req struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return ""
	}

	// Extract size hint from model name (e.g. "qwen3:72b" → "72b")
	model := req.Model
	if idx := strings.LastIndex(model, ":"); idx != -1 {
		return model[idx+1:]
	}
	return ""
}
