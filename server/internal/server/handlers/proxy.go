package handlers

import (
	"encoding/json"
	"io"
	"net/http"

	"github.com/ThatCatDev/tanrenai/server/internal/gpuclient"
	"github.com/ThatCatDev/tanrenai/server/internal/gpuprovider"
	"github.com/ThatCatDev/tanrenai/server/pkg/api"
)

// ProxyHandler transparently proxies requests to the GPU server.
type ProxyHandler struct {
	GPUClient *gpuclient.Client
	Provider  gpuprovider.Provider
}

// ensureGPU ensures the GPU is running and records activity.
func (h *ProxyHandler) ensureGPU(w http.ResponseWriter, r *http.Request) bool {
	h.Provider.RecordActivity()
	if err := h.Provider.EnsureRunning(r.Context()); err != nil {
		writeError(w, http.StatusServiceUnavailable, "gpu_unavailable", "GPU server not available: "+err.Error())
		return false
	}
	return true
}

// ChatCompletions proxies POST /v1/chat/completions to the GPU server.
func (h *ProxyHandler) ChatCompletions(w http.ResponseWriter, r *http.Request) {
	if !h.ensureGPU(w, r) {
		return
	}

	var req api.ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", "failed to parse request body: "+err.Error())
		return
	}

	if req.Stream {
		h.streamProxy(w, r, &req)
	} else {
		h.completeProxy(w, r, &req)
	}
}

func (h *ProxyHandler) completeProxy(w http.ResponseWriter, r *http.Request, req *api.ChatCompletionRequest) {
	resp, err := h.GPUClient.ChatCompletion(r.Context(), req)
	if err != nil {
		writeError(w, http.StatusBadGateway, "gpu_error", err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (h *ProxyHandler) streamProxy(w http.ResponseWriter, r *http.Request, req *api.ChatCompletionRequest) {
	body, err := h.GPUClient.StreamCompletionRaw(r.Context(), req)
	if err != nil {
		writeError(w, http.StatusBadGateway, "gpu_error", err.Error())
		return
	}
	defer body.Close()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if ok {
		flusher.Flush()
	}

	// Stream the raw SSE data through
	buf := make([]byte, 4096)
	for {
		n, err := body.Read(buf)
		if n > 0 {
			w.Write(buf[:n])
			if ok {
				flusher.Flush()
			}
		}
		if err != nil {
			break
		}
	}
}

// Tokenize proxies POST /tokenize to the GPU server.
func (h *ProxyHandler) Tokenize(w http.ResponseWriter, r *http.Request) {
	if !h.ensureGPU(w, r) {
		return
	}

	var req struct {
		Content string `json:"content"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}

	count, err := h.GPUClient.Tokenize(r.Context(), req.Content)
	if err != nil {
		writeError(w, http.StatusBadGateway, "gpu_error", err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]int{"count": count})
}

// ListModels proxies GET /v1/models to the GPU server.
func (h *ProxyHandler) ListModels(w http.ResponseWriter, r *http.Request) {
	if !h.ensureGPU(w, r) {
		return
	}

	result, err := h.GPUClient.ListModels(r.Context())
	if err != nil {
		writeError(w, http.StatusBadGateway, "gpu_error", err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

// LoadModel proxies POST /api/load to the GPU server.
func (h *ProxyHandler) LoadModel(w http.ResponseWriter, r *http.Request) {
	if !h.ensureGPU(w, r) {
		return
	}

	var req struct {
		Model string `json:"model"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}

	if err := h.GPUClient.LoadModel(r.Context(), req.Model); err != nil {
		writeError(w, http.StatusBadGateway, "gpu_error", err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

// PullModel proxies POST /api/pull to the GPU server.
func (h *ProxyHandler) PullModel(w http.ResponseWriter, r *http.Request) {
	if !h.ensureGPU(w, r) {
		return
	}

	var req struct {
		URL string `json:"url"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}

	if err := h.GPUClient.PullModel(r.Context(), req.URL); err != nil {
		writeError(w, http.StatusBadGateway, "gpu_error", err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

// RawProxy forwards a request to the GPU server at the same path and copies the response back.
func (h *ProxyHandler) RawProxy(w http.ResponseWriter, r *http.Request) {
	if !h.ensureGPU(w, r) {
		return
	}

	gpuURL := h.GPUClient.BaseURL() + r.URL.Path
	gpuReq, err := http.NewRequestWithContext(r.Context(), r.Method, gpuURL, r.Body)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "proxy_error", err.Error())
		return
	}
	gpuReq.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(gpuReq)
	if err != nil {
		writeError(w, http.StatusBadGateway, "gpu_error", err.Error())
		return
	}
	defer resp.Body.Close()

	w.Header().Set("Content-Type", resp.Header.Get("Content-Type"))
	w.WriteHeader(resp.StatusCode)
	io.Copy(w, resp.Body)
}

func writeError(w http.ResponseWriter, status int, code, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(api.ErrorResponse{
		Error: api.ErrorDetail{
			Message: message,
			Type:    "error",
			Code:    code,
		},
	})
}

// readBody is a helper for reading and discarding a request body.
func readBody(r *http.Request) ([]byte, error) {
	defer r.Body.Close()
	return io.ReadAll(r.Body)
}
