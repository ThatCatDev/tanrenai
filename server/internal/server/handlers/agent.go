package handlers

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/ThatCatDev/tanrenai/server/internal/agent"
	"github.com/ThatCatDev/tanrenai/server/internal/runner"
	"github.com/ThatCatDev/tanrenai/server/internal/tools"
	"github.com/ThatCatDev/tanrenai/server/pkg/api"
)

// AgentHandler handles POST /v1/agent/completions.
// It runs the full agentic loop server-side: the model can call tools,
// and the server executes them and feeds results back until the model is done.
type AgentHandler struct {
	GetRunner func() runner.Runner
	LoadFunc  func(ctx context.Context, model string) error
	Registry  *tools.Registry
}

func (h *AgentHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	var req api.ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", "failed to parse request body: "+err.Error())
		return
	}

	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "invalid_request", "messages must not be empty")
		return
	}

	// Auto-load the model if needed
	currentRunner := h.GetRunner()
	if currentRunner == nil || (req.Model != "" && normalizeModelName(currentRunner.ModelName()) != normalizeModelName(req.Model)) {
		if req.Model == "" {
			writeError(w, http.StatusBadRequest, "invalid_request", "no model specified and no model loaded")
			return
		}
		if err := h.LoadFunc(r.Context(), req.Model); err != nil {
			writeError(w, http.StatusInternalServerError, "model_error", "failed to load model: "+err.Error())
			return
		}
		currentRunner = h.GetRunner()
	}

	// Build a CompletionFunc from the runner
	rn := currentRunner
	completeFn := func(ctx context.Context, innerReq *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error) {
		innerReq.Model = req.Model
		return rn.ChatCompletion(ctx, innerReq)
	}

	cfg := agent.Config{
		MaxIterations: 20,
		Tools:         h.Registry,
	}

	history, err := agent.Run(r.Context(), completeFn, req.Messages, cfg)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "agent_error", err.Error())
		return
	}

	// Return the final response: last assistant message + full history
	resp := api.ChatCompletionResponse{
		Object: "chat.completion",
		Model:  req.Model,
	}

	// Find the last assistant message
	for i := len(history) - 1; i >= 0; i-- {
		if history[i].Role == "assistant" {
			resp.Choices = []api.Choice{
				{
					Index:        0,
					Message:      history[i],
					FinishReason: "stop",
				},
			}
			break
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}
