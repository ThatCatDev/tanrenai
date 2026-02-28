package handlers

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"

	"github.com/ThatCatDev/tanrenai/server/internal/memory"
	"github.com/ThatCatDev/tanrenai/server/pkg/api"
)

// MemoryHandler handles memory CRUD endpoints.
type MemoryHandler struct {
	MemStore memory.Store
}

// Search handles POST /v1/memory/search.
func (h *MemoryHandler) Search(w http.ResponseWriter, r *http.Request) {
	var req api.MemorySearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}

	if req.Query == "" {
		writeError(w, http.StatusBadRequest, "invalid_request", "query must not be empty")
		return
	}

	results, err := h.MemStore.Search(r.Context(), req.Query, req.Limit)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "memory_error", err.Error())
		return
	}

	// Convert to API types
	apiResults := make([]api.MemorySearchResult, len(results))
	for i, sr := range results {
		apiResults[i] = api.MemorySearchResult{
			Entry: api.MemoryEntry{
				ID:        sr.Entry.ID,
				UserMsg:   sr.Entry.UserMsg,
				AssistMsg: sr.Entry.AssistMsg,
				Timestamp: sr.Entry.Timestamp,
				SessionID: sr.Entry.SessionID,
			},
			SemanticScore: sr.SemanticScore,
			KeywordScore:  sr.KeywordScore,
			CombinedScore: sr.CombinedScore,
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(api.MemorySearchResponse{Results: apiResults})
}

// Store handles POST /v1/memory/store.
func (h *MemoryHandler) Store(w http.ResponseWriter, r *http.Request) {
	var req api.MemoryStoreRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}

	entry := memory.Entry{
		UserMsg:   req.UserMsg,
		AssistMsg: req.AssistMsg,
	}

	if err := h.MemStore.Add(r.Context(), entry); err != nil {
		writeError(w, http.StatusInternalServerError, "memory_error", err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(api.MemoryStoreResponse{ID: entry.ID})
}

// List handles GET /v1/memory/list.
func (h *MemoryHandler) List(w http.ResponseWriter, r *http.Request) {
	limit := 0
	if l := r.URL.Query().Get("limit"); l != "" {
		limit, _ = strconv.Atoi(l)
	}

	entries, err := h.MemStore.List(r.Context(), limit)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "memory_error", err.Error())
		return
	}

	apiEntries := make([]api.MemoryEntry, len(entries))
	for i, e := range entries {
		apiEntries[i] = api.MemoryEntry{
			ID:        e.ID,
			UserMsg:   e.UserMsg,
			AssistMsg: e.AssistMsg,
			Timestamp: e.Timestamp,
			SessionID: e.SessionID,
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(api.MemoryListResponse{
		Entries: apiEntries,
		Total:   h.MemStore.Count(),
	})
}

// Delete handles DELETE /v1/memory/{id}.
func (h *MemoryHandler) Delete(w http.ResponseWriter, r *http.Request) {
	// Extract id from path: /v1/memory/{id}
	path := r.URL.Path
	parts := strings.Split(strings.TrimPrefix(path, "/v1/memory/"), "/")
	if len(parts) == 0 || parts[0] == "" {
		writeError(w, http.StatusBadRequest, "invalid_request", "memory ID required")
		return
	}
	id := parts[0]

	if err := h.MemStore.Delete(r.Context(), id); err != nil {
		writeError(w, http.StatusInternalServerError, "memory_error", err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "deleted"})
}

// Clear handles DELETE /v1/memory.
func (h *MemoryHandler) Clear(w http.ResponseWriter, r *http.Request) {
	if err := h.MemStore.Clear(r.Context()); err != nil {
		writeError(w, http.StatusInternalServerError, "memory_error", err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "cleared"})
}

// Count handles GET /v1/memory/count.
func (h *MemoryHandler) Count(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(api.MemoryCountResponse{Count: h.MemStore.Count()})
}
