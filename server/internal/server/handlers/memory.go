package handlers

import (
	"encoding/json"
	"errors"
	"net/http"
	"strconv"

	"github.com/ThatCatDev/tanrenai/server/internal/memory"
	"github.com/ThatCatDev/tanrenai/server/pkg/api"
)

// MemoryHandler handles memory CRUD endpoints.
type MemoryHandler struct {
	MemStore memory.Store
}

// Search handles POST /v1/memory/search.
func (h *MemoryHandler) Search(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20) // 1MB limit

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
		writeMemoryError(w, err)
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
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20) // 1MB limit

	var req api.MemoryStoreRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}

	entry := memory.Entry{
		UserMsg:   req.UserMsg,
		AssistMsg: req.AssistMsg,
	}

	if err := h.MemStore.Add(r.Context(), &entry); err != nil {
		writeMemoryError(w, err)
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
		writeMemoryError(w, err)
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
	id := r.PathValue("id")
	if id == "" {
		writeError(w, http.StatusBadRequest, "invalid_request", "memory ID required")
		return
	}

	if err := h.MemStore.Delete(r.Context(), id); err != nil {
		writeMemoryError(w, err)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "deleted"})
}

// Clear handles DELETE /v1/memory.
func (h *MemoryHandler) Clear(w http.ResponseWriter, r *http.Request) {
	if err := h.MemStore.Clear(r.Context()); err != nil {
		writeMemoryError(w, err)
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

// writeMemoryError maps memory sentinel errors to appropriate HTTP status codes.
func writeMemoryError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, memory.ErrNotFound):
		writeError(w, http.StatusNotFound, "not_found", err.Error())
	case errors.Is(err, memory.ErrEmpty):
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
	default:
		writeError(w, http.StatusInternalServerError, "memory_error", err.Error())
	}
}
