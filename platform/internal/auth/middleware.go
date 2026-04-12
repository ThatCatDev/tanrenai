package auth

import (
	"encoding/json"
	"log/slog"
	"net/http"
	"strings"

	"github.com/ThatCatDev/tanrenai/platform/internal/database"
)

// Middleware validates OIDC tokens and injects the user into the request context.
type Middleware struct {
	verifier *OIDCVerifier
	db       *database.DB
}

// NewMiddleware creates auth middleware that validates tokens and auto-registers users.
func NewMiddleware(verifier *OIDCVerifier, db *database.DB) *Middleware {
	return &Middleware{verifier: verifier, db: db}
}

// Wrap returns an HTTP handler that requires authentication.
func (m *Middleware) Wrap(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		token := extractBearerToken(r)
		if token == "" {
			writeAuthError(w, http.StatusUnauthorized, "missing or invalid Authorization header")
			return
		}

		claims, err := m.verifier.Verify(r.Context(), token)
		if err != nil {
			slog.Debug("token verification failed", "error", err)
			writeAuthError(w, http.StatusUnauthorized, "invalid or expired token")
			return
		}

		// Auto-register or update user on every authenticated request
		name := claims.Name
		if name == "" {
			name = claims.Email
		}

		user, err := m.db.CreateOrUpdateUser(r.Context(), claims.Subject, claims.Email, name)
		if err != nil {
			slog.Error("failed to upsert user", "error", err, "sub", claims.Subject)
			writeAuthError(w, http.StatusInternalServerError, "internal error")
			return
		}

		ctx := ContextWithUser(r.Context(), user)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// WrapFunc is a convenience wrapper for http.HandlerFunc.
func (m *Middleware) WrapFunc(next http.HandlerFunc) http.Handler {
	return m.Wrap(http.HandlerFunc(next))
}

func extractBearerToken(r *http.Request) string {
	auth := r.Header.Get("Authorization")
	if auth == "" {
		return ""
	}
	parts := strings.SplitN(auth, " ", 2)
	if len(parts) != 2 || !strings.EqualFold(parts[0], "bearer") {
		return ""
	}
	return strings.TrimSpace(parts[1])
}

func writeAuthError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{
		"error":   "unauthorized",
		"message": message,
	})
}
