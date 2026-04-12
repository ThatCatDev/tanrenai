package auth

import (
	"context"

	"github.com/ThatCatDev/tanrenai/platform/internal/database"
)

type contextKey struct{}

// ContextWithUser stores a user in the context.
func ContextWithUser(ctx context.Context, user *database.User) context.Context {
	return context.WithValue(ctx, contextKey{}, user)
}

// UserFromContext extracts the user from the context. Returns nil if not set.
func UserFromContext(ctx context.Context) *database.User {
	u, _ := ctx.Value(contextKey{}).(*database.User)
	return u
}
