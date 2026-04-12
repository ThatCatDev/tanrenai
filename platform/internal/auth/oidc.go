package auth

import (
	"context"
	"fmt"

	"github.com/coreos/go-oidc/v3/oidc"
	"golang.org/x/oauth2"
)

// OIDCVerifier validates ID tokens from an OIDC provider.
type OIDCVerifier struct {
	provider *oidc.Provider
	verifier *oidc.IDTokenVerifier
	clientID string
}

// Claims holds the claims extracted from an ID token.
type Claims struct {
	Subject string `json:"sub"`
	Email   string `json:"email"`
	Name    string `json:"name"`
}

// NewOIDCVerifier creates a verifier that discovers the OIDC provider and validates tokens.
func NewOIDCVerifier(ctx context.Context, issuerURL, clientID string) (*OIDCVerifier, error) {
	provider, err := oidc.NewProvider(ctx, issuerURL)
	if err != nil {
		return nil, fmt.Errorf("discover OIDC provider at %s: %w", issuerURL, err)
	}

	verifier := provider.Verifier(&oidc.Config{
		ClientID: clientID,
	})

	return &OIDCVerifier{
		provider: provider,
		verifier: verifier,
		clientID: clientID,
	}, nil
}

// Verify validates a raw ID token and returns the extracted claims.
func (v *OIDCVerifier) Verify(ctx context.Context, rawToken string) (*Claims, error) {
	idToken, err := v.verifier.Verify(ctx, rawToken)
	if err != nil {
		return nil, fmt.Errorf("verify token: %w", err)
	}

	var claims Claims
	if err := idToken.Claims(&claims); err != nil {
		return nil, fmt.Errorf("extract claims: %w", err)
	}

	if claims.Subject == "" {
		return nil, fmt.Errorf("token missing subject claim")
	}

	return &claims, nil
}

// Endpoint returns the OIDC provider's OAuth2 endpoint for authorization code flow.
func (v *OIDCVerifier) Endpoint() oauth2.Endpoint {
	return v.provider.Endpoint()
}
