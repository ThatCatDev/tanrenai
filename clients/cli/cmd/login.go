package cmd

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"net"
	"net/http"
	"os/exec"
	"runtime"
	"time"

	"github.com/spf13/cobra"
	"golang.org/x/oauth2"
)

var loginCmd = &cobra.Command{
	Use:   "login",
	Short: "Authenticate with the tanrenai platform",
	Long:  "Opens a browser to log in via OIDC (Dex). Stores tokens locally for CLI use.",
	RunE: func(cmd *cobra.Command, args []string) error {
		platformURL, _ := cmd.Flags().GetString("platform-url")
		clientID, _ := cmd.Flags().GetString("client-id")
		issuer, _ := cmd.Flags().GetString("oidc-issuer")

		if issuer == "" {
			return fmt.Errorf("--oidc-issuer is required (e.g. http://localhost:5556/dex)")
		}

		// Generate PKCE verifier and challenge
		verifier := generateCodeVerifier()
		challenge := generateCodeChallenge(verifier)

		// Start local callback server on a fixed port (must match Dex client config)
		const callbackPort = 18293
		listener, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", callbackPort))
		if err != nil {
			return fmt.Errorf("start callback server on port %d (is another login running?): %w", callbackPort, err)
		}
		redirectURL := fmt.Sprintf("http://localhost:%d/callback", callbackPort)

		cfg := &oauth2.Config{
			ClientID:    clientID,
			Endpoint:    oauth2.Endpoint{
				AuthURL:  issuer + "/auth",
				TokenURL: issuer + "/token",
			},
			RedirectURL: redirectURL,
			Scopes:      []string{"openid", "email", "profile", "offline_access"},
		}

		// Generate auth URL
		state := generateState()
		authURL := cfg.AuthCodeURL(state,
			oauth2.SetAuthURLParam("code_challenge", challenge),
			oauth2.SetAuthURLParam("code_challenge_method", "S256"),
		)

		fmt.Printf("Opening browser for login...\n")
		fmt.Printf("If the browser doesn't open, visit:\n  %s\n\n", authURL)
		_ = openBrowser(authURL)

		// Wait for callback
		codeCh := make(chan string, 1)
		errCh := make(chan error, 1)

		mux := http.NewServeMux()
		mux.HandleFunc("/callback", func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Query().Get("state") != state {
				http.Error(w, "invalid state", http.StatusBadRequest)
				errCh <- fmt.Errorf("state mismatch")
				return
			}
			code := r.URL.Query().Get("code")
			if code == "" {
				errDesc := r.URL.Query().Get("error_description")
				http.Error(w, "login failed", http.StatusBadRequest)
				errCh <- fmt.Errorf("no authorization code: %s", errDesc)
				return
			}
			fmt.Fprintf(w, "<html><body><h2>Login successful!</h2><p>You can close this tab.</p></body></html>")
			codeCh <- code
		})

		srv := &http.Server{Handler: mux}
		go func() { _ = srv.Serve(listener) }()

		// Wait for code or timeout
		var code string
		select {
		case code = <-codeCh:
		case err := <-errCh:
			_ = srv.Close()
			return err
		case <-time.After(2 * time.Minute):
			_ = srv.Close()
			return fmt.Errorf("login timed out — no callback received")
		}
		_ = srv.Close()

		// Exchange code for tokens
		ctx := context.Background()
		token, err := cfg.Exchange(ctx, code,
			oauth2.SetAuthURLParam("code_verifier", verifier),
		)
		if err != nil {
			return fmt.Errorf("exchange code for token: %w", err)
		}

		// Extract ID token
		idToken, ok := token.Extra("id_token").(string)
		if !ok || idToken == "" {
			return fmt.Errorf("no id_token in response")
		}

		// Save credentials
		creds := &Credentials{
			ServerURL:    platformURL,
			AccessToken:  idToken,
			RefreshToken: token.RefreshToken,
			ExpiresAt:    token.Expiry,
		}
		if err := saveCredentials(creds); err != nil {
			return fmt.Errorf("save credentials: %w", err)
		}

		fmt.Printf("Logged in successfully! Credentials saved.\n")
		fmt.Printf("Platform: %s\n", platformURL)
		return nil
	},
}

func init() {
	loginCmd.Flags().String("platform-url", "http://localhost:3000", "platform service URL")
	loginCmd.Flags().String("client-id", "tanrenai-cli", "OIDC client ID")
	loginCmd.Flags().String("oidc-issuer", "", "OIDC issuer URL (e.g. http://localhost:5556/dex)")
	rootCmd.AddCommand(loginCmd)
}

var logoutCmd = &cobra.Command{
	Use:   "logout",
	Short: "Remove stored credentials",
	RunE: func(cmd *cobra.Command, args []string) error {
		if err := deleteCredentials(); err != nil {
			return fmt.Errorf("remove credentials: %w", err)
		}
		fmt.Println("Logged out. Credentials removed.")
		return nil
	},
}

func init() {
	rootCmd.AddCommand(logoutCmd)
}

func generateCodeVerifier() string {
	b := make([]byte, 32)
	_, _ = rand.Read(b)
	return base64.RawURLEncoding.EncodeToString(b)
}

func generateCodeChallenge(verifier string) string {
	h := sha256.Sum256([]byte(verifier))
	return base64.RawURLEncoding.EncodeToString(h[:])
}

func generateState() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return base64.RawURLEncoding.EncodeToString(b)
}

func openBrowser(url string) error {
	switch runtime.GOOS {
	case "linux":
		return exec.Command("xdg-open", url).Start()
	case "darwin":
		return exec.Command("open", url).Start()
	case "windows":
		return exec.Command("rundll32", "url.dll,FileProtocolHandler", url).Start()
	}
	return nil
}
