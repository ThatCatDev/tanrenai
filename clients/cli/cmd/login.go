package cmd

import (
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"
)

// callbackPort is the fixed port the CLI opens for the browser-driven
// login handshake. The frontend is expected to redirect here with the
// access token as a query string.
const callbackPort = 18293

var loginCmd = &cobra.Command{
	Use:   "login",
	Short: "Authenticate with the tanrenai platform",
	Long: "Opens a browser to the tanrenai web UI, waits for sign-in, and stores " +
		"the resulting session token locally for CLI use. The CLI does not know " +
		"(or care) which auth provider the web UI uses.",
	RunE: func(cmd *cobra.Command, args []string) error {
		platformURL := firstNonEmpty(
			flagString(cmd, "platform-url"),
			os.Getenv("TANRENAI_SERVER_URL"),
			serverURL,
		)

		frontendURL := firstNonEmpty(
			flagString(cmd, "web-url"),
			os.Getenv("TANRENAI_WEB_URL"),
		)
		if frontendURL == "" {
			return fmt.Errorf("--web-url is required (or set TANRENAI_WEB_URL), e.g. https://dev.tanrenai.com")
		}
		frontendURL = strings.TrimRight(frontendURL, "/")

		listener, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", callbackPort))
		if err != nil {
			return fmt.Errorf("start callback server on port %d (is another login running?): %w", callbackPort, err)
		}

		callbackURL := fmt.Sprintf("http://localhost:%d/callback", callbackPort)
		loginURL := fmt.Sprintf("%s/cli-login?callback=%s", frontendURL, url.QueryEscape(callbackURL))

		type result struct {
			accessToken  string
			refreshToken string
			expiresAt    time.Time
			err          error
		}
		resultCh := make(chan result, 1)

		mux := http.NewServeMux()
		mux.HandleFunc("/callback", func(w http.ResponseWriter, r *http.Request) {
			q := r.URL.Query()

			if errParam := q.Get("error"); errParam != "" {
				desc := q.Get("error_description")
				http.Error(w, "login failed: "+errParam+" "+desc, http.StatusBadRequest)
				resultCh <- result{err: fmt.Errorf("login failed: %s %s", errParam, desc)}
				return
			}

			access := q.Get("access_token")
			if access == "" {
				http.Error(w, "missing access_token", http.StatusBadRequest)
				resultCh <- result{err: fmt.Errorf("callback missing access_token")}
				return
			}

			var expiresAt time.Time
			if v := q.Get("expires_at"); v != "" {
				if n, err := strconv.ParseInt(v, 10, 64); err == nil {
					expiresAt = time.Unix(n, 0)
				}
			}
			if expiresAt.IsZero() {
				if v := q.Get("expires_in"); v != "" {
					if n, err := strconv.ParseInt(v, 10, 64); err == nil {
						expiresAt = time.Now().Add(time.Duration(n) * time.Second)
					}
				}
			}

			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			_, _ = w.Write([]byte(`<!doctype html><meta charset=utf-8>
<title>tanrenai CLI login</title>
<body style="font-family:system-ui;padding:3rem;background:#0f1419;color:#e6e6e6">
<h2>Signed in.</h2><p>You can close this tab.</p></body>`))

			resultCh <- result{
				accessToken:  access,
				refreshToken: q.Get("refresh_token"),
				expiresAt:    expiresAt,
			}
		})

		srv := &http.Server{Handler: mux, ReadHeaderTimeout: 5 * time.Second}
		go func() { _ = srv.Serve(listener) }()
		defer func() { _ = srv.Close() }()

		fmt.Printf("Opening browser for sign-in...\n")
		fmt.Printf("If the browser doesn't open, visit:\n  %s\n\n", loginURL)
		_ = openBrowser(loginURL)

		var r result
		select {
		case r = <-resultCh:
		case <-time.After(5 * time.Minute):
			return fmt.Errorf("login timed out — no callback received")
		}
		if r.err != nil {
			return r.err
		}

		creds := &Credentials{
			ServerURL:    platformURL,
			AccessToken:  r.accessToken,
			RefreshToken: r.refreshToken,
			ExpiresAt:    r.expiresAt,
		}
		if err := saveCredentials(creds); err != nil {
			return fmt.Errorf("save credentials: %w", err)
		}

		fmt.Println("Logged in.")
		fmt.Printf("Platform: %s\n", platformURL)
		if !r.expiresAt.IsZero() {
			fmt.Printf("Token expires: %s\n", r.expiresAt.Local().Format(time.RFC3339))
		}
		return nil
	},
}

var logoutCmd = &cobra.Command{
	Use:   "logout",
	Short: "Remove stored credentials",
	RunE: func(cmd *cobra.Command, args []string) error {
		if err := deleteCredentials(); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("remove credentials: %w", err)
		}
		fmt.Println("Logged out.")
		return nil
	},
}

func init() {
	loginCmd.Flags().String("platform-url", "", "Platform API URL (defaults to --server-url or TANRENAI_SERVER_URL)")
	loginCmd.Flags().String("web-url", "", "tanrenai web UI URL, used for the sign-in flow (or set TANRENAI_WEB_URL)")
	rootCmd.AddCommand(loginCmd)
	rootCmd.AddCommand(logoutCmd)
}

func flagString(cmd *cobra.Command, name string) string {
	v, _ := cmd.Flags().GetString(name)
	return v
}

func firstNonEmpty(vs ...string) string {
	for _, v := range vs {
		if v != "" {
			return v
		}
	}
	return ""
}

func openBrowser(u string) error {
	switch runtime.GOOS {
	case "linux":
		return exec.Command("xdg-open", u).Start()
	case "darwin":
		return exec.Command("open", u).Start()
	case "windows":
		return exec.Command("rundll32", "url.dll,FileProtocolHandler", u).Start()
	}
	return nil
}
