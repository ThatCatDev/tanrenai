package cmd

import (
	"fmt"
	"html"
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
				w.Header().Set("Content-Type", "text/html; charset=utf-8")
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(cliAuthPage("Sign-in failed", strings.TrimSpace(errParam+" "+desc), true)))
				resultCh <- result{err: fmt.Errorf("login failed: %s %s", errParam, desc)}
				return
			}

			access := q.Get("access_token")
			if access == "" {
				w.Header().Set("Content-Type", "text/html; charset=utf-8")
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(cliAuthPage("Sign-in failed",
					"The callback didn't include an access token. Please run the login again.", true)))
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
			_, _ = w.Write([]byte(cliAuthPage("Signed in",
				"Your CLI session is ready — you can close this tab and return to the terminal.", false)))

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

// cliAuthPage renders the standalone page the browser lands on after the CLI
// auth handoff, styled to match the web app's design system (same palette,
// Inter, card on a dark surface). Self-contained — no network assets required.
func cliAuthPage(title, message string, isError bool) string {
	badgeBG, badgeFG, icon := "rgba(0,180,160,.12)", "#00b4a0", cliCheckIcon
	if isError {
		badgeBG, badgeFG, icon = "rgba(255,71,87,.12)", "#ff4757", cliXIcon
	}
	return strings.NewReplacer(
		"{{BADGE_BG}}", badgeBG,
		"{{BADGE_FG}}", badgeFG,
		"{{ICON}}", icon,
		"{{TITLE}}", html.EscapeString(title),
		"{{MESSAGE}}", html.EscapeString(message),
	).Replace(cliAuthPageTmpl)
}

const (
	cliCheckIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>`
	cliXIcon     = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18M6 6l12 12"/></svg>`
)

const cliAuthPageTmpl = `<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tanrenai CLI</title>
<style>
  *{box-sizing:border-box}
  html,body{height:100%;margin:0}
  body{display:grid;place-items:center;min-height:100vh;background:#0e0f11;color:#e4e5e7;
    font-family:'Inter',-apple-system,BlinkMacSystemFont,system-ui,sans-serif;
    -webkit-font-smoothing:antialiased}
  .card{max-width:420px;margin:1.5rem;padding:2.5rem 2rem;text-align:center;
    background:#16171a;border:1px solid #252830;border-radius:8px}
  .badge{width:56px;height:56px;margin:0 auto 1.25rem;border-radius:50%;
    display:grid;place-items:center;background:{{BADGE_BG}};color:{{BADGE_FG}}}
  .badge svg{width:28px;height:28px}
  h1{margin:0 0 .5rem;font-size:1.25rem;font-weight:600;letter-spacing:-.01em}
  p{margin:0;color:#717780;font-size:.875rem;line-height:1.55}
  .mark{margin-top:1.5rem;font-family:'JetBrains Mono',ui-monospace,monospace;
    font-size:.75rem;letter-spacing:.12em;color:#00b4a0}
</style></head>
<body>
  <div class="card">
    <div class="badge">{{ICON}}</div>
    <h1>{{TITLE}}</h1>
    <p>{{MESSAGE}}</p>
    <div class="mark">tanrenai</div>
  </div>
</body></html>`
