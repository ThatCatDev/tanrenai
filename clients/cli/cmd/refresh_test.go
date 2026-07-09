package cmd

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestRefreshUsesNewerOnDiskCredentials: if another process already rotated
// the pair (disk holds a different, still-valid token), refreshCredentials
// must return the on-disk pair without a network round-trip — hitting the
// endpoint with our stale token would burn a rotation and can revoke the
// token family.
func TestRefreshUsesNewerOnDiskCredentials(t *testing.T) {
	t.Setenv("TANRENAI_DATA_DIR", t.TempDir())

	onDisk := &Credentials{
		ServerURL:    "http://127.0.0.1:1", // unroutable — must not be contacted
		AccessToken:  "access-new",
		RefreshToken: "refresh-new",
		ExpiresAt:    time.Now().Add(time.Hour),
	}
	if err := saveCredentials(onDisk); err != nil {
		t.Fatal(err)
	}

	stale := &Credentials{
		ServerURL:    "http://127.0.0.1:1",
		AccessToken:  "access-old",
		RefreshToken: "refresh-old",
		ExpiresAt:    time.Now().Add(-time.Minute),
	}
	got, err := refreshCredentials(stale)
	if err != nil {
		t.Fatalf("refreshCredentials: %v", err)
	}
	if got.AccessToken != "access-new" || got.RefreshToken != "refresh-new" {
		t.Errorf("expected on-disk pair, got access=%q refresh=%q", got.AccessToken, got.RefreshToken)
	}
}

// TestRefreshDeadTokenIsSessionExpired: a 400 "Invalid Refresh Token" must
// surface as errSessionExpired so callers can tell the user to re-login
// instead of dumping a raw status.
func TestRefreshDeadTokenIsSessionExpired(t *testing.T) {
	t.Setenv("TANRENAI_DATA_DIR", t.TempDir())

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"error_description": "Invalid Refresh Token: Already Used",
		})
	}))
	defer srv.Close()

	creds := &Credentials{
		ServerURL:    srv.URL,
		AccessToken:  "access-old",
		RefreshToken: "refresh-old",
		ExpiresAt:    time.Now().Add(-time.Minute),
	}
	if err := saveCredentials(creds); err != nil {
		t.Fatal(err)
	}

	_, err := refreshCredentials(creds)
	if !errors.Is(err, errSessionExpired) {
		t.Errorf("expected errSessionExpired, got %v", err)
	}
}

// TestRefreshRotatesAndSaves: a successful refresh persists the rotated pair.
func TestRefreshRotatesAndSaves(t *testing.T) {
	t.Setenv("TANRENAI_DATA_DIR", t.TempDir())

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(refreshResponse{
			AccessToken:  "access-2",
			RefreshToken: "refresh-2",
			ExpiresIn:    3600,
		})
	}))
	defer srv.Close()

	creds := &Credentials{
		ServerURL:    srv.URL,
		AccessToken:  "access-1",
		RefreshToken: "refresh-1",
		ExpiresAt:    time.Now().Add(-time.Minute),
	}
	if err := saveCredentials(creds); err != nil {
		t.Fatal(err)
	}

	got, err := refreshCredentials(creds)
	if err != nil {
		t.Fatalf("refreshCredentials: %v", err)
	}
	if got.AccessToken != "access-2" || got.RefreshToken != "refresh-2" {
		t.Errorf("unexpected pair after refresh: access=%q refresh=%q", got.AccessToken, got.RefreshToken)
	}

	onDisk, err := loadCredentials()
	if err != nil {
		t.Fatal(err)
	}
	if onDisk.AccessToken != "access-2" || onDisk.RefreshToken != "refresh-2" {
		t.Errorf("rotated pair not persisted: access=%q refresh=%q", onDisk.AccessToken, onDisk.RefreshToken)
	}
}

// TestRefreshConcurrentSingleRotation: N concurrent refreshes of the same
// stale pair must produce exactly one rotation server-side — the losers of
// the lock race re-read the winner's pair from disk and skip the network.
// This is the regression test for "Invalid Refresh Token: Already Used".
func TestRefreshConcurrentSingleRotation(t *testing.T) {
	t.Setenv("TANRENAI_DATA_DIR", t.TempDir())

	var rotations atomic.Int32
	var mu sync.Mutex
	current := "refresh-1"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			RefreshToken string `json:"refresh_token"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		mu.Lock()
		defer mu.Unlock()
		if body.RefreshToken != current {
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(map[string]string{
				"error_description": "Invalid Refresh Token: Already Used",
			})
			return
		}
		rotations.Add(1)
		current = "refresh-2"
		_ = json.NewEncoder(w).Encode(refreshResponse{
			AccessToken:  "access-2",
			RefreshToken: "refresh-2",
			ExpiresIn:    3600,
		})
	}))
	defer srv.Close()

	stale := Credentials{
		ServerURL:    srv.URL,
		AccessToken:  "access-1",
		RefreshToken: "refresh-1",
		ExpiresAt:    time.Now().Add(-time.Minute),
	}
	if err := saveCredentials(&stale); err != nil {
		t.Fatal(err)
	}

	const n = 5
	var wg sync.WaitGroup
	results := make([]*Credentials, n)
	errs := make([]error, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			c := stale // each caller starts from its own stale copy
			results[i], errs[i] = refreshCredentials(&c)
		}(i)
	}
	wg.Wait()

	for i := 0; i < n; i++ {
		if errs[i] != nil {
			t.Errorf("caller %d: %v", i, errs[i])
			continue
		}
		if results[i].AccessToken != "access-2" {
			t.Errorf("caller %d: got access token %q, want access-2", i, results[i].AccessToken)
		}
	}
	if got := rotations.Load(); got != 1 {
		t.Errorf("server saw %d rotations, want exactly 1", got)
	}
}

func TestIsDeadRefreshToken(t *testing.T) {
	cases := []struct {
		status int
		msg    string
		want   bool
	}{
		{400, "Invalid Refresh Token: Already Used", true},
		{401, "Refresh Token Not Found", true},
		{403, "invalid refresh_token", true},
		{400, "malformed request", false},
		{500, "Invalid Refresh Token", false},
		{502, "bad gateway", false},
	}
	for _, c := range cases {
		if got := isDeadRefreshToken(c.status, c.msg); got != c.want {
			t.Errorf("isDeadRefreshToken(%d, %q) = %v, want %v", c.status, c.msg, got, c.want)
		}
	}
}

// TestSaveCredentialsRestrictedPerms: the atomic write path must still land
// the file with 0600.
func TestSaveCredentialsRestrictedPerms(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("unix permission bits")
	}
	t.Setenv("TANRENAI_DATA_DIR", t.TempDir())

	if err := saveCredentials(&Credentials{AccessToken: "a"}); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(credentialsPath())
	if err != nil {
		t.Fatal(err)
	}
	if perm := info.Mode().Perm(); perm != 0600 {
		t.Errorf("credentials perm = %o, want 0600", perm)
	}
}
