package cmd

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	"github.com/ThatCatDev/tanrenai/shared/mcp"
)

// runMCPSubcommand invokes the `mcp` subtree with a fresh command
// tree so flag state from a previous test doesn't bleed in. Returns
// stdout, stderr, and the run error.
//
// Working directory is set to `workspace` so `mcp add` writes to
// <workspace>/.tanrenai/mcp.json (the project scope). User-scope
// tests override mcp.UserConfigPath via the package-level
// userConfigDir hook in the shared package — but since that hook is
// not exported here, user-scope tests do all the writes via
// mcp.SaveFile directly and just exercise read/list paths through
// the command.
func runMCPSubcommand(t *testing.T, workspace string, args ...string) (string, string, error) {
	t.Helper()

	// Chdir to workspace so project-scope subcommands resolve there.
	// t.Chdir restores on cleanup so test ordering is independent.
	t.Chdir(workspace)

	// Cobra commands hold global flag state across invocations; build
	// a fresh tree pointing at the same RunE handlers we register in
	// init(). For tests we only need the mcp branch.
	root := &cobra.Command{Use: "tanrenai"}
	mcpClone := &cobra.Command{Use: "mcp"}

	addClone := &cobra.Command{Use: "add", Args: cobra.MinimumNArgs(1), RunE: runMCPAdd}
	addClone.Flags().String("scope", "project", "")
	addClone.Flags().String("transport", "stdio", "")
	addClone.Flags().StringSliceP("header", "H", nil, "")
	addClone.Flags().StringSliceP("env", "e", nil, "")
	addClone.Flags().Bool("disabled", false, "")

	listClone := &cobra.Command{Use: "list", RunE: runMCPList}
	listClone.Flags().String("scope", "merged", "")

	removeClone := &cobra.Command{Use: "remove", Args: cobra.ExactArgs(1), RunE: runMCPRemove}
	removeClone.Flags().String("scope", "project", "")

	mcpClone.AddCommand(addClone, listClone, removeClone)
	root.AddCommand(mcpClone)

	var stdout, stderr bytes.Buffer
	root.SetOut(&stdout)
	root.SetErr(&stderr)
	root.SetArgs(append([]string{"mcp"}, args...))
	// Swap os.Stdout briefly so the run* handlers' Fprintln calls land
	// in the buffer too — cobra's SetOut only captures cmd.Println,
	// but our handlers use fmt.Fprintln(os.Stdout, ...) for output.
	origStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w
	done := make(chan struct{})
	go func() {
		var buf bytes.Buffer
		_, _ = buf.ReadFrom(r)
		stdout.Write(buf.Bytes())
		close(done)
	}()

	err := root.Execute()

	w.Close()
	<-done
	os.Stdout = origStdout

	return stdout.String(), stderr.String(), err
}

func TestMCPAdd_StdioWritesProjectConfig(t *testing.T) {
	ws := t.TempDir()
	out, _, err := runMCPSubcommand(t, ws,
		"add", "playwright", "--", "npx", "-y", "@playwright/mcp@latest",
	)
	if err != nil {
		t.Fatalf("add: %v\nstdout: %s", err, out)
	}
	if !strings.Contains(out, "Added MCP server \"playwright\"") {
		t.Errorf("missing success line in stdout: %q", out)
	}

	cfg, err := mcp.LoadFile(filepath.Join(ws, ".tanrenai", "mcp.json"))
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}
	srv, ok := cfg.Servers["playwright"]
	if !ok {
		t.Fatal("playwright server not persisted")
	}
	if srv.Command != "npx" {
		t.Errorf("command = %q, want npx", srv.Command)
	}
	if got, want := srv.Args, []string{"-y", "@playwright/mcp@latest"}; !equalSlice(got, want) {
		t.Errorf("args = %v, want %v", got, want)
	}
	if srv.URL != "" {
		t.Errorf("stdio server should not have URL set, got %q", srv.URL)
	}
}

func TestMCPAdd_HTTPWithHeaders(t *testing.T) {
	ws := t.TempDir()
	out, _, err := runMCPSubcommand(t, ws,
		"add", "github",
		"--transport", "http",
		"https://api.example.com/mcp",
		"-H", "Authorization: Bearer secret",
		"-H", "X-API-Key: abcd",
	)
	if err != nil {
		t.Fatalf("add: %v\nstdout: %s", err, out)
	}
	cfg, err := mcp.LoadFile(filepath.Join(ws, ".tanrenai", "mcp.json"))
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}
	srv := cfg.Servers["github"]
	if srv.URL != "https://api.example.com/mcp" {
		t.Errorf("url = %q", srv.URL)
	}
	if srv.Headers["Authorization"] != "Bearer secret" {
		t.Errorf("Authorization header missing/wrong: %v", srv.Headers)
	}
	if srv.Headers["X-API-Key"] != "abcd" {
		t.Errorf("X-API-Key header missing/wrong: %v", srv.Headers)
	}
}

func TestMCPAdd_RejectsDuplicates(t *testing.T) {
	ws := t.TempDir()
	if _, _, err := runMCPSubcommand(t, ws, "add", "foo", "--", "echo"); err != nil {
		t.Fatalf("first add: %v", err)
	}
	// Second add of the same name should error rather than silently
	// overwrite — keeps config drift visible.
	_, _, err := runMCPSubcommand(t, ws, "add", "foo", "--", "echo")
	if err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Errorf("expected duplicate error, got %v", err)
	}
}

func TestMCPAdd_RejectsInvalidName(t *testing.T) {
	ws := t.TempDir()
	// Names containing `__` collide with the registry's namespace
	// separator and would break dispatch on the agent side.
	_, _, err := runMCPSubcommand(t, ws, "add", "bad__name", "--", "echo")
	if err == nil || !strings.Contains(err.Error(), "invalid") {
		t.Errorf("expected invalid-name error, got %v", err)
	}
}

func TestMCPAdd_StdioRequiresCommand(t *testing.T) {
	ws := t.TempDir()
	// Just `tanrenai mcp add foo` with no command is meaningless;
	// surface a helpful error rather than persist a broken row.
	_, _, err := runMCPSubcommand(t, ws, "add", "foo")
	if err == nil || !strings.Contains(err.Error(), "command") {
		t.Errorf("expected needs-command error, got %v", err)
	}
}

func TestMCPAdd_HTTPRequiresURL(t *testing.T) {
	ws := t.TempDir()
	_, _, err := runMCPSubcommand(t, ws, "add", "foo", "--transport", "http")
	if err == nil || !strings.Contains(err.Error(), "URL") {
		t.Errorf("expected needs-URL error, got %v", err)
	}
}

func TestMCPAdd_InvalidHeaderFormat(t *testing.T) {
	ws := t.TempDir()
	_, _, err := runMCPSubcommand(t, ws,
		"add", "foo",
		"--transport", "http",
		"https://x",
		"-H", "no-colon-here",
	)
	if err == nil || !strings.Contains(err.Error(), "header") {
		t.Errorf("expected invalid-header error, got %v", err)
	}
}

func TestMCPList_Empty(t *testing.T) {
	ws := t.TempDir()
	out, _, err := runMCPSubcommand(t, ws, "list")
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if !strings.Contains(out, "no MCP servers configured") {
		t.Errorf("expected empty message, got %q", out)
	}
}

func TestMCPList_TableContent(t *testing.T) {
	ws := t.TempDir()
	if _, _, err := runMCPSubcommand(t, ws, "add", "alpha", "--", "echo"); err != nil {
		t.Fatalf("add alpha: %v", err)
	}
	if _, _, err := runMCPSubcommand(t, ws,
		"add", "beta", "--transport", "http", "https://example.com/mcp",
	); err != nil {
		t.Fatalf("add beta: %v", err)
	}

	out, _, err := runMCPSubcommand(t, ws, "list", "--scope", "project")
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	// Sorted output — alpha before beta. Each line carries the
	// transport so the user can scan stdio vs http at a glance.
	for _, want := range []string{"NAME", "TRANSPORT", "alpha", "stdio", "beta", "http"} {
		if !strings.Contains(out, want) {
			t.Errorf("list output missing %q\n%s", want, out)
		}
	}
	idxAlpha := strings.Index(out, "alpha")
	idxBeta := strings.Index(out, "beta")
	if idxAlpha < 0 || idxBeta < 0 || idxAlpha >= idxBeta {
		t.Errorf("alpha should sort before beta in list output:\n%s", out)
	}
}

func TestMCPList_DisabledTag(t *testing.T) {
	ws := t.TempDir()
	if _, _, err := runMCPSubcommand(t, ws,
		"add", "x", "--disabled", "--", "echo",
	); err != nil {
		t.Fatalf("add: %v", err)
	}
	out, _, err := runMCPSubcommand(t, ws, "list", "--scope", "project")
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if !strings.Contains(out, "disabled") {
		t.Errorf("disabled flag should surface as a notes tag:\n%s", out)
	}
}

func TestMCPRemove(t *testing.T) {
	ws := t.TempDir()
	if _, _, err := runMCPSubcommand(t, ws, "add", "x", "--", "echo"); err != nil {
		t.Fatalf("add: %v", err)
	}
	out, _, err := runMCPSubcommand(t, ws, "remove", "x")
	if err != nil {
		t.Fatalf("remove: %v", err)
	}
	if !strings.Contains(out, "Removed MCP server") {
		t.Errorf("expected success line, got %q", out)
	}
	cfg, err := mcp.LoadFile(filepath.Join(ws, ".tanrenai", "mcp.json"))
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}
	if _, ok := cfg.Servers["x"]; ok {
		t.Error("server should be gone after remove")
	}
}

func TestMCPRemove_NotFound(t *testing.T) {
	ws := t.TempDir()
	_, _, err := runMCPSubcommand(t, ws, "remove", "ghost")
	if err == nil || !strings.Contains(err.Error(), "not found") {
		t.Errorf("expected not-found error, got %v", err)
	}
}

func TestMCPAdd_EnvFlags(t *testing.T) {
	ws := t.TempDir()
	out, _, err := runMCPSubcommand(t, ws,
		"add", "foo", "-e", "API_KEY=xyz", "-e", "EMPTY=",
		"--", "/bin/echo",
	)
	if err != nil {
		t.Fatalf("add: %v\nstdout: %s", err, out)
	}
	cfg, err := mcp.LoadFile(filepath.Join(ws, ".tanrenai", "mcp.json"))
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}
	srv := cfg.Servers["foo"]
	if srv.Env["API_KEY"] != "xyz" {
		t.Errorf("API_KEY env not set: %v", srv.Env)
	}
	if _, ok := srv.Env["EMPTY"]; !ok {
		t.Errorf("EMPTY env (with empty value) should still be recorded: %v", srv.Env)
	}
}

func equalSlice(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Silence the linter on the imported errors package — kept for future
// errors.Is checks as the test suite grows.
var _ = errors.New
