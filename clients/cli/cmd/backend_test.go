package cmd

import (
	"context"
	"errors"
	"testing"

	"github.com/spf13/cobra"
)

// runWithBackendCmd builds a cobra command tree that mirrors how rootCmd
// wires --local as a persistent flag, then executes the subcommand with
// the given args so cobra merges inherited persistent flags into the
// subcommand's flag set — the same merging step that lets the real pull
// and list commands read --local in production. The body fn receives the
// subcommand after the merge is done.
func runWithBackendCmd(t *testing.T, args []string, body func(cmd *cobra.Command)) {
	t.Helper()
	root := &cobra.Command{Use: "root"}
	root.PersistentFlags().Bool("local", false, "")
	root.PersistentFlags().Int("gpu-layers", -1, "")
	root.PersistentFlags().Bool("flash-attn", true, "")
	sub := &cobra.Command{
		Use: "sub",
		RunE: func(cmd *cobra.Command, _ []string) error {
			body(cmd)
			return nil
		},
	}
	root.AddCommand(sub)
	root.SetArgs(append([]string{"sub"}, args...))
	if err := root.Execute(); err != nil {
		t.Fatalf("cobra execute: %v", err)
	}
}

// TestResolveBackend_RemoteDefault confirms that without --local the
// command uses the package-level serverURL and returns a cleanup that's
// safe to call.
func TestResolveBackend_RemoteDefault(t *testing.T) {
	prev := serverURL
	serverURL = "http://example-backend:8080"
	defer func() { serverURL = prev }()

	runWithBackendCmd(t, nil, func(cmd *cobra.Command) {
		url, cleanup, err := resolveBackend(t.Context(), cmd)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if url != "http://example-backend:8080" {
			t.Errorf("url = %q, want %q", url, "http://example-backend:8080")
		}
		cleanup()
	})
}

// TestResolveBackend_LocalStartsEmbedded is the regression test for the
// `tanrenai --local list` bug: when --local is true, resolveBackend must
// invoke the local-startup function instead of returning the default
// serverURL. We swap startLocalServersFn for a fake so we don't actually
// boot GPU/backend processes in unit tests.
func TestResolveBackend_LocalStartsEmbedded(t *testing.T) {
	called := false
	cleanupCalled := false
	prevFn := startLocalServersFn
	startLocalServersFn = func(_ context.Context, _ localOpts, _ *startupLog) (string, func(), error) {
		called = true
		return "http://127.0.0.1:55555", func() { cleanupCalled = true }, nil
	}
	defer func() { startLocalServersFn = prevFn }()

	prev := serverURL
	serverURL = "http://this-should-be-ignored"
	defer func() { serverURL = prev }()

	runWithBackendCmd(t, []string{"--local"}, func(cmd *cobra.Command) {
		url, cleanup, err := resolveBackend(t.Context(), cmd)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !called {
			t.Error("expected startLocalServersFn to be called when --local is true")
		}
		if url != "http://127.0.0.1:55555" {
			t.Errorf("url = %q, want the embedded backend URL", url)
		}
		cleanup()
		if !cleanupCalled {
			t.Error("expected returned cleanup to invoke the embedded cleanup")
		}
	})
}

// TestResolveBackend_LocalPropagatesStartupError ensures that a startup
// failure surfaces to the caller rather than silently falling back to the
// remote default.
func TestResolveBackend_LocalPropagatesStartupError(t *testing.T) {
	wantErr := errors.New("port in use")
	prevFn := startLocalServersFn
	startLocalServersFn = func(_ context.Context, _ localOpts, _ *startupLog) (string, func(), error) {
		return "", nil, wantErr
	}
	defer func() { startLocalServersFn = prevFn }()

	runWithBackendCmd(t, []string{"--local"}, func(cmd *cobra.Command) {
		_, _, err := resolveBackend(t.Context(), cmd)
		if !errors.Is(err, wantErr) {
			t.Errorf("err = %v, want %v", err, wantErr)
		}
	})
}
