package cmd

import (
	"fmt"
	"log/slog"
	"os"

	"github.com/spf13/cobra"

	"github.com/ThatCatDev/tanrenai/infra/internal/vastai"
)

var rootCmd = &cobra.Command{
	Use:   "tanrenai-infra",
	Short: "Tanrenai infrastructure management",
	Long:  "Tanrenai (鍛錬AI) infra — deploy GPU servers to vast.ai with optional WireGuard tunnel.",
}

func Execute() error {
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stderr, nil)))

	return rootCmd.Execute()
}

// osExit is a variable so tests can replace it to avoid os.Exit.
var osExit = os.Exit

func exitError(msg string, args ...any) {
	fmt.Fprintf(os.Stderr, "Error: "+msg+"\n", args...)
	osExit(1)
}

// newVastaiClient is a variable so tests can inject a client backed by a test server.
var newVastaiClient = func(apiKey string) *vastai.Client {
	return vastai.NewClient(apiKey)
}
