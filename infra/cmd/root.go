package cmd

import (
	"fmt"
	"log/slog"
	"os"

	"github.com/spf13/cobra"
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

func exitError(msg string, args ...any) {
	fmt.Fprintf(os.Stderr, "Error: "+msg+"\n", args...)
	os.Exit(1)
}
