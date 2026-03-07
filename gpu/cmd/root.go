package cmd

import (
	"fmt"
	"log/slog"
	"os"

	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:   "tanrenai-gpu",
	Short: "Tanrenai GPU server — inference + training",
	Long:  "Tanrenai GPU server provides LLM inference (chat completions, embeddings) and fine-tuning endpoints.",
}

func Execute() error {
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stderr, nil)))
	return rootCmd.Execute()
}

func init() {
	rootCmd.PersistentFlags().StringP("models-dir", "m", "", "model storage directory")
}

func exitError(msg string, args ...any) {
	fmt.Fprintf(os.Stderr, "Error: "+msg+"\n", args...)
	os.Exit(1)
}
