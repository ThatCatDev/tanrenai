package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:   "tanrenai",
	Short: "Tanrenai — local LLM server",
	Long:  "Tanrenai (鍛錬AI) — a self-evolving local LLM system that serves, manages, and improves models over time.",
}

func Execute() error {
	return rootCmd.Execute()
}

func init() {
	rootCmd.PersistentFlags().StringP("models-dir", "m", "", "model storage directory")
}

func exitError(msg string, args ...any) {
	fmt.Fprintf(os.Stderr, "Error: "+msg+"\n", args...)
	os.Exit(1)
}
