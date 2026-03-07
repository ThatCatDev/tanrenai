package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

var serverURL string

var rootCmd = &cobra.Command{
	Use:   "tanrenai",
	Short: "Tanrenai — AI assistant client",
	Long:  "Tanrenai (鍛錬AI) client — connects to the tanrenai backend for LLM inference, memory, and tool use.",
	PersistentPreRun: func(cmd *cobra.Command, args []string) {
		// Environment variable fallback: flag wins if explicitly set
		if !cmd.Flags().Changed("server-url") {
			if v := os.Getenv("TANRENAI_SERVER_URL"); v != "" {
				serverURL = v
			}
		}
	},
}

func Execute() error {
	return rootCmd.Execute()
}

func init() {
	rootCmd.PersistentFlags().StringVar(&serverURL, "server-url", "http://127.0.0.1:8080", "backend server URL")
}

func exitError(msg string, args ...any) {
	fmt.Fprintf(os.Stderr, "Error: "+msg+"\n", args...)
	os.Exit(1)
}
