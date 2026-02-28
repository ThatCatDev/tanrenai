package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:   "tanrenai-server",
	Short: "Tanrenai backend server",
	Long:  "Tanrenai (鍛錬AI) backend — orchestration layer with memory, GPU proxy, and instance management.",
}

func Execute() error {
	return rootCmd.Execute()
}

func exitError(msg string, args ...any) {
	fmt.Fprintf(os.Stderr, "Error: "+msg+"\n", args...)
	os.Exit(1)
}
