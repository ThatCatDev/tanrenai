package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
	"github.com/ThatCatDev/tanrenai/server/internal/config"
	"github.com/ThatCatDev/tanrenai/server/internal/models"
)

var listCmd = &cobra.Command{
	Use:   "list",
	Short: "List available models",
	RunE: func(cmd *cobra.Command, args []string) error {
		dir, _ := cmd.Flags().GetString("models-dir")
		if dir == "" {
			dir = config.ModelsDir()
		}

		store := models.NewStore(dir)
		entries := store.List()

		if len(entries) == 0 {
			fmt.Printf("No models found in %s\n", dir)
			fmt.Println("Download a model with: tanrenai pull <model>")
			return nil
		}

		fmt.Printf("%-40s %10s\n", "NAME", "SIZE")
		fmt.Println("─────────────────────────────────────────────────────")
		for _, e := range entries {
			fmt.Printf("%-40s %10s\n", e.Name, formatSize(e.Size))
		}

		return nil
	},
}

func formatSize(bytes int64) string {
	const (
		MB = 1024 * 1024
		GB = 1024 * MB
	)
	switch {
	case bytes >= GB:
		return fmt.Sprintf("%.1f GB", float64(bytes)/float64(GB))
	case bytes >= MB:
		return fmt.Sprintf("%.1f MB", float64(bytes)/float64(MB))
	default:
		return fmt.Sprintf("%d B", bytes)
	}
}

func init() {
	rootCmd.AddCommand(listCmd)
}
