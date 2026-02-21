package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"github.com/thatcatdev/tanrenai/server/internal/config"
	"github.com/thatcatdev/tanrenai/server/internal/models"
)

var pullCmd = &cobra.Command{
	Use:   "pull <url>",
	Short: "Download a GGUF model from a URL",
	Long: `Download a GGUF model file from a direct URL.

Examples:
  tanrenai pull https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf
  tanrenai pull https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf

Set HF_TOKEN environment variable for gated models.`,
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		url := args[0]

		dir, _ := cmd.Flags().GetString("models-dir")
		if dir == "" {
			dir = config.ModelsDir()
		}

		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Errorf("create models dir: %w", err)
		}

		fmt.Printf("Downloading to %s...\n", dir)

		path, err := models.Download(url, dir, func(downloaded, total int64) {
			if total > 0 {
				pct := float64(downloaded) / float64(total) * 100
				fmt.Printf("\r  %.1f%% (%s / %s)", pct, formatSize(downloaded), formatSize(total))
			} else {
				fmt.Printf("\r  %s downloaded", formatSize(downloaded))
			}
		})
		if err != nil {
			return err
		}

		fmt.Printf("\n\nSaved to %s\n", path)
		return nil
	},
}

func init() {
	rootCmd.AddCommand(pullCmd)
}
