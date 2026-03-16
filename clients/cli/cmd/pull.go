package cmd

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"github.com/ThatCatDev/tanrenai/shared/apiclient"
)

var pullCmd = &cobra.Command{
	Use:   "pull <model>",
	Short: "Download a GGUF model (supports hf://owner/repo/quant or direct URL)",
	Long: `Download a GGUF model via the backend.

Supports:
  hf://unsloth/Qwen3.5-27B-GGUF                  auto-pick best quant
  hf://unsloth/Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL  specific quant (incl. split files)
  https://huggingface.co/.../model.gguf             direct URL`,
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		modelURL := args[0]

		activeURL := serverURL
		local, _ := cmd.Flags().GetBool("local")
		if local {
			gpuLayers, _ := cmd.Flags().GetInt("gpu-layers")
			flashAttn, _ := cmd.Flags().GetBool("flash-attn")
			url, cleanup, err := startLocalServers(cmd.Context(), localOpts{
				GPULayers:      gpuLayers,
				FlashAttention: flashAttn,
			}, &startupLog{})
			if err != nil {
				return err
			}
			defer cleanup()
			activeURL = url
		}

		client := apiclient.New(activeURL)

		ch, err := client.PullModel(context.Background(), modelURL)
		if err != nil {
			return fmt.Errorf("failed to pull model: %w", err)
		}

		for ev := range ch {
			if ev.Err != nil {
				return fmt.Errorf("download failed: %w", ev.Err)
			}
			switch ev.Event.Status {
			case "resolving":
				if ev.Event.TotalFiles > 1 {
					_, _ = fmt.Fprintf(os.Stdout, "Downloading %d files...\n", ev.Event.TotalFiles)
				}
			case "downloading":
				prefix := ""
				if ev.Event.TotalFiles > 1 {
					prefix = fmt.Sprintf("[%d/%d] ", ev.Event.File, ev.Event.TotalFiles)
				}
				printProgress(prefix, ev.Event.Percent, ev.Event.Downloaded, ev.Event.Total)
			case "downloaded":
				_, _ = fmt.Fprintf(os.Stdout, "\rDownloaded: %s\n", ev.Event.Path)
			case "error":
				return fmt.Errorf("download failed: %s", ev.Event.Path)
			}
		}

		return nil
	},
}

func printProgress(prefix string, percent int, downloaded, total int64) {
	const barWidth = 30
	filled := barWidth * percent / 100
	bar := strings.Repeat("█", filled) + strings.Repeat("░", barWidth-filled)
	_, _ = fmt.Fprintf(os.Stdout, "\r%s[%s] %3d%%  %s / %s", prefix, bar, percent, formatBytes(downloaded), formatBytes(total))
}

func formatBytes(b int64) string {
	const (
		MB = 1024 * 1024
		GB = 1024 * MB
	)
	switch {
	case b >= GB:
		return fmt.Sprintf("%.1f GB", float64(b)/float64(GB))
	case b >= MB:
		return fmt.Sprintf("%.1f MB", float64(b)/float64(MB))
	default:
		return fmt.Sprintf("%d B", b)
	}
}

func init() {
	rootCmd.AddCommand(pullCmd)
}
