package cmd

import (
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
	rootCmd.PersistentFlags().Bool("local", false, "start embedded GPU + backend servers (single-binary mode)")
	rootCmd.PersistentFlags().Int("gpu-layers", -1, "GPU offload layers (-1 = all); only used with --local")
	rootCmd.PersistentFlags().Bool("flash-attn", true, "enable flash attention; only used with --local")
	rootCmd.PersistentFlags().Bool("cpu-moe", false, "keep MoE expert weights on CPU; only used with --local")
	rootCmd.PersistentFlags().Bool("no-kv-offload", false, "keep KV cache on CPU to save VRAM; only used with --local")
	rootCmd.PersistentFlags().Bool("fit", false, "auto-adjust to fit device memory; only used with --local")
}
