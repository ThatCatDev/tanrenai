package cmd

import (
	"context"
	"os/signal"
	"syscall"

	"github.com/spf13/cobra"

	gpuserve "github.com/ThatCatDev/tanrenai/gpu/pkg/serve"
)

var gpuCmd = &cobra.Command{
	Use:   "gpu",
	Short: "GPU inference server commands",
}

var gpuServeCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start the GPU inference server",
	RunE: func(cmd *cobra.Command, args []string) error {
		cfg := gpuserve.Config{}

		cfg.Host, _ = cmd.Flags().GetString("host")
		cfg.Port, _ = cmd.Flags().GetInt("port")
		cfg.GPULayers, _ = cmd.Flags().GetInt("gpu-layers")
		cfg.CtxSize, _ = cmd.Flags().GetInt("ctx-size")
		cfg.ChatTemplate, _ = cmd.Flags().GetString("chat-template")
		cfg.ChatTemplateFile, _ = cmd.Flags().GetString("chat-template-file")
		cfg.EmbeddingModel, _ = cmd.Flags().GetString("embedding-model")
		cfg.ReasoningFormat, _ = cmd.Flags().GetString("reasoning-format")
		cfg.FlashAttention, _ = cmd.Flags().GetBool("flash-attn")
		cfg.NoAutoTemplate, _ = cmd.Flags().GetBool("no-auto-template")
		cfg.ModelsDir, _ = cmd.Flags().GetString("models-dir")
		cfg.BinDir, _ = cmd.Flags().GetString("bin-dir")

		ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
		defer stop()

		return gpuserve.Start(ctx, cfg)
	},
}

func init() {
	gpuServeCmd.Flags().String("host", "127.0.0.1", "bind address")
	gpuServeCmd.Flags().Int("port", 11435, "listen port")
	gpuServeCmd.Flags().Int("gpu-layers", -1, "GPU layers to offload (-1 = auto)")
	gpuServeCmd.Flags().Int("ctx-size", 4096, "context window size")
	gpuServeCmd.Flags().String("chat-template", "", "named chat template (e.g. chatml)")
	gpuServeCmd.Flags().String("chat-template-file", "", "path to custom Jinja chat template file")
	gpuServeCmd.Flags().String("embedding-model", "", "embedding model name (e.g. nomic-embed-text)")
	gpuServeCmd.Flags().String("reasoning-format", "", "reasoning format for thinking mode (e.g. deepseek)")
	gpuServeCmd.Flags().Bool("flash-attn", true, "enable flash attention")
	gpuServeCmd.Flags().Bool("no-auto-template", false, "disable automatic chat template detection from GGUF metadata")
	gpuServeCmd.Flags().String("models-dir", "", "models directory (default: ~/.local/share/tanrenai/models)")
	gpuServeCmd.Flags().String("bin-dir", "", "binary directory for llama-server (default: ~/.local/share/tanrenai/bin)")

	gpuCmd.AddCommand(gpuServeCmd)
	rootCmd.AddCommand(gpuCmd)
}
