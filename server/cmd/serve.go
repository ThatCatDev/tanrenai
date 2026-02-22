package cmd

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/spf13/cobra"
	"github.com/ThatCatDev/tanrenai/server/internal/config"
	"github.com/ThatCatDev/tanrenai/server/internal/runner"
	"github.com/ThatCatDev/tanrenai/server/internal/server"
)

var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start the tanrenai API server",
	RunE: func(cmd *cobra.Command, args []string) error {
		cfg := config.DefaultConfig()

		if host, _ := cmd.Flags().GetString("host"); host != "" {
			cfg.Host = host
		}
		if port, _ := cmd.Flags().GetInt("port"); port != 0 {
			cfg.Port = port
		}
		if dir, _ := cmd.Flags().GetString("models-dir"); dir != "" {
			cfg.ModelsDir = dir
		}
		if gpu, _ := cmd.Flags().GetInt("gpu-layers"); cmd.Flags().Changed("gpu-layers") {
			cfg.GPULayers = gpu
		}
		if ctx, _ := cmd.Flags().GetInt("ctx-size"); ctx != 0 {
			cfg.CtxSize = ctx
		}
		if tpl, _ := cmd.Flags().GetString("chat-template-file"); tpl != "" {
			cfg.ChatTemplateFile = tpl
		}
		// Named template shortcut: --chat-template qwen2.5 writes the built-in
		// template to a temp file and passes it automatically.
		if name, _ := cmd.Flags().GetString("chat-template"); name != "" && cfg.ChatTemplateFile == "" {
			switch name {
			case "qwen2.5", "qwen2", "qwen":
				path, err := runner.WriteQwen25Template()
				if err != nil {
					return fmt.Errorf("failed to write chat template: %w", err)
				}
				defer os.Remove(path)
				cfg.ChatTemplateFile = path
				fmt.Printf("Using Qwen2.5 native chat template\n")
			default:
				return fmt.Errorf("unknown chat template %q (available: qwen2.5)", name)
			}
		}

		if err := config.EnsureDirs(); err != nil {
			return err
		}

		ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
		defer stop()

		srv := server.New(cfg)
		return srv.Start(ctx)
	},
}

func init() {
	serveCmd.Flags().String("host", "127.0.0.1", "bind address")
	serveCmd.Flags().Int("port", 11435, "listen port")
	serveCmd.Flags().Int("gpu-layers", -1, "GPU layers to offload (-1 = auto)")
	serveCmd.Flags().Int("ctx-size", 4096, "context window size")
	serveCmd.Flags().String("chat-template", "", "named chat template to use (e.g. qwen2.5)")
	serveCmd.Flags().String("chat-template-file", "", "path to custom Jinja chat template file")
	rootCmd.AddCommand(serveCmd)
}
