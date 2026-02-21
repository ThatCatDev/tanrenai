package cmd

import (
	"context"
	"os/signal"
	"syscall"

	"github.com/spf13/cobra"
	"github.com/thatcatdev/tanrenai/server/internal/config"
	"github.com/thatcatdev/tanrenai/server/internal/server"
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
	rootCmd.AddCommand(serveCmd)
}
