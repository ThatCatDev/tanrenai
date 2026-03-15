package cmd

import (
	"context"
	"os"
	"os/signal"
	"strconv"
	"syscall"

	"github.com/spf13/cobra"

	srvserve "github.com/ThatCatDev/tanrenai/server/pkg/serve"
)

var srvCmd = &cobra.Command{
	Use:   "server",
	Short: "Backend server commands",
}

var srvServeCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start the backend server",
	RunE: func(cmd *cobra.Command, args []string) error {
		cfg := srvserve.Config{}

		cfg.Host, _ = cmd.Flags().GetString("host")
		cfg.Port, _ = cmd.Flags().GetInt("port")
		cfg.GPUURL, _ = cmd.Flags().GetString("gpu-url")
		cfg.MemoryEnabled, _ = cmd.Flags().GetBool("memory")
		cfg.MemoryDir, _ = cmd.Flags().GetString("memory-dir")
		cfg.VastaiAPIKey, _ = cmd.Flags().GetString("vastai-api-key")
		cfg.VastaiInstance, _ = cmd.Flags().GetString("vastai-instance-id")
		cfg.IdleTimeout, _ = cmd.Flags().GetString("idle-timeout")

		// Environment variable fallbacks (flag wins if explicitly set).
		if !cmd.Flags().Changed("gpu-url") {
			if v := os.Getenv("TANRENAI_GPU_URL"); v != "" {
				cfg.GPUURL = v
			}
		}
		if !cmd.Flags().Changed("port") {
			if v := os.Getenv("TANRENAI_SERVER_PORT"); v != "" {
				if p, err := strconv.Atoi(v); err == nil {
					cfg.Port = p
				}
			}
		}
		if !cmd.Flags().Changed("vastai-api-key") {
			if v := os.Getenv("TANRENAI_VASTAI_API_KEY"); v != "" {
				cfg.VastaiAPIKey = v
			}
		}

		ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
		defer stop()

		return srvserve.Start(ctx, cfg)
	},
}

func init() {
	srvServeCmd.Flags().String("host", "0.0.0.0", "bind address")
	srvServeCmd.Flags().Int("port", 8080, "listen port")
	srvServeCmd.Flags().String("gpu-url", "http://localhost:11435", "GPU server URL")
	srvServeCmd.Flags().Bool("memory", false, "enable memory/RAG")
	srvServeCmd.Flags().String("memory-dir", "", "memory storage directory")
	srvServeCmd.Flags().String("vastai-api-key", "", "vast.ai API key")
	srvServeCmd.Flags().String("vastai-instance-id", "", "vast.ai instance ID to manage")
	srvServeCmd.Flags().String("idle-timeout", "20m", "auto-stop after inactivity")

	srvCmd.AddCommand(srvServeCmd)
	rootCmd.AddCommand(srvCmd)
}
