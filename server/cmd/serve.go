package cmd

import (
	"context"
	"log"
	"os/signal"
	"syscall"
	"time"

	"github.com/spf13/cobra"

	"github.com/ThatCatDev/tanrenai/server/internal/config"
	"github.com/ThatCatDev/tanrenai/server/internal/gpuclient"
	"github.com/ThatCatDev/tanrenai/server/internal/gpuprovider"
	"github.com/ThatCatDev/tanrenai/server/internal/memory"
	"github.com/ThatCatDev/tanrenai/server/internal/server"
	"github.com/ThatCatDev/tanrenai/server/internal/vastai"
)

var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start the tanrenai backend server",
	RunE: func(cmd *cobra.Command, args []string) error {
		cfg := config.DefaultConfig()

		if host, _ := cmd.Flags().GetString("host"); host != "" {
			cfg.Host = host
		}
		if port, _ := cmd.Flags().GetInt("port"); port != 0 {
			cfg.Port = port
		}
		if gpuURL, _ := cmd.Flags().GetString("gpu-url"); gpuURL != "" {
			cfg.GPUURL = gpuURL
		}
		if memEnabled, _ := cmd.Flags().GetBool("memory"); memEnabled {
			cfg.MemoryEnabled = true
		}
		if memDir, _ := cmd.Flags().GetString("memory-dir"); memDir != "" {
			cfg.MemoryDir = memDir
		}
		if apiKey, _ := cmd.Flags().GetString("vastai-api-key"); apiKey != "" {
			cfg.VastaiAPIKey = apiKey
		}
		if instanceID, _ := cmd.Flags().GetString("vastai-instance-id"); instanceID != "" {
			cfg.VastaiInstance = instanceID
		}
		if timeout, _ := cmd.Flags().GetString("idle-timeout"); timeout != "" {
			cfg.IdleTimeout = timeout
		}

		if err := config.EnsureDirs(cfg); err != nil {
			return err
		}

		// Create GPU client
		gpu := gpuclient.New(cfg.GPUURL)

		// Create memory store if enabled
		var memStore memory.Store
		if cfg.MemoryEnabled {
			embedFunc := memory.NewRemoteEmbedFunc(gpu)
			store, err := memory.NewChromemStore(cfg.MemoryDir, embedFunc)
			if err != nil {
				return err
			}
			memStore = store
			log.Printf("Memory store initialized at %s", cfg.MemoryDir)
		}

		// Create GPU provider
		var provider gpuprovider.Provider
		if cfg.VastaiAPIKey != "" && cfg.VastaiInstance != "" {
			idleTimeout, err := time.ParseDuration(cfg.IdleTimeout)
			if err != nil {
				idleTimeout = 20 * time.Minute
			}
			vastClient := vastai.NewClient(cfg.VastaiAPIKey)
			provider = gpuprovider.NewVastAIProvider(vastClient, gpu, cfg.VastaiInstance, cfg.GPUURL, idleTimeout)
			log.Printf("GPU provider: vastai (instance: %s, idle timeout: %v)", cfg.VastaiInstance, idleTimeout)
		} else {
			provider = gpuprovider.NewLocalProvider(gpu)
			log.Printf("GPU provider: local (%s)", cfg.GPUURL)
		}

		ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
		defer stop()

		srv := server.New(cfg, gpu, memStore, provider)
		return srv.Start(ctx)
	},
}

func init() {
	serveCmd.Flags().String("host", "0.0.0.0", "bind address")
	serveCmd.Flags().Int("port", 8080, "listen port")
	serveCmd.Flags().String("gpu-url", "http://localhost:11435", "GPU server URL")
	serveCmd.Flags().Bool("memory", false, "enable memory/RAG")
	serveCmd.Flags().String("memory-dir", "", "memory storage directory")
	serveCmd.Flags().String("vastai-api-key", "", "vast.ai API key")
	serveCmd.Flags().String("vastai-instance-id", "", "vast.ai instance ID to manage")
	serveCmd.Flags().String("idle-timeout", "20m", "auto-stop after inactivity")
	rootCmd.AddCommand(serveCmd)
}
