package cmd

import (
	"context"
	"fmt"
	"log/slog"
	"os/signal"
	"syscall"
	"time"

	"github.com/spf13/cobra"

	gpuserve "github.com/ThatCatDev/tanrenai-gpu/pkg/serve"
	srvserve "github.com/ThatCatDev/tanrenai/server/pkg/serve"
)

var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start GPU and backend servers together",
	Long:  "Start both the GPU inference server and the backend server in a single process. Useful for running the servers on a remote machine while connecting a client from elsewhere.",
	RunE: func(cmd *cobra.Command, args []string) error {
		// Build GPU config.
		gpuCfg := gpuserve.Config{
			Host: "127.0.0.1",
		}
		gpuCfg.Port, _ = cmd.Flags().GetInt("gpu-port")
		gpuCfg.GPULayers, _ = cmd.Flags().GetInt("gpu-layers")
		gpuCfg.CtxSize, _ = cmd.Flags().GetInt("ctx-size")
		gpuCfg.FlashAttention, _ = cmd.Flags().GetBool("flash-attn")
		gpuCfg.ChatTemplate, _ = cmd.Flags().GetString("chat-template")
		gpuCfg.ChatTemplateFile, _ = cmd.Flags().GetString("chat-template-file")
		gpuCfg.EmbeddingModel, _ = cmd.Flags().GetString("embedding-model")
		gpuCfg.ReasoningFormat, _ = cmd.Flags().GetString("reasoning-format")
		gpuCfg.NoAutoTemplate, _ = cmd.Flags().GetBool("no-auto-template")
		gpuCfg.ModelsDir, _ = cmd.Flags().GetString("models-dir")

		// Build backend config.
		srvCfg := srvserve.Config{}
		srvCfg.Host, _ = cmd.Flags().GetString("host")
		srvCfg.Port, _ = cmd.Flags().GetInt("port")
		srvCfg.MemoryEnabled, _ = cmd.Flags().GetBool("memory")
		srvCfg.MemoryDir, _ = cmd.Flags().GetString("memory-dir")

		ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
		defer stop()

		// Start GPU server in background.
		gpuAddr := fmt.Sprintf("http://127.0.0.1:%d", gpuCfg.Port)
		gpuErrCh := make(chan error, 1)
		go func() {
			gpuErrCh <- gpuserve.Start(ctx, gpuCfg)
		}()

		// Wait for GPU server health.
		if err := waitForHealth(ctx, gpuAddr, 30*time.Second); err != nil {
			stop()

			return fmt.Errorf("GPU server failed to start: %w", err)
		}
		slog.Info("GPU server ready", "port", gpuCfg.Port)

		// Point backend at the GPU server.
		srvCfg.GPUURL = gpuAddr

		// Start backend server in background.
		backendAddr := fmt.Sprintf("http://127.0.0.1:%d", srvCfg.Port)
		srvErrCh := make(chan error, 1)
		go func() {
			srvErrCh <- srvserve.Start(ctx, srvCfg)
		}()

		// Wait for backend server health, but fail fast if the server exits with an error.
		healthCh := make(chan error, 1)
		go func() {
			healthCh <- waitForHealth(ctx, backendAddr, 30*time.Second)
		}()
		select {
		case err := <-healthCh:
			if err != nil {
				stop()
				<-srvErrCh
				<-gpuErrCh

				return fmt.Errorf("backend server failed to start: %w", err)
			}
		case err := <-srvErrCh:
			stop()
			<-gpuErrCh

			return fmt.Errorf("backend server failed: %w", err)
		}
		slog.Info("Backend server ready", "addr", backendAddr)

		// Block until signal or server exit.
		select {
		case err := <-srvErrCh:
			stop()
			<-gpuErrCh

			return err
		case err := <-gpuErrCh:
			stop()
			<-srvErrCh

			return err
		case <-ctx.Done():
			<-srvErrCh
			<-gpuErrCh

			return nil
		}
	},
}

func init() {
	// Backend flags.
	serveCmd.Flags().String("host", "0.0.0.0", "backend bind address")
	serveCmd.Flags().Int("port", 8080, "backend listen port")
	serveCmd.Flags().Bool("memory", false, "enable memory/RAG")
	serveCmd.Flags().String("memory-dir", "", "memory storage directory")

	// GPU flags.
	serveCmd.Flags().Int("gpu-port", 11435, "GPU server listen port")
	serveCmd.Flags().Int("gpu-layers", -1, "GPU layers to offload (-1 = auto)")
	serveCmd.Flags().Int("ctx-size", 4096, "context window size")
	serveCmd.Flags().Bool("flash-attn", true, "enable flash attention")
	serveCmd.Flags().String("chat-template", "", "named chat template (e.g. chatml)")
	serveCmd.Flags().String("chat-template-file", "", "path to custom Jinja chat template file")
	serveCmd.Flags().String("embedding-model", "", "embedding model name (e.g. nomic-embed-text)")
	serveCmd.Flags().String("reasoning-format", "", "reasoning format for thinking mode (e.g. deepseek)")
	serveCmd.Flags().Bool("no-auto-template", false, "disable automatic chat template detection from GGUF metadata")
	serveCmd.Flags().String("models-dir", "", "models directory")

	rootCmd.AddCommand(serveCmd)
}
