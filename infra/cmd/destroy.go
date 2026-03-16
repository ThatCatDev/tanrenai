package cmd

import (
	"context"
	"fmt"
	"os"
	"os/signal"

	"github.com/spf13/cobra"

	"github.com/ThatCatDev/tanrenai/infra/internal/config"
	"github.com/ThatCatDev/tanrenai/infra/internal/vastai"
)

var destroyCmd = &cobra.Command{
	Use:   "destroy",
	Short: "Destroy a vast.ai instance",
	Run:   runDestroy,
}

func init() {
	f := destroyCmd.Flags()
	f.String("vastai-api-key", "", "vast.ai API key (env: VASTAI_API_KEY)")
	f.String("vastai-instance-id", "", "vast.ai instance ID to destroy (env: VASTAI_INSTANCE_ID)")

	rootCmd.AddCommand(destroyCmd)
}

func runDestroy(cmd *cobra.Command, args []string) {
	cfg := config.Defaults()

	if v, _ := cmd.Flags().GetString("vastai-api-key"); v != "" {
		cfg.VastaiAPIKey = v
	}
	if v, _ := cmd.Flags().GetString("vastai-instance-id"); v != "" {
		cfg.VastaiInstance = v
	}

	if cfg.VastaiAPIKey == "" {
		exitError("vast.ai API key required (--vastai-api-key or VASTAI_API_KEY)")
	}
	if cfg.VastaiInstance == "" {
		exitError("--vastai-instance-id is required for destroy")
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	client := vastai.NewClient(cfg.VastaiAPIKey)

	_, _ = fmt.Fprintf(os.Stdout, "Destroying instance %s...\n", cfg.VastaiInstance)
	if err := client.DestroyInstance(ctx, cfg.VastaiInstance); err != nil {
		exitError("destroy instance: %v", err)
	}

	_, _ = fmt.Fprintf(os.Stdout, "Instance destroyed.\n")
}
