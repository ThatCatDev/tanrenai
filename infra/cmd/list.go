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

var listCmd = &cobra.Command{
	Use:   "list",
	Short: "List your vast.ai instances",
	Run:   runList,
}

func init() {
	listCmd.Flags().String("vastai-api-key", "", "vast.ai API key (env: VASTAI_API_KEY)")
	rootCmd.AddCommand(listCmd)
}

func runList(cmd *cobra.Command, args []string) {
	cfg := config.Defaults()

	if v, _ := cmd.Flags().GetString("vastai-api-key"); v != "" {
		cfg.VastaiAPIKey = v
	}

	if cfg.VastaiAPIKey == "" {
		exitError("vast.ai API key required (--vastai-api-key or VASTAI_API_KEY)")
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	client := vastai.NewClient(cfg.VastaiAPIKey)
	instances, err := client.ListInstances(ctx)
	if err != nil {
		exitError("list instances: %v", err)
	}

	if len(instances) == 0 {
		_, _ = fmt.Fprintf(os.Stdout, "No instances found.\n")

		return
	}

	for _, inst := range instances {
		_, _ = fmt.Fprintf(os.Stdout, "  %d  %-10s  %-20s  x%d  $%.3f/hr",
			inst.ID, inst.Status, inst.GPUName, inst.NumGPUs, inst.CostPerHr)
		if inst.SSHHost != "" {
			_, _ = fmt.Fprintf(os.Stdout, "  ssh://%s:%d", inst.SSHHost, inst.SSHPort)
		}
		_, _ = fmt.Fprintf(os.Stdout, "\n")
	}
}
