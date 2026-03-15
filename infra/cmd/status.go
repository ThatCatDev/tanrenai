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

var statusCmd = &cobra.Command{
	Use:   "status",
	Short: "Show vast.ai instance status",
	Run:   runStatus,
}

func init() {
	f := statusCmd.Flags()
	f.String("vastai-api-key", "", "vast.ai API key (env: VASTAI_API_KEY)")
	f.String("vastai-instance-id", "", "vast.ai instance ID (env: VASTAI_INSTANCE_ID)")

	rootCmd.AddCommand(statusCmd)
}

func runStatus(cmd *cobra.Command, args []string) {
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

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	client := vastai.NewClient(cfg.VastaiAPIKey)

	if cfg.VastaiInstance != "" {
		inst, err := client.GetInstance(ctx, cfg.VastaiInstance)
		if err != nil {
			exitError("get instance: %v", err)
		}
		printInstance(inst)
		return
	}

	// List all instances
	instances, err := client.ListInstances(ctx)
	if err != nil {
		exitError("list instances: %v", err)
	}

	if len(instances) == 0 {
		fmt.Println("No instances found.")
		return
	}

	for i, inst := range instances {
		if i > 0 {
			fmt.Println("---")
		}
		printInstance(&inst)
	}
}

func printInstance(inst *vastai.Instance) {
	fmt.Printf("ID:      %d\n", inst.ID)
	fmt.Printf("Status:  %s\n", inst.Status)
	fmt.Printf("GPU:     %s (x%d)\n", inst.GPUName, inst.NumGPUs)
	fmt.Printf("Cost:    $%.3f/hr\n", inst.CostPerHr)
	if inst.SSHHost != "" {
		fmt.Printf("SSH:     %s:%d\n", inst.SSHHost, inst.SSHPort)
	}
}
