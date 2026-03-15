package cmd

import (
	"context"
	"fmt"
	"os"
	"os/signal"

	"github.com/spf13/cobra"

	"github.com/ThatCatDev/tanrenai/infra/internal/config"
	"github.com/ThatCatDev/tanrenai/infra/internal/deploy"
	"github.com/ThatCatDev/tanrenai/infra/internal/network"
	"github.com/ThatCatDev/tanrenai/infra/internal/vastai"
)

var deployCmd = &cobra.Command{
	Use:   "deploy",
	Short: "Deploy GPU server to a vast.ai instance",
	Long: `Deploy the tanrenai GPU server to a vast.ai instance, optionally setting up
a WireGuard tunnel (Headscale or Tailscale) for connectivity.

If --vastai-instance-id is provided, uses that instance.
Otherwise, searches for the cheapest matching offer and creates a new instance.`,
	Run: runDeploy,
}

func init() {
	f := deployCmd.Flags()
	f.String("vastai-api-key", "", "vast.ai API key (env: VASTAI_API_KEY)")
	f.String("vastai-instance-id", "", "existing vast.ai instance ID (env: VASTAI_INSTANCE_ID)")
	f.String("gpu", "", "filter offers by GPU name (e.g. A100, 4090, H100)")
	f.String("model-size", "", "auto-set min GPU RAM from model size (e.g. 8b, 27b, 72b, 120b)")
	f.Float64("min-gpu-ram", 24, "minimum GPU RAM in GB for offer search")
	f.Float64("max-cost", 1.0, "maximum $/hr for offer search")
	f.Float64("disk-gb", 50, "disk space in GB")
	f.String("network", "none", "network tunnel: headscale, tailscale, or none")
	f.String("headscale-url", "", "Headscale server URL (env: HEADSCALE_URL)")
	f.String("headscale-api-key", "", "Headscale API key (env: HEADSCALE_API_KEY)")
	f.String("headscale-user", "", "Headscale user name (env: HEADSCALE_USER, default: tanrenai)")
	f.String("tailscale-auth-key", "", "Tailscale auth key (env: TAILSCALE_AUTH_KEY)")
	f.Int("gpu-port", 11435, "port for the GPU server")
	f.String("model", "", "model to pull during setup")
	f.BoolP("verbose", "v", false, "show full SSH output during setup")

	rootCmd.AddCommand(deployCmd)
}

func runDeploy(cmd *cobra.Command, args []string) {
	cfg := config.Defaults()

	// Override from flags
	if v, _ := cmd.Flags().GetString("vastai-api-key"); v != "" {
		cfg.VastaiAPIKey = v
	}
	if v, _ := cmd.Flags().GetString("vastai-instance-id"); v != "" {
		cfg.VastaiInstance = v
	}
	if v, _ := cmd.Flags().GetString("gpu"); v != "" {
		cfg.GPUName = v
	}
	if v, _ := cmd.Flags().GetString("model-size"); v != "" {
		vram, err := config.VRAMForModelSize(v)
		if err != nil {
			exitError("%v", err)
		}
		cfg.MinGPURAM = vram
		// Auto-set disk if not explicitly provided
		if !cmd.Flags().Changed("disk-gb") {
			disk, _ := config.DiskForModelSize(v)
			cfg.DiskGB = disk
		}
		fmt.Printf("Model size %s → need %.0f GB VRAM, %.0f GB disk\n", v, vram, cfg.DiskGB)
	}
	if v, _ := cmd.Flags().GetFloat64("min-gpu-ram"); cmd.Flags().Changed("min-gpu-ram") {
		cfg.MinGPURAM = v
	}
	if v, _ := cmd.Flags().GetFloat64("max-cost"); cmd.Flags().Changed("max-cost") {
		cfg.MaxCostPerHr = v
	}
	if v, _ := cmd.Flags().GetFloat64("disk-gb"); cmd.Flags().Changed("disk-gb") {
		cfg.DiskGB = v
	}
	if v, _ := cmd.Flags().GetString("network"); v != "" {
		cfg.Network = v
	}
	if v, _ := cmd.Flags().GetString("headscale-url"); v != "" {
		cfg.HeadscaleURL = v
	}
	if v, _ := cmd.Flags().GetString("headscale-api-key"); v != "" {
		cfg.HeadscaleAPI = v
	}
	if v, _ := cmd.Flags().GetString("headscale-user"); v != "" {
		cfg.HeadscaleUser = v
	}
	if v, _ := cmd.Flags().GetString("tailscale-auth-key"); v != "" {
		cfg.TailscaleAuth = v
	}
	if v, _ := cmd.Flags().GetInt("gpu-port"); cmd.Flags().Changed("gpu-port") {
		cfg.GPUPort = v
	}
	cfg.Model, _ = cmd.Flags().GetString("model")

	if cfg.VastaiAPIKey == "" {
		exitError("vast.ai API key required (--vastai-api-key or VASTAI_API_KEY)")
	}

	// Build network provider
	var netProvider network.Provider
	switch cfg.Network {
	case "headscale":
		if cfg.HeadscaleURL == "" {
			exitError("--headscale-url required when using headscale network")
		}
		if cfg.HeadscaleAPI == "" {
			exitError("--headscale-api-key required when using headscale network")
		}
		netProvider = network.NewHeadscaleProvider(cfg.HeadscaleURL, cfg.HeadscaleAPI, cfg.HeadscaleUser)
	case "tailscale":
		netProvider = network.NewTailscaleProvider(cfg.TailscaleAuth)
	case "none":
		netProvider = network.NewNoneProvider()
	default:
		exitError("unknown network provider: %s (use headscale, tailscale, or none)", cfg.Network)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	verbose, _ := cmd.Flags().GetBool("verbose")

	client := vastai.NewClient(cfg.VastaiAPIKey)
	deployer := deploy.New(client, netProvider, cfg, os.Stdout, verbose)

	result, err := deployer.Run(ctx)
	if err != nil {
		exitError("deploy failed: %v", err)
	}

	fmt.Println()
	fmt.Println("=== Deploy Complete ===")
	fmt.Printf("Instance:  %d\n", result.InstanceID)
	fmt.Printf("GPU:       %s\n", result.GPUName)
	fmt.Printf("Cost:      $%.3f/hr\n", result.CostPerHr)
	fmt.Printf("GPU URL:   %s\n", result.GPUURL)
	fmt.Println()
	fmt.Printf("Start backend:\n  tanrenai-server serve --gpu-url %s --memory\n", result.GPUURL)
}
