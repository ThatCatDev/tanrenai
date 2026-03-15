package cmd

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"strings"

	"github.com/spf13/cobra"

	"github.com/ThatCatDev/tanrenai/infra/internal/config"
	"github.com/ThatCatDev/tanrenai/infra/internal/network"
)

var networkCmd = &cobra.Command{
	Use:   "network",
	Short: "Manage Headscale/Tailscale network",
}

var networkAuthKeyCmd = &cobra.Command{
	Use:   "auth-key",
	Short: "Generate a pre-auth key for joining the network",
	Run:   runNetworkAuthKey,
}

var networkNodesCmd = &cobra.Command{
	Use:   "nodes",
	Short: "List all nodes in the network",
	Run:   runNetworkNodes,
}

var networkJoinCmd = &cobra.Command{
	Use:   "join",
	Short: "Join the Headscale network on this machine",
	Long:  "Install Tailscale (if needed), start tailscaled, and join the Headscale network.",
	Run:   runNetworkJoin,
}

func init() {
	for _, c := range []*cobra.Command{networkAuthKeyCmd, networkNodesCmd, networkJoinCmd} {
		c.Flags().String("headscale-url", "", "Headscale server URL (env: HEADSCALE_URL)")
		c.Flags().String("headscale-api-key", "", "Headscale API key (env: HEADSCALE_API_KEY)")
		c.Flags().String("headscale-user", "", "Headscale user (env: HEADSCALE_USER, default: tanrenai)")
	}

	networkAuthKeyCmd.Flags().Bool("reusable", false, "make the key reusable")
	networkAuthKeyCmd.Flags().Bool("ephemeral", true, "make nodes ephemeral")

	networkJoinCmd.Flags().String("hostname", "", "hostname for this node (required)")
	networkJoinCmd.MarkFlagRequired("hostname")

	networkCmd.AddCommand(networkAuthKeyCmd)
	networkCmd.AddCommand(networkNodesCmd)
	networkCmd.AddCommand(networkJoinCmd)
	rootCmd.AddCommand(networkCmd)
}

func headscaleProvider(cmd *cobra.Command) *network.HeadscaleProvider {
	cfg := config.Defaults()
	if v, _ := cmd.Flags().GetString("headscale-url"); v != "" {
		cfg.HeadscaleURL = v
	}
	if v, _ := cmd.Flags().GetString("headscale-api-key"); v != "" {
		cfg.HeadscaleAPI = v
	}
	if v, _ := cmd.Flags().GetString("headscale-user"); v != "" {
		cfg.HeadscaleUser = v
	}
	if cfg.HeadscaleURL == "" {
		exitError("--headscale-url or HEADSCALE_URL required")
	}
	if cfg.HeadscaleAPI == "" {
		exitError("--headscale-api-key or HEADSCALE_API_KEY required")
	}
	return network.NewHeadscaleProvider(cfg.HeadscaleURL, cfg.HeadscaleAPI, cfg.HeadscaleUser)
}

func runNetworkAuthKey(cmd *cobra.Command, args []string) {
	provider := headscaleProvider(cmd)

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	key, err := provider.GenerateAuthKey(ctx)
	if err != nil {
		exitError("generate auth key: %v", err)
	}

	fmt.Println(key)
}

func runNetworkNodes(cmd *cobra.Command, args []string) {
	provider := headscaleProvider(cmd)

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	nodes, err := provider.ListNodes(ctx)
	if err != nil {
		exitError("list nodes: %v", err)
	}

	if len(nodes) == 0 {
		fmt.Println("No nodes found.")
		return
	}

	for _, n := range nodes {
		status := "offline"
		if n.Online {
			status = "online"
		}
		ips := strings.Join(n.IPs, ", ")
		fmt.Printf("  %-30s  %-8s  %s\n", n.Name, status, ips)
	}
}

func runNetworkJoin(cmd *cobra.Command, args []string) {
	provider := headscaleProvider(cmd)
	hostname, _ := cmd.Flags().GetString("hostname")

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	fmt.Println("Generating auth key...")
	authKey, err := provider.GenerateAuthKey(ctx)
	if err != nil {
		exitError("generate auth key: %v", err)
	}

	cmds := provider.InstallCommands(authKey, hostname)
	for _, c := range cmds {
		display := c
		if len(display) > 100 {
			display = display[:100] + "..."
		}
		fmt.Printf("$ %s\n", display)

		sh := exec.CommandContext(ctx, "bash", "-c", c)
		sh.Stdout = os.Stdout
		sh.Stderr = os.Stderr
		if err := sh.Run(); err != nil {
			exitError("command failed: %v", err)
		}
	}

	fmt.Println("Joined network successfully.")
}
