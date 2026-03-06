package cmd

import (
	"fmt"

	"github.com/ThatCatDev/tanrenai/shared/apiclient"
	"github.com/spf13/cobra"
)

var listCmd = &cobra.Command{
	Use:   "list",
	Short: "List available models",
	RunE: func(cmd *cobra.Command, args []string) error {
		client := apiclient.New(serverURL)
		resp, err := client.ListModels(cmd.Context())
		if err != nil {
			return fmt.Errorf("failed to list models: %w", err)
		}

		if len(resp.Data) == 0 {
			fmt.Println("No models available.")
			return nil
		}

		fmt.Printf("%-40s %10s\n", "NAME", "OWNER")
		fmt.Println("─────────────────────────────────────────────────────")
		for _, m := range resp.Data {
			fmt.Printf("%-40s %10s\n", m.ID, m.OwnedBy)
		}

		return nil
	},
}

func init() {
	rootCmd.AddCommand(listCmd)
}
