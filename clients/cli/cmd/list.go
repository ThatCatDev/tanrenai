package cmd

import (
	"fmt"
	"os"

	"github.com/ThatCatDev/tanrenai/shared/apiclient"
	"github.com/spf13/cobra"
)

var listCmd = &cobra.Command{
	Use:   "list",
	Short: "List available models",
	RunE: func(cmd *cobra.Command, args []string) error {
		client := apiclient.New(serverURL)
		if authToken != "" {
			client.SetAuthToken(authToken)
		}
		resp, err := client.ListModels(cmd.Context())
		if err != nil {
			return fmt.Errorf("failed to list models: %w", err)
		}

		if len(resp.Data) == 0 {
			_, _ = fmt.Fprintf(os.Stdout, "No models available.\n")

			return nil
		}

		_, _ = fmt.Fprintf(os.Stdout, "%-40s %10s\n", "NAME", "OWNER")
		_, _ = fmt.Fprintf(os.Stdout, "─────────────────────────────────────────────────────\n")
		for _, m := range resp.Data {
			_, _ = fmt.Fprintf(os.Stdout, "%-40s %10s\n", m.ID, m.OwnedBy)
		}

		return nil
	},
}

func init() {
	rootCmd.AddCommand(listCmd)
}
