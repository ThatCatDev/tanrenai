package cmd

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"
)

var setupCmd = &cobra.Command{
	Use:   "setup",
	Short: "Configure tanrenai services",
}

var setupVastaiCmd = &cobra.Command{
	Use:   "vastai",
	Short: "Store your vast.ai API key on the platform",
	RunE: func(cmd *cobra.Command, args []string) error {
		apiKey, _ := cmd.Flags().GetString("api-key")

		if apiKey == "" {
			fmt.Print("Enter your vast.ai API key: ")
			reader := bufio.NewReader(os.Stdin)
			input, err := reader.ReadString('\n')
			if err != nil {
				return fmt.Errorf("read input: %w", err)
			}
			apiKey = strings.TrimSpace(input)
		}

		if apiKey == "" {
			return fmt.Errorf("API key cannot be empty")
		}

		// Load credentials for auth token and server URL
		creds, err := loadCredentials()
		if err != nil {
			return fmt.Errorf("not logged in — run 'tanrenai login' first")
		}

		body, _ := json.Marshal(map[string]string{"api_key": apiKey})
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, creds.ServerURL+"/api/user/vastai-key", bytes.NewReader(body))
		if err != nil {
			return err
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+creds.AccessToken)

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return fmt.Errorf("send request: %w", err)
		}
		defer func() { _ = resp.Body.Close() }()

		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("server returned %d", resp.StatusCode)
		}

		fmt.Println("Vast.ai API key saved successfully!")
		return nil
	},
}

func init() {
	setupVastaiCmd.Flags().String("api-key", "", "vast.ai API key (or enter interactively)")
	setupCmd.AddCommand(setupVastaiCmd)
	rootCmd.AddCommand(setupCmd)
}
