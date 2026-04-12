package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/spf13/cobra"
)

var instanceCmd = &cobra.Command{
	Use:   "instance",
	Short: "Manage GPU instances",
}

var instanceStatusCmd = &cobra.Command{
	Use:   "status",
	Short: "Show current GPU instance status",
	RunE: func(cmd *cobra.Command, args []string) error {
		result, err := platformGet("/api/instance/status")
		if err != nil {
			return err
		}
		data, _ := json.MarshalIndent(result, "", "  ")
		fmt.Println(string(data))
		return nil
	},
}

var instanceCostCmd = &cobra.Command{
	Use:   "cost",
	Short: "Show current GPU instance cost",
	RunE: func(cmd *cobra.Command, args []string) error {
		result, err := platformGet("/api/instance/cost")
		if err != nil {
			return err
		}
		data, _ := json.MarshalIndent(result, "", "  ")
		fmt.Println(string(data))
		return nil
	},
}

var instanceDestroyCmd = &cobra.Command{
	Use:   "destroy",
	Short: "Destroy the current GPU instance",
	RunE: func(cmd *cobra.Command, args []string) error {
		result, err := platformPost("/api/instance/destroy", nil)
		if err != nil {
			return err
		}
		data, _ := json.MarshalIndent(result, "", "  ")
		fmt.Println(string(data))
		return nil
	},
}

func init() {
	instanceCmd.AddCommand(instanceStatusCmd)
	instanceCmd.AddCommand(instanceCostCmd)
	instanceCmd.AddCommand(instanceDestroyCmd)
	rootCmd.AddCommand(instanceCmd)
}

func platformGet(path string) (map[string]any, error) {
	creds, err := loadCredentials()
	if err != nil {
		return nil, fmt.Errorf("not logged in — run 'tanrenai login' first")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, creds.ServerURL+path, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+creds.AccessToken)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	var result map[string]any
	body, _ := io.ReadAll(resp.Body)
	_ = json.Unmarshal(body, &result)

	if resp.StatusCode != http.StatusOK {
		return result, fmt.Errorf("server returned %d: %s", resp.StatusCode, string(body))
	}
	return result, nil
}

func platformPost(path string, body io.Reader) (map[string]any, error) {
	creds, err := loadCredentials()
	if err != nil {
		return nil, fmt.Errorf("not logged in — run 'tanrenai login' first")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, creds.ServerURL+path, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+creds.AccessToken)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	var result map[string]any
	respBody, _ := io.ReadAll(resp.Body)
	_ = json.Unmarshal(respBody, &result)

	if resp.StatusCode != http.StatusOK {
		return result, fmt.Errorf("server returned %d: %s", resp.StatusCode, string(respBody))
	}
	return result, nil
}
