package cmd

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/spf13/cobra"
)

var pullCmd = &cobra.Command{
	Use:   "pull <url>",
	Short: "Download a GGUF model via the backend",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		url := args[0]

		body, _ := json.Marshal(map[string]string{"url": url})
		resp, err := http.Post(serverURL+"/api/pull", "application/json", bytes.NewReader(body))
		if err != nil {
			return fmt.Errorf("failed to pull model: %w", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			respBody, _ := io.ReadAll(resp.Body)
			return fmt.Errorf("server returned %d: %s", resp.StatusCode, string(respBody))
		}

		var result map[string]string
		json.NewDecoder(resp.Body).Decode(&result)
		fmt.Printf("Model downloaded: %s\n", result["path"])
		return nil
	},
}

func init() {
	rootCmd.AddCommand(pullCmd)
}
