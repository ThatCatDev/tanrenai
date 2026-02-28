package cmd

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

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

		// Read SSE progress events
		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := line[len("data: "):]

			var evt map[string]any
			if err := json.Unmarshal([]byte(data), &evt); err != nil {
				continue
			}

			switch evt["status"] {
			case "downloading":
				percent := int(evt["percent"].(float64))
				downloaded := int64(evt["downloaded"].(float64))
				total := int64(evt["total"].(float64))
				printProgress(percent, downloaded, total)
			case "downloaded":
				fmt.Printf("\rDownloaded: %s\n", evt["path"])
			case "error":
				return fmt.Errorf("download failed: %s", evt["error"])
			}
		}

		return scanner.Err()
	},
}

func printProgress(percent int, downloaded, total int64) {
	const barWidth = 30
	filled := barWidth * percent / 100
	bar := strings.Repeat("█", filled) + strings.Repeat("░", barWidth-filled)
	fmt.Printf("\r[%s] %3d%%  %s / %s", bar, percent, formatBytes(downloaded), formatBytes(total))
}

func formatBytes(b int64) string {
	const (
		MB = 1024 * 1024
		GB = 1024 * MB
	)
	switch {
	case b >= GB:
		return fmt.Sprintf("%.1f GB", float64(b)/float64(GB))
	case b >= MB:
		return fmt.Sprintf("%.1f MB", float64(b)/float64(MB))
	default:
		return fmt.Sprintf("%d B", b)
	}
}

func init() {
	rootCmd.AddCommand(pullCmd)
}
