package cmd

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/spf13/cobra"
	"github.com/thatcatdev/tanrenai/server/pkg/api"
)

var runCmd = &cobra.Command{
	Use:   "run <model>",
	Short: "Load a model and start an interactive chat",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		model := args[0]
		port, _ := cmd.Flags().GetInt("port")
		baseURL := fmt.Sprintf("http://127.0.0.1:%d", port)
		systemPrompt, _ := cmd.Flags().GetString("system")

		// Load the model
		fmt.Printf("Loading model %s...\n", model)
		if err := loadModelRemote(baseURL, model); err != nil {
			return fmt.Errorf("failed to load model (is 'tanrenai serve' running?): %w", err)
		}
		fmt.Printf("Model %s loaded. Type your message (Ctrl+C to quit).\n\n", model)

		return chatLoop(baseURL, model, systemPrompt)
	},
}

var chatCmd = &cobra.Command{
	Use:   "chat",
	Short: "Interactive chat with a loaded model",
	RunE: func(cmd *cobra.Command, args []string) error {
		model, _ := cmd.Flags().GetString("model")
		port, _ := cmd.Flags().GetInt("port")
		baseURL := fmt.Sprintf("http://127.0.0.1:%d", port)
		systemPrompt, _ := cmd.Flags().GetString("system")

		if model == "" {
			return fmt.Errorf("specify a model with --model")
		}

		fmt.Printf("Chatting with %s. Type your message (Ctrl+C to quit).\n\n", model)
		return chatLoop(baseURL, model, systemPrompt)
	},
}

func chatLoop(baseURL, model, systemPrompt string) error {
	scanner := bufio.NewScanner(os.Stdin)
	var history []api.Message

	if systemPrompt != "" {
		history = append(history, api.Message{Role: "system", Content: systemPrompt})
	}

	for {
		fmt.Print(">>> ")
		if !scanner.Scan() {
			fmt.Println()
			return nil
		}

		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}
		if input == "/quit" || input == "/exit" {
			return nil
		}

		history = append(history, api.Message{Role: "user", Content: input})

		req := api.ChatCompletionRequest{
			Model:    model,
			Messages: history,
			Stream:   true,
		}

		body, _ := json.Marshal(req)
		resp, err := http.Post(baseURL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			continue
		}

		assistant, err := readStream(resp.Body)
		resp.Body.Close()

		if err != nil {
			fmt.Fprintf(os.Stderr, "\nStream error: %v\n", err)
			continue
		}

		fmt.Println()
		fmt.Println()

		if assistant != "" {
			history = append(history, api.Message{Role: "assistant", Content: assistant})
		}
	}
}

func readStream(body io.Reader) (string, error) {
	scanner := bufio.NewScanner(body)
	var full strings.Builder

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var chunk api.ChatCompletionChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		for _, choice := range chunk.Choices {
			content := choice.Delta.Content
			if content != "" {
				fmt.Print(content)
				full.WriteString(content)
			}
		}
	}

	return full.String(), scanner.Err()
}

func loadModelRemote(baseURL, model string) error {
	body, _ := json.Marshal(map[string]string{"model": model})
	resp, err := http.Post(baseURL+"/api/load", "application/json", bytes.NewReader(body))
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("server returned %d: %s", resp.StatusCode, string(respBody))
	}
	return nil
}

func init() {
	runCmd.Flags().Int("port", 11435, "server port")
	runCmd.Flags().String("system", "", "system prompt")
	chatCmd.Flags().String("model", "", "model to chat with")
	chatCmd.Flags().Int("port", 11435, "server port")
	chatCmd.Flags().String("system", "", "system prompt")
	rootCmd.AddCommand(runCmd)
	rootCmd.AddCommand(chatCmd)
}
