package cmd

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"github.com/thatcatdev/tanrenai/server/internal/agent"
	"github.com/thatcatdev/tanrenai/server/internal/chatctx"
	"github.com/thatcatdev/tanrenai/server/internal/config"
	"github.com/thatcatdev/tanrenai/server/internal/memory"
	"github.com/thatcatdev/tanrenai/server/internal/models"
	"github.com/thatcatdev/tanrenai/server/internal/runner"
	"github.com/thatcatdev/tanrenai/server/internal/tools"
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
		systemFile, _ := cmd.Flags().GetString("system-file")
		agentMode, _ := cmd.Flags().GetBool("agent")
		ctxSize, _ := cmd.Flags().GetInt("ctx-size")
		responseBudget, _ := cmd.Flags().GetInt("response-budget")
		contextFiles, _ := cmd.Flags().GetStringSlice("context-file")
		memoryEnabled, _ := cmd.Flags().GetBool("memory")
		embeddingModel, _ := cmd.Flags().GetString("embedding-model")
		embeddingPort, _ := cmd.Flags().GetInt("embedding-port")
		embeddingCtxSize, _ := cmd.Flags().GetInt("embedding-ctx-size")

		// Read system prompt from file if specified
		if systemFile != "" {
			data, err := os.ReadFile(systemFile)
			if err != nil {
				return fmt.Errorf("failed to read system file: %w", err)
			}
			systemPrompt = string(data)
		}

		// Load the model
		fmt.Printf("Loading model %s...\n", model)
		if err := loadModelRemote(baseURL, model); err != nil {
			return fmt.Errorf("failed to load model (is 'tanrenai serve' running?): %w", err)
		}
		fmt.Printf("Model %s loaded. Type your message (Ctrl+C to quit).\n\n", model)

		// Set up context manager
		estimator := chatctx.NewTokenEstimator()
		calibrateEstimator(baseURL, estimator)

		mgr := chatctx.NewManager(chatctx.Config{
			CtxSize:        ctxSize,
			ResponseBudget: responseBudget,
		}, estimator)

		// Load context files
		for _, path := range contextFiles {
			if err := loadContextFile(mgr, path); err != nil {
				fmt.Fprintf(os.Stderr, "Warning: failed to load context file %s: %v\n", path, err)
			}
		}

		// Initialize memory if enabled
		var memStore memory.Store
		if memoryEnabled && agentMode {
			store, cleanup, err := initMemory(context.Background(), embeddingModel, config.BinDir(), embeddingPort, embeddingCtxSize)
			if err != nil {
				return fmt.Errorf("failed to initialize memory: %w", err)
			}
			defer cleanup()
			memStore = store
			fmt.Printf("Memory enabled (%d stored memories)\n", memStore.Count())
		}

		if agentMode {
			return agentLoop(baseURL, model, systemPrompt, mgr, memStore)
		}
		return chatLoop(baseURL, model, systemPrompt, mgr)
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
		systemFile, _ := cmd.Flags().GetString("system-file")
		agentMode, _ := cmd.Flags().GetBool("agent")
		ctxSize, _ := cmd.Flags().GetInt("ctx-size")
		responseBudget, _ := cmd.Flags().GetInt("response-budget")
		contextFiles, _ := cmd.Flags().GetStringSlice("context-file")
		memoryEnabled, _ := cmd.Flags().GetBool("memory")
		embeddingModel, _ := cmd.Flags().GetString("embedding-model")
		embeddingPort, _ := cmd.Flags().GetInt("embedding-port")
		embeddingCtxSize, _ := cmd.Flags().GetInt("embedding-ctx-size")

		if model == "" {
			return fmt.Errorf("specify a model with --model")
		}

		// Read system prompt from file if specified
		if systemFile != "" {
			data, err := os.ReadFile(systemFile)
			if err != nil {
				return fmt.Errorf("failed to read system file: %w", err)
			}
			systemPrompt = string(data)
		}

		fmt.Printf("Chatting with %s. Type your message (Ctrl+C to quit).\n\n", model)

		// Set up context manager
		estimator := chatctx.NewTokenEstimator()
		calibrateEstimator(baseURL, estimator)

		mgr := chatctx.NewManager(chatctx.Config{
			CtxSize:        ctxSize,
			ResponseBudget: responseBudget,
		}, estimator)

		// Load context files
		for _, path := range contextFiles {
			if err := loadContextFile(mgr, path); err != nil {
				fmt.Fprintf(os.Stderr, "Warning: failed to load context file %s: %v\n", path, err)
			}
		}

		// Initialize memory if enabled
		var memStore memory.Store
		if memoryEnabled && agentMode {
			store, cleanup, err := initMemory(context.Background(), embeddingModel, config.BinDir(), embeddingPort, embeddingCtxSize)
			if err != nil {
				return fmt.Errorf("failed to initialize memory: %w", err)
			}
			defer cleanup()
			memStore = store
			fmt.Printf("Memory enabled (%d stored memories)\n", memStore.Count())
		}

		if agentMode {
			return agentLoop(baseURL, model, systemPrompt, mgr, memStore)
		}
		return chatLoop(baseURL, model, systemPrompt, mgr)
	},
}

// calibrateEstimator attempts to calibrate the token estimator against the server's tokenizer.
func calibrateEstimator(baseURL string, estimator *chatctx.TokenEstimator) {
	tokenizeFn := func(text string) (int, error) {
		payload := struct {
			Content string `json:"content"`
		}{Content: text}
		body, _ := json.Marshal(payload)

		resp, err := http.Post(baseURL+"/tokenize", "application/json", bytes.NewReader(body))
		if err != nil {
			return 0, err
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			return 0, fmt.Errorf("tokenize returned %d", resp.StatusCode)
		}

		var result struct {
			Tokens []int `json:"tokens"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			return 0, err
		}
		return len(result.Tokens), nil
	}

	if err := estimator.Calibrate(tokenizeFn); err != nil {
		// Calibration is optional; fall back to default ratio
		fmt.Fprintf(os.Stderr, "Note: token estimation using default ratio (calibration unavailable)\n")
	}
}

// loadContextFile reads a file and adds it to the context manager.
func loadContextFile(mgr *chatctx.Manager, path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	mgr.AddContextFile(path, string(data))
	fmt.Printf("Loaded context file: %s (%d bytes)\n", path, len(data))
	return nil
}

// handleREPLCommand processes REPL slash commands. Returns true if the input was a command.
func handleREPLCommand(input string, mgr *chatctx.Manager, memStore memory.Store) bool {
	switch {
	case input == "/clear":
		mgr.Clear()
		fmt.Println("History cleared. System prompt and context files preserved.")
		return true

	case input == "/tokens":
		budget := mgr.Budget()
		fmt.Printf("Token budget:\n")
		fmt.Printf("  Total context:  %d\n", budget.Total)
		fmt.Printf("  System/pinned:  %d\n", budget.System)
		fmt.Printf("  Memory:         %d\n", budget.Memory)
		fmt.Printf("  Summary:        %d\n", budget.Summary)
		fmt.Printf("  History:        %d (%d messages, %d total)\n", budget.History, budget.HistoryCount, budget.TotalHistory)
		fmt.Printf("  Available:      %d\n", budget.Available)
		return true

	case input == "/context list":
		files := mgr.ContextFiles()
		if len(files) == 0 {
			fmt.Println("No context files loaded.")
		} else {
			fmt.Println("Context files:")
			for _, f := range files {
				fmt.Printf("  - %s\n", f)
			}
		}
		return true

	case input == "/context clear":
		mgr.ClearContextFiles()
		fmt.Println("Context files cleared.")
		return true

	case strings.HasPrefix(input, "/context add "):
		path := strings.TrimPrefix(input, "/context add ")
		path = strings.TrimSpace(path)
		if path == "" {
			fmt.Println("Usage: /context add <file-path>")
			return true
		}
		if err := loadContextFile(mgr, path); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		}
		return true

	case input == "/memory" || input == "/memory list":
		if memStore == nil {
			fmt.Println("Memory is not enabled. Use --memory flag to enable.")
			return true
		}
		entries, err := memStore.List(context.Background(), 10)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error listing memories: %v\n", err)
			return true
		}
		fmt.Printf("Memories (%d total):\n", memStore.Count())
		for _, e := range entries {
			fmt.Printf("  [%s] %s — %s\n", e.ID[:8], e.Timestamp.Format("2006-01-02 15:04"), truncate(e.UserMsg, 80))
		}
		return true

	case strings.HasPrefix(input, "/memory search "):
		if memStore == nil {
			fmt.Println("Memory is not enabled. Use --memory flag to enable.")
			return true
		}
		query := strings.TrimPrefix(input, "/memory search ")
		query = strings.TrimSpace(query)
		if query == "" {
			fmt.Println("Usage: /memory search <query>")
			return true
		}
		results, err := memStore.Search(context.Background(), query, 5)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error searching memories: %v\n", err)
			return true
		}
		if len(results) == 0 {
			fmt.Println("No matching memories found.")
			return true
		}
		fmt.Printf("Search results (%d):\n", len(results))
		for _, r := range results {
			fmt.Printf("  [%s] score=%.3f (sem=%.3f kw=%.3f) %s\n",
				r.Entry.ID[:8], r.CombinedScore, r.SemanticScore, r.KeywordScore,
				truncate(r.Entry.UserMsg, 70))
		}
		return true

	case strings.HasPrefix(input, "/memory forget "):
		if memStore == nil {
			fmt.Println("Memory is not enabled. Use --memory flag to enable.")
			return true
		}
		idPrefix := strings.TrimPrefix(input, "/memory forget ")
		idPrefix = strings.TrimSpace(idPrefix)
		if idPrefix == "" {
			fmt.Println("Usage: /memory forget <id-prefix>")
			return true
		}
		// Find entry by prefix
		entries, _ := memStore.List(context.Background(), 0)
		for _, e := range entries {
			if strings.HasPrefix(e.ID, idPrefix) {
				if err := memStore.Delete(context.Background(), e.ID); err != nil {
					fmt.Fprintf(os.Stderr, "Error deleting memory: %v\n", err)
				} else {
					fmt.Printf("Deleted memory %s\n", e.ID[:8])
				}
				return true
			}
		}
		fmt.Printf("No memory found with prefix %q\n", idPrefix)
		return true

	case input == "/memory clear":
		if memStore == nil {
			fmt.Println("Memory is not enabled. Use --memory flag to enable.")
			return true
		}
		if err := memStore.Clear(context.Background()); err != nil {
			fmt.Fprintf(os.Stderr, "Error clearing memories: %v\n", err)
		} else {
			fmt.Println("All memories cleared.")
		}
		return true

	case input == "/help":
		fmt.Println("Commands:")
		fmt.Println("  /clear              - Clear conversation history")
		fmt.Println("  /tokens             - Show token budget breakdown")
		fmt.Println("  /context add <path> - Load file into context")
		fmt.Println("  /context list       - Show loaded context files")
		fmt.Println("  /context clear      - Remove all context files")
		fmt.Println("  /memory             - List recent memories")
		fmt.Println("  /memory search <q>  - Search memories")
		fmt.Println("  /memory forget <id> - Delete a memory by ID prefix")
		fmt.Println("  /memory clear       - Clear all memories")
		fmt.Println("  /quit, /exit        - Exit")
		return true
	}

	return false
}

func chatLoop(baseURL, model, systemPrompt string, mgr *chatctx.Manager) error {
	scanner := bufio.NewScanner(os.Stdin)

	if systemPrompt != "" {
		mgr.SetSystemPrompt(systemPrompt)
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
		if handleREPLCommand(input, mgr, nil) {
			continue
		}

		mgr.Append(api.Message{Role: "user", Content: input})

		// Get windowed messages for the request
		windowedMsgs := mgr.Messages()

		req := api.ChatCompletionRequest{
			Model:    model,
			Messages: windowedMsgs,
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
			mgr.Append(api.Message{Role: "assistant", Content: assistant})
		}
	}
}

const defaultAgentSystemPrompt = `You are a helpful assistant with access to tools. Use them when the user asks you to interact with the filesystem or run commands.

Tool usage rules:
- For file paths, use actual paths: "." for current directory, ".." for parent, or absolute paths like "/home/user".
- Never use placeholder names like "current_directory" — use "." instead.
- If a tool call fails, read the error message carefully and try a different approach. Do NOT repeat the exact same failing call.
- Follow through on multi-step tasks automatically. Do not stop to ask for confirmation on steps you can infer.
- Always prefer the provided tools (file_read, list_dir, etc.) over shell_exec. Only use shell_exec when no other tool can do the job.
- Use multiple tool calls in a single response when possible. For example, if you need to read 5 files, call file_read 5 times in one response rather than one at a time.
- Only respond with text once you have fully completed what the user asked for, or to briefly explain what you are about to do.`

// wrapStreamWithCleanup wraps a stream event channel, ensuring the HTTP response
// body is closed when the source channel is drained.
func wrapStreamWithCleanup(events <-chan runner.StreamEvent, body io.ReadCloser) <-chan runner.StreamEvent {
	out := make(chan runner.StreamEvent)
	go func() {
		defer body.Close()
		defer close(out)
		for ev := range events {
			out <- ev
		}
	}()
	return out
}

func agentLoop(baseURL, model, systemPrompt string, mgr *chatctx.Manager, memStore memory.Store) error {
	scanner := bufio.NewScanner(os.Stdin)

	// Always inject agent system prompt; append user's system prompt if provided
	agentSystem := defaultAgentSystemPrompt
	if systemPrompt != "" {
		agentSystem += "\n\n" + systemPrompt
	}
	mgr.SetSystemPrompt(agentSystem)

	registry := tools.DefaultRegistry()

	// Non-streaming CompletionFunc for auto-summarization
	completeFn := func(ctx context.Context, req *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error) {
		req.Model = model
		body, err := json.Marshal(req)
		if err != nil {
			return nil, fmt.Errorf("marshal request: %w", err)
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/v1/chat/completions", bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("create request: %w", err)
		}
		httpReq.Header.Set("Content-Type", "application/json")

		resp, err := http.DefaultClient.Do(httpReq)
		if err != nil {
			return nil, fmt.Errorf("send request: %w", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			respBody, _ := io.ReadAll(resp.Body)
			return nil, fmt.Errorf("server returned %d: %s", resp.StatusCode, string(respBody))
		}

		var result api.ChatCompletionResponse
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			return nil, fmt.Errorf("decode response: %w", err)
		}
		return &result, nil
	}

	// Streaming CompletionFunc that returns an event channel
	streamCompleteFn := func(ctx context.Context, req *api.ChatCompletionRequest) (<-chan runner.StreamEvent, error) {
		req.Model = model
		req.Stream = true
		body, err := json.Marshal(req)
		if err != nil {
			return nil, fmt.Errorf("marshal request: %w", err)
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/v1/chat/completions", bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("create request: %w", err)
		}
		httpReq.Header.Set("Content-Type", "application/json")

		resp, err := http.DefaultClient.Do(httpReq)
		if err != nil {
			return nil, fmt.Errorf("send request: %w", err)
		}

		if resp.StatusCode != http.StatusOK {
			respBody, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			return nil, fmt.Errorf("server returned %d: %s", resp.StatusCode, string(respBody))
		}

		events := runner.ParseSSEStream(resp.Body)
		return wrapStreamWithCleanup(events, resp.Body), nil
	}

	fmt.Println("Agent mode enabled. Tools: file_read, file_write, list_dir, shell_exec")
	fmt.Println()

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
		if handleREPLCommand(input, mgr, memStore) {
			continue
		}

		mgr.Append(api.Message{Role: "user", Content: input})

		// Retrieve and inject relevant memories
		if memStore != nil {
			results, err := memStore.Search(context.Background(), input, 5)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Warning: memory search failed: %v\n", err)
			} else if len(results) > 0 {
				var memMsgs []api.Message
				for _, r := range results {
					memContent := fmt.Sprintf("[Memory from %s] User asked: %s\nAssistant replied: %s",
						r.Entry.Timestamp.Format("2006-01-02"), r.Entry.UserMsg, r.Entry.AssistMsg)
					memMsgs = append(memMsgs, api.Message{Role: "system", Content: memContent})
				}
				mgr.SetMemories(memMsgs)
			} else {
				mgr.ClearMemories()
			}
		}

		// Auto-summarize if needed before calling the agent
		if mgr.NeedsSummary() {
			fmt.Println("[summarizing conversation...]")
			if err := mgr.Summarize(context.Background(), completeFn); err != nil {
				fmt.Fprintf(os.Stderr, "Warning: summarization failed: %v\n", err)
			}
		}

		// Get windowed messages for the agent
		windowedMsgs := mgr.Messages()

		cfg := agent.StreamingConfig{
			Config: agent.Config{
				MaxIterations: 20,
				Tools:         registry,
				Hooks: agent.Hooks{
					OnToolCall: func(call api.ToolCall) {
						fmt.Printf("\n[tool] %s(%s)\n", call.Function.Name, truncate(call.Function.Arguments, 200))
					},
					OnToolResult: func(call api.ToolCall, result string) {
						fmt.Printf("[result] %s\n", truncate(result, 500))
					},
				},
			},
			OnIterationStart: func(iteration, maxIterations int) {
				if iteration > 1 {
					fmt.Printf("\n[iteration %d/%d]\n", iteration, maxIterations)
				}
			},
			OnThinking: func() {
				fmt.Print("Thinking...")
			},
			OnThinkingDone: func() {
				// Clear the "Thinking..." line
				fmt.Print("\033[2K\r")
			},
			OnContentDelta: func(delta string) {
				fmt.Print(delta)
			},
		}

		result, err := agent.RunStreaming(context.Background(), streamCompleteFn, windowedMsgs, cfg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Agent error: %v\n", err)
		}

		// Extract only new messages added by the agent and append to manager
		newMsgs := result[len(windowedMsgs):]
		mgr.AppendMany(newMsgs)

		// Store the turn in memory asynchronously
		if memStore != nil {
			// Extract assistant content from new messages
			var assistContent string
			for _, msg := range newMsgs {
				if msg.Role == "assistant" && msg.Content != "" {
					if assistContent != "" {
						assistContent += "\n"
					}
					assistContent += msg.Content
				}
			}
			if assistContent != "" {
				go func() {
					entry := memory.Entry{
						UserMsg:   input,
						AssistMsg: assistContent,
						Timestamp: time.Now(),
					}
					if err := memStore.Add(context.Background(), entry); err != nil {
						fmt.Fprintf(os.Stderr, "Warning: failed to store memory: %v\n", err)
					}
				}()
			}
		}

		fmt.Println()
	}
}

// initMemory starts the embedding server and creates the memory store.
func initMemory(ctx context.Context, embeddingModel, binDir string, port, embeddingCtxSize int) (memory.Store, func(), error) {
	// Resolve embedding model path
	store := models.NewStore(config.ModelsDir())
	modelPath, err := store.Resolve(embeddingModel)
	if err != nil {
		return nil, nil, fmt.Errorf("embedding model %q not found — download it with 'tanrenai pull <url>': %w", embeddingModel, err)
	}

	// Start embedding server
	fmt.Println("Starting embedding server...")
	runner, err := memory.NewEmbeddingRunner(ctx, memory.EmbeddingRunnerConfig{
		ModelPath: modelPath,
		BinDir:    binDir,
		Port:      port,
		GPULayers: 0,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("start embedding server: %w", err)
	}
	fmt.Println("Embedding server ready.")

	// Auto-detect embedding context size from the model if not overridden.
	if embeddingCtxSize <= 0 {
		if detected := runner.ContextSize(ctx); detected > 0 {
			embeddingCtxSize = detected
			fmt.Printf("Detected embedding context: %d tokens\n", embeddingCtxSize)
		}
	}

	// Create embed function and memory store
	embedFunc := memory.NewLlamaEmbedFunc(runner.BaseURL(), embeddingCtxSize)
	memDir := config.MemoryDir()
	if err := os.MkdirAll(memDir, 0755); err != nil {
		runner.Close()
		return nil, nil, fmt.Errorf("create memory directory: %w", err)
	}

	memStore, err := memory.NewChromemStore(memDir, embedFunc)
	if err != nil {
		runner.Close()
		return nil, nil, fmt.Errorf("create memory store: %w", err)
	}

	cleanup := func() {
		memStore.Close()
		runner.Close()
	}

	return memStore, cleanup, nil
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

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}

func addRunFlags(cmd *cobra.Command) {
	cmd.Flags().Int("port", 11435, "server port")
	cmd.Flags().String("system", "", "system prompt")
	cmd.Flags().String("system-file", "", "read system prompt from file")
	cmd.Flags().Bool("agent", false, "enable agent mode with tool calling")
	cmd.Flags().Int("ctx-size", 4096, "context window size in tokens")
	cmd.Flags().Int("response-budget", 512, "tokens reserved for model response")
	cmd.Flags().StringSlice("context-file", nil, "files to load into context")
	cmd.Flags().Bool("memory", false, "enable memory/RAG (requires --agent)")
	cmd.Flags().String("embedding-model", "all-MiniLM-L6-v2.Q8_0", "embedding model name for memory")
	cmd.Flags().Int("embedding-port", 18081, "port for embedding server")
	cmd.Flags().Int("embedding-ctx-size", 0, "embedding model context window in tokens (0 = auto-detect from model)")
}

func init() {
	addRunFlags(runCmd)
	chatCmd.Flags().String("model", "", "model to chat with")
	addRunFlags(chatCmd)
	rootCmd.AddCommand(runCmd)
	rootCmd.AddCommand(chatCmd)
}
