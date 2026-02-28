package cmd

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/ThatCatDev/tanrenai/client/internal/apiclient"
	"github.com/ThatCatDev/tanrenai/client/internal/chatctx"
	"github.com/ThatCatDev/tanrenai/client/internal/tools"
	"github.com/ThatCatDev/tanrenai/client/pkg/api"
	"github.com/spf13/cobra"
)

const defaultAgentSystemPrompt = `You are a helpful assistant with access to tools for interacting with the filesystem and system.

Important rules:
1. Gather information with tools BEFORE answering. Do not guess or speculate when you can look it up.
2. Call tools directly — do not narrate what you plan to do. Just do it.
3. Use multiple tool calls in one response when possible.
4. Complete multi-step tasks automatically without stopping for confirmation.
5. Only use shell_exec when no other tool fits. Prefer file_read, list_dir, grep_search, find_files.
6. Use "." for the current directory. Never use placeholder names.
7. If a tool call fails, try different arguments. Never repeat an identical failing call.
8. To edit existing files, use patch_file. Only use file_write for creating new files or when you need to rewrite the entire file. Always use file_read first to understand what you're changing.
9. After making changes, verify your work by building or running tests with shell_exec.`

var runCmd = &cobra.Command{
	Use:   "run <model>",
	Short: "Load a model and start an interactive chat",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		model := args[0]
		systemPrompt, _ := cmd.Flags().GetString("system")
		systemFile, _ := cmd.Flags().GetString("system-file")
		agentMode, _ := cmd.Flags().GetBool("agent")
		ctxSize, _ := cmd.Flags().GetInt("ctx-size")
		responseBudget, _ := cmd.Flags().GetInt("response-budget")
		contextFiles, _ := cmd.Flags().GetStringSlice("context-file")
		memoryEnabled, _ := cmd.Flags().GetBool("memory")
		maxIterations, _ := cmd.Flags().GetInt("max-iterations")

		if systemFile != "" {
			data, err := os.ReadFile(systemFile)
			if err != nil {
				return fmt.Errorf("failed to read system file: %w", err)
			}
			systemPrompt = string(data)
		}

		client := apiclient.New(serverURL)

		fmt.Printf("Loading model %s...\n", model)
		if err := client.LoadModel(cmd.Context(), model); err != nil {
			return fmt.Errorf("failed to load model (is the backend running?): %w", err)
		}

		estimator := chatctx.NewTokenEstimator()
		calibrateEstimator(client, estimator)

		toolsBudget := 0
		if agentMode {
			toolsBudget = 4000
		}

		mgr := chatctx.NewManager(chatctx.Config{
			CtxSize:        ctxSize,
			ResponseBudget: responseBudget,
			ToolsBudget:    toolsBudget,
		}, estimator)

		for _, path := range contextFiles {
			if err := loadContextFile(mgr, path); err != nil {
				fmt.Fprintf(os.Stderr, "Warning: failed to load context file %s: %v\n", path, err)
			}
		}

		if memoryEnabled && agentMode {
			count, err := client.MemoryCount(cmd.Context())
			if err != nil {
				fmt.Fprintf(os.Stderr, "Warning: memory not available: %v\n", err)
				memoryEnabled = false
			} else {
				fmt.Printf("Memory enabled (%d stored memories)\n", count)
			}
		}

		return startTUI(client, model, systemPrompt, mgr, agentMode, memoryEnabled, maxIterations)
	},
}

var chatCmd = &cobra.Command{
	Use:   "chat",
	Short: "Interactive chat with a loaded model",
	RunE: func(cmd *cobra.Command, args []string) error {
		model, _ := cmd.Flags().GetString("model")
		systemPrompt, _ := cmd.Flags().GetString("system")
		systemFile, _ := cmd.Flags().GetString("system-file")
		agentMode, _ := cmd.Flags().GetBool("agent")
		ctxSize, _ := cmd.Flags().GetInt("ctx-size")
		responseBudget, _ := cmd.Flags().GetInt("response-budget")
		contextFiles, _ := cmd.Flags().GetStringSlice("context-file")
		memoryEnabled, _ := cmd.Flags().GetBool("memory")
		maxIterations, _ := cmd.Flags().GetInt("max-iterations")

		if model == "" {
			return fmt.Errorf("specify a model with --model")
		}

		if systemFile != "" {
			data, err := os.ReadFile(systemFile)
			if err != nil {
				return fmt.Errorf("failed to read system file: %w", err)
			}
			systemPrompt = string(data)
		}

		client := apiclient.New(serverURL)

		estimator := chatctx.NewTokenEstimator()
		calibrateEstimator(client, estimator)

		toolsBudget := 0
		if agentMode {
			toolsBudget = 4000
		}

		mgr := chatctx.NewManager(chatctx.Config{
			CtxSize:        ctxSize,
			ResponseBudget: responseBudget,
			ToolsBudget:    toolsBudget,
		}, estimator)

		for _, path := range contextFiles {
			if err := loadContextFile(mgr, path); err != nil {
				fmt.Fprintf(os.Stderr, "Warning: failed to load context file %s: %v\n", path, err)
			}
		}

		if memoryEnabled && agentMode {
			count, err := client.MemoryCount(cmd.Context())
			if err != nil {
				fmt.Fprintf(os.Stderr, "Warning: memory not available: %v\n", err)
				memoryEnabled = false
			} else {
				fmt.Printf("Memory enabled (%d stored memories)\n", count)
			}
		}

		return startTUI(client, model, systemPrompt, mgr, agentMode, memoryEnabled, maxIterations)
	},
}

func startTUI(client *apiclient.Client, model, systemPrompt string, mgr *chatctx.Manager, agentMode, memoryEnabled bool, maxIterations int) error {
	if agentMode {
		agentSystem := defaultAgentSystemPrompt
		if systemPrompt != "" {
			agentSystem += "\n\n" + systemPrompt
		}
		mgr.SetSystemPrompt(agentSystem)
	} else if systemPrompt != "" {
		mgr.SetSystemPrompt(systemPrompt)
	}

	var registry *tools.Registry
	if agentMode {
		registry = tools.DefaultRegistry()
	}

	completeFn := func(ctx context.Context, req *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error) {
		req.Model = model
		return client.ChatCompletion(ctx, req)
	}

	streamFn := func(ctx context.Context, req *api.ChatCompletionRequest) (<-chan apiclient.StreamEvent, error) {
		req.Model = model
		return client.StreamCompletion(ctx, req)
	}

	m := newTUIModel(client, model, mgr, registry, memoryEnabled, maxIterations, agentMode, completeFn, streamFn)

	p := tea.NewProgram(m, tea.WithAltScreen(), tea.WithMouseCellMotion())
	m.shared.program = p

	_, err := p.Run()
	return err
}

func calibrateEstimator(client *apiclient.Client, estimator *chatctx.TokenEstimator) {
	tokenizeFn := func(text string) (int, error) {
		return client.Tokenize(context.Background(), text)
	}
	if err := estimator.Calibrate(tokenizeFn); err != nil {
		fmt.Fprintf(os.Stderr, "Note: token estimation using default ratio (calibration unavailable)\n")
	}
}

func loadContextFile(mgr *chatctx.Manager, path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	mgr.AddContextFile(path, string(data))
	fmt.Printf("Loaded context file: %s (%d bytes)\n", path, len(data))
	return nil
}

func handleREPLCommand(w io.Writer, input string, mgr *chatctx.Manager, client *apiclient.Client, memoryEnabled bool) bool {
	switch {
	case input == "/clear":
		mgr.Clear()
		fmt.Fprintln(w, "History cleared. System prompt and context files preserved.")
		return true

	case input == "/tokens":
		budget := mgr.Budget()
		fmt.Fprintf(w, "Token budget:\n")
		fmt.Fprintf(w, "  Total context:  %d\n", budget.Total)
		fmt.Fprintf(w, "  System/pinned:  %d\n", budget.System)
		fmt.Fprintf(w, "  Memory:         %d\n", budget.Memory)
		fmt.Fprintf(w, "  Summary:        %d\n", budget.Summary)
		fmt.Fprintf(w, "  History:        %d (%d messages, %d total)\n", budget.History, budget.HistoryCount, budget.TotalHistory)
		fmt.Fprintf(w, "  Available:      %d\n", budget.Available)
		return true

	case input == "/context list":
		files := mgr.ContextFiles()
		if len(files) == 0 {
			fmt.Fprintln(w, "No context files loaded.")
		} else {
			fmt.Fprintln(w, "Context files:")
			for _, f := range files {
				fmt.Fprintf(w, "  - %s\n", f)
			}
		}
		return true

	case input == "/context clear":
		mgr.ClearContextFiles()
		fmt.Fprintln(w, "Context files cleared.")
		return true

	case strings.HasPrefix(input, "/context add "):
		path := strings.TrimPrefix(input, "/context add ")
		path = strings.TrimSpace(path)
		if path == "" {
			fmt.Fprintln(w, "Usage: /context add <file-path>")
			return true
		}
		if err := loadContextFile(mgr, path); err != nil {
			fmt.Fprintf(w, "Error: %v\n", err)
		}
		return true

	case input == "/memory" || input == "/memory list":
		if !memoryEnabled {
			fmt.Fprintln(w, "Memory is not enabled. Use --memory flag to enable.")
			return true
		}
		resp, err := client.MemoryList(context.Background(), 10)
		if err != nil {
			fmt.Fprintf(w, "Error listing memories: %v\n", err)
			return true
		}
		fmt.Fprintf(w, "Memories (%d total):\n", resp.Total)
		for _, e := range resp.Entries {
			fmt.Fprintf(w, "  [%s] %s — %s\n", e.ID[:8], e.Timestamp.Format("2006-01-02 15:04"), truncate(e.UserMsg, 80))
		}
		return true

	case strings.HasPrefix(input, "/memory search "):
		if !memoryEnabled {
			fmt.Fprintln(w, "Memory is not enabled. Use --memory flag to enable.")
			return true
		}
		query := strings.TrimPrefix(input, "/memory search ")
		query = strings.TrimSpace(query)
		if query == "" {
			fmt.Fprintln(w, "Usage: /memory search <query>")
			return true
		}
		resp, err := client.MemorySearch(context.Background(), query, 5)
		if err != nil {
			fmt.Fprintf(w, "Error searching memories: %v\n", err)
			return true
		}
		if len(resp.Results) == 0 {
			fmt.Fprintln(w, "No matching memories found.")
			return true
		}
		fmt.Fprintf(w, "Search results (%d):\n", len(resp.Results))
		for _, r := range resp.Results {
			fmt.Fprintf(w, "  [%s] score=%.3f (sem=%.3f kw=%.3f) %s\n",
				r.Entry.ID[:8], r.CombinedScore, r.SemanticScore, r.KeywordScore,
				truncate(r.Entry.UserMsg, 70))
		}
		return true

	case strings.HasPrefix(input, "/memory forget "):
		if !memoryEnabled {
			fmt.Fprintln(w, "Memory is not enabled. Use --memory flag to enable.")
			return true
		}
		idPrefix := strings.TrimPrefix(input, "/memory forget ")
		idPrefix = strings.TrimSpace(idPrefix)
		if idPrefix == "" {
			fmt.Fprintln(w, "Usage: /memory forget <id-prefix>")
			return true
		}
		resp, err := client.MemoryList(context.Background(), 0)
		if err != nil {
			fmt.Fprintf(w, "Error: %v\n", err)
			return true
		}
		for _, e := range resp.Entries {
			if strings.HasPrefix(e.ID, idPrefix) {
				if err := client.MemoryDelete(context.Background(), e.ID); err != nil {
					fmt.Fprintf(w, "Error deleting memory: %v\n", err)
				} else {
					fmt.Fprintf(w, "Deleted memory %s\n", e.ID[:8])
				}
				return true
			}
		}
		fmt.Fprintf(w, "No memory found with prefix %q\n", idPrefix)
		return true

	case input == "/memory clear":
		if !memoryEnabled {
			fmt.Fprintln(w, "Memory is not enabled. Use --memory flag to enable.")
			return true
		}
		if err := client.MemoryClear(context.Background()); err != nil {
			fmt.Fprintf(w, "Error clearing memories: %v\n", err)
		} else {
			fmt.Fprintln(w, "All memories cleared.")
		}
		return true

	case input == "/help":
		fmt.Fprintln(w, "Commands:")
		fmt.Fprintln(w, "  /clear                        - Clear conversation history")
		fmt.Fprintln(w, "  /compact                      - Summarize conversation to free context")
		fmt.Fprintln(w, "  /tokens                       - Show token budget breakdown")
		fmt.Fprintln(w, "  /context add <path>           - Load file into context")
		fmt.Fprintln(w, "  /context list                 - Show loaded context files")
		fmt.Fprintln(w, "  /context clear                - Remove all context files")
		fmt.Fprintln(w, "  /memory                       - List recent memories")
		fmt.Fprintln(w, "  /memory search <q>            - Search memories")
		fmt.Fprintln(w, "  /memory forget <id>           - Delete a memory by ID prefix")
		fmt.Fprintln(w, "  /memory clear                 - Clear all memories")
		fmt.Fprintln(w, "  /quit, /exit                  - Exit")
		return true
	}

	return false
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}

func addRunFlags(cmd *cobra.Command) {
	cmd.Flags().String("system", "", "system prompt")
	cmd.Flags().String("system-file", "", "read system prompt from file")
	cmd.Flags().Bool("agent", false, "enable agent mode with tool calling")
	cmd.Flags().Int("ctx-size", 4096, "context window size in tokens")
	cmd.Flags().Int("response-budget", 512, "tokens reserved for model response")
	cmd.Flags().StringSlice("context-file", nil, "files to load into context")
	cmd.Flags().Bool("memory", false, "enable memory/RAG")
	cmd.Flags().Int("max-iterations", 200, "maximum agent tool-call iterations per turn (0 = unlimited)")
}

var _ = time.Now

func init() {
	addRunFlags(runCmd)
	chatCmd.Flags().String("model", "", "model to chat with")
	addRunFlags(chatCmd)
	rootCmd.AddCommand(runCmd)
	rootCmd.AddCommand(chatCmd)
}
