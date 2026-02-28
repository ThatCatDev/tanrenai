package tools

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

const (
	maxGrepMatches = 100
	maxGrepOutput  = 32 * 1024 // 32KB
)

// GrepSearchTool searches file contents for a pattern.
type GrepSearchTool struct{}

type grepSearchArgs struct {
	Pattern   string `json:"pattern"`
	Path      string `json:"path"`
	FileGlob  string `json:"file_glob"`
	MaxResults int   `json:"max_results"`
}

func (t *GrepSearchTool) Name() string { return "grep_search" }

func (t *GrepSearchTool) Description() string {
	return "Search file contents for a regex pattern. Returns matching lines with file paths and line numbers. Searches recursively from the given path. Use file_glob to filter by file type (e.g. \"*.go\", \"*.py\")."
}

func (t *GrepSearchTool) Parameters() json.RawMessage {
	return Schema{
		Type: "object",
		Properties: map[string]SchemaProperty{
			"pattern":     {Type: "string", Description: "Regex pattern to search for (e.g. \"func main\", \"TODO\", \"import.*fmt\")"},
			"path":        {Type: "string", Description: "Directory or file to search in (default: \".\")"},
			"file_glob":   {Type: "string", Description: "Glob pattern to filter files (e.g. \"*.go\", \"*.ts\"). Empty = all files."},
			"max_results": {Type: "integer", Description: "Maximum number of matches to return (default: 50)"},
		},
		Required: []string{"pattern"},
	}.MustMarshal()
}

func (t *GrepSearchTool) Execute(_ context.Context, arguments string) (*ToolResult, error) {
	var args grepSearchArgs
	if err := json.Unmarshal([]byte(arguments), &args); err != nil {
		return ErrorResult(fmt.Sprintf("invalid arguments: %v", err)), nil
	}
	if args.Pattern == "" {
		return ErrorResult("pattern is required"), nil
	}
	if args.Path == "" {
		args.Path = "."
	}
	if args.MaxResults <= 0 {
		args.MaxResults = 50
	}
	if args.MaxResults > maxGrepMatches {
		args.MaxResults = maxGrepMatches
	}

	re, err := regexp.Compile(args.Pattern)
	if err != nil {
		return ErrorResult(fmt.Sprintf("invalid regex: %v", err)), nil
	}

	var b strings.Builder
	matches := 0

	walkErr := filepath.Walk(args.Path, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // skip unreadable entries
		}
		if matches >= args.MaxResults {
			return filepath.SkipAll
		}
		if info.IsDir() {
			name := info.Name()
			// Skip common non-source directories
			if name == ".git" || name == "node_modules" || name == "vendor" || name == "__pycache__" || name == ".venv" || name == "venv" {
				return filepath.SkipDir
			}
			return nil
		}

		// Skip binary/large files
		if info.Size() > 1024*1024 { // 1MB
			return nil
		}

		// Apply file glob filter
		if args.FileGlob != "" {
			matched, _ := filepath.Match(args.FileGlob, info.Name())
			if !matched {
				return nil
			}
		}

		// Skip likely binary files
		if isBinaryFilename(info.Name()) {
			return nil
		}

		f, err := os.Open(path)
		if err != nil {
			return nil
		}
		defer f.Close()

		scanner := bufio.NewScanner(f)
		lineNum := 0
		for scanner.Scan() {
			lineNum++
			line := scanner.Text()
			if re.MatchString(line) {
				fmt.Fprintf(&b, "%s:%d: %s\n", path, lineNum, line)
				matches++
				if matches >= args.MaxResults {
					break
				}
			}
		}
		return nil
	})

	if walkErr != nil {
		return ErrorResult(fmt.Sprintf("search failed: %v", walkErr)), nil
	}

	if matches == 0 {
		return &ToolResult{Output: fmt.Sprintf("No matches found for pattern %q", args.Pattern)}, nil
	}

	output := b.String()
	if len(output) > maxGrepOutput {
		output = output[:maxGrepOutput] + "\n[truncated]"
	}

	if matches >= args.MaxResults {
		output += fmt.Sprintf("\n[showing first %d matches]", args.MaxResults)
	}

	return &ToolResult{Output: output}, nil
}

func isBinaryFilename(name string) bool {
	ext := strings.ToLower(filepath.Ext(name))
	switch ext {
	case ".exe", ".bin", ".so", ".dylib", ".dll", ".o", ".a",
		".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
		".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
		".gguf", ".safetensors", ".pt", ".onnx",
		".wasm", ".pyc", ".class":
		return true
	}
	return false
}
