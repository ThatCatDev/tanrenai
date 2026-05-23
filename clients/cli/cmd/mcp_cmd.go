package cmd

import (
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"
	"text/tabwriter"

	"github.com/spf13/cobra"

	"github.com/ThatCatDev/tanrenai/shared/mcp"
)

// `tanrenai mcp` — manage external MCP servers for the agent. Reads
// and writes the same .tanrenai/mcp.json (project) and
// ~/.config/tanrenai/mcp.json (user) files the agent loads at startup,
// so changes here take effect on the next `tanrenai run` without any
// further setup.

var mcpCmd = &cobra.Command{
	Use:   "mcp",
	Short: "Manage external MCP servers",
	Long: "Register Model Context Protocol servers whose tools become available " +
		"to the agent. Stored in .tanrenai/mcp.json (project, default) or " +
		"~/.config/tanrenai/mcp.json (--scope user).",
}

var mcpAddCmd = &cobra.Command{
	Use:   "add <name> [-- <command> [args...]]",
	Short: "Add an MCP server (stdio or HTTP)",
	Long: `Add an MCP server config. Two flavours:

  Stdio (subprocess):
    tanrenai mcp add playwright -- npx -y @playwright/mcp@latest

  HTTP:
    tanrenai mcp add github --transport http https://api.githubcopilot.com/mcp/ \
                            --header "Authorization: Bearer ghp_..."

The everything-after-` + "`--`" + ` syntax mirrors ` + "`claude mcp add`" + ` so
copy-pasting from claude docs works. Use --scope user to write to the
per-user config instead of the project config.`,
	Args: cobra.MinimumNArgs(1),
	RunE: runMCPAdd,
}

var mcpListCmd = &cobra.Command{
	Use:   "list",
	Short: "List configured MCP servers",
	Long: "Shows servers from the merged project + user config (project " +
		"overrides user on collision). Disabled servers are shown with a " +
		"`(disabled)` tag.",
	RunE: runMCPList,
}

var mcpRemoveCmd = &cobra.Command{
	Use:     "remove <name>",
	Aliases: []string{"rm"},
	Short:   "Remove an MCP server",
	Args:    cobra.ExactArgs(1),
	RunE:    runMCPRemove,
}

func init() {
	mcpAddCmd.Flags().String("scope", "project", "config scope: project (default) or user")
	mcpAddCmd.Flags().String("transport", "stdio", "transport: stdio (subprocess) or http")
	mcpAddCmd.Flags().StringSliceP("header", "H", nil, "HTTP header (repeatable), e.g. -H \"Authorization: Bearer xyz\"")
	mcpAddCmd.Flags().StringSliceP("env", "e", nil, "stdio env override (repeatable), e.g. -e API_KEY=xyz")
	mcpAddCmd.Flags().Bool("disabled", false, "add but mark disabled (skipped at startup)")

	mcpRemoveCmd.Flags().String("scope", "project", "config scope: project (default) or user")
	mcpListCmd.Flags().String("scope", "merged", "scope to list: merged (default), project, or user")

	mcpCmd.AddCommand(mcpAddCmd, mcpListCmd, mcpRemoveCmd)
	rootCmd.AddCommand(mcpCmd)
}

// configPath resolves the file path for a given scope. Empty workspace
// is OK for user scope. Centralised here so add/remove use the exact
// same resolution as the agent's loader.
func configPath(scope string) (string, error) {
	switch scope {
	case "project":
		wd, err := os.Getwd()
		if err != nil {
			return "", fmt.Errorf("get working directory: %w", err)
		}
		return mcp.ProjectConfigPath(wd), nil
	case "user":
		return mcp.UserConfigPath()
	default:
		return "", fmt.Errorf("invalid --scope %q (want: project, user)", scope)
	}
}

func runMCPAdd(cmd *cobra.Command, args []string) error {
	name := args[0]
	if name == "" || strings.Contains(name, mcp.NameSeparator) {
		// Names with `__` would collide with the namespacing the
		// agent uses on tool registration. Reject early.
		return fmt.Errorf("server name %q is invalid (must not be empty or contain %q)", name, mcp.NameSeparator)
	}

	scope, _ := cmd.Flags().GetString("scope")
	transport, _ := cmd.Flags().GetString("transport")
	rawHeaders, _ := cmd.Flags().GetStringSlice("header")
	rawEnv, _ := cmd.Flags().GetStringSlice("env")
	disabled, _ := cmd.Flags().GetBool("disabled")

	server := mcp.ServerConfig{Disabled: disabled}

	switch transport {
	case "stdio":
		// Args after the bare name are the command + its args. The
		// `--` separator before the command is a cobra convention but
		// is consumed by the parser; we just take everything past the
		// name.
		if len(args) < 2 {
			return errors.New("stdio transport needs a command — e.g. `tanrenai mcp add foo -- npx -y @foo/bar`")
		}
		server.Command = args[1]
		if len(args) > 2 {
			server.Args = args[2:]
		}
		if len(rawEnv) > 0 {
			server.Env = parseEnvFlags(rawEnv)
		}
	case "http":
		if len(args) < 2 {
			return errors.New("http transport needs a URL — e.g. `tanrenai mcp add foo --transport http https://example.com/mcp`")
		}
		server.URL = args[1]
		if len(rawHeaders) > 0 {
			h, err := parseHeaderFlags(rawHeaders)
			if err != nil {
				return err
			}
			server.Headers = h
		}
	default:
		return fmt.Errorf("--transport %q invalid (want: stdio, http)", transport)
	}

	path, err := configPath(scope)
	if err != nil {
		return err
	}
	cfg, err := mcp.LoadFile(path)
	if err != nil {
		return fmt.Errorf("read existing config: %w", err)
	}
	if cfg.Servers == nil {
		cfg.Servers = map[string]mcp.ServerConfig{}
	}
	if _, exists := cfg.Servers[name]; exists {
		// Overwriting silently would hide config drift across PRs.
		// Force users to remove + re-add explicitly when they want
		// to change something.
		return fmt.Errorf("server %q already exists in %s — `tanrenai mcp remove %s` first", name, path, name)
	}
	cfg.Servers[name] = server

	// Validate before writing so we never persist a broken file the
	// agent will fail to load later.
	if err := cfg.Validate(); err != nil {
		return fmt.Errorf("invalid resulting config: %w", err)
	}
	if err := mcp.SaveFile(path, cfg); err != nil {
		return err
	}
	fmt.Fprintf(os.Stdout, "Added MCP server %q (%s) to %s\n", name, transport, path)
	return nil
}

func runMCPList(cmd *cobra.Command, _ []string) error {
	scope, _ := cmd.Flags().GetString("scope")

	var cfg mcp.Config
	var sources []string
	switch scope {
	case "merged":
		wd, _ := os.Getwd()
		var err error
		cfg, err = mcp.Load(wd)
		if err != nil {
			return err
		}
		// For the merged view, annotate origin per row by re-reading
		// each scope. Done after the join so users see which file an
		// entry came from.
		sources = []string{"project", "user"}
	case "project":
		path, err := configPath("project")
		if err != nil {
			return err
		}
		cfg, err = mcp.LoadFile(path)
		if err != nil {
			return err
		}
		sources = []string{"project"}
	case "user":
		path, err := configPath("user")
		if err != nil {
			return err
		}
		cfg, err = mcp.LoadFile(path)
		if err != nil {
			return err
		}
		sources = []string{"user"}
	default:
		return fmt.Errorf("invalid --scope %q (want: merged, project, user)", scope)
	}

	if len(cfg.Servers) == 0 {
		fmt.Fprintln(os.Stdout, "(no MCP servers configured)")
		return nil
	}

	// Per-entry origin annotation for the merged view. Re-read each
	// scope separately; the entry in the merged map matches whichever
	// scope provided it (project wins on collision).
	origin := map[string]string{}
	if len(sources) > 1 {
		proj, _ := loadScopeQuiet("project")
		user, _ := loadScopeQuiet("user")
		for name := range cfg.Servers {
			switch {
			case hasServer(proj, name):
				origin[name] = "project"
			case hasServer(user, name):
				origin[name] = "user"
			}
		}
	} else {
		for name := range cfg.Servers {
			origin[name] = sources[0]
		}
	}

	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "NAME\tTRANSPORT\tTARGET\tSCOPE\tNOTES")
	names := make([]string, 0, len(cfg.Servers))
	for n := range cfg.Servers {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		s := cfg.Servers[n]
		target := s.Command
		if s.URL != "" {
			target = s.URL
		}
		if len(s.Args) > 0 {
			target = target + " " + strings.Join(s.Args, " ")
		}
		notes := ""
		if s.Disabled {
			notes = "disabled"
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\n", n, s.Transport(), target, origin[n], notes)
	}
	return tw.Flush()
}

func runMCPRemove(cmd *cobra.Command, args []string) error {
	name := args[0]
	scope, _ := cmd.Flags().GetString("scope")
	path, err := configPath(scope)
	if err != nil {
		return err
	}
	cfg, err := mcp.LoadFile(path)
	if err != nil {
		return fmt.Errorf("read config: %w", err)
	}
	if _, ok := cfg.Servers[name]; !ok {
		return fmt.Errorf("server %q not found in %s", name, path)
	}
	delete(cfg.Servers, name)
	if err := mcp.SaveFile(path, cfg); err != nil {
		return err
	}
	fmt.Fprintf(os.Stdout, "Removed MCP server %q from %s\n", name, path)
	return nil
}

// loadScopeQuiet returns an empty config on any error — used only by
// the merged-list view to annotate origins, where a missing file
// shouldn't taint the main output. Real load errors (parse failures)
// would have surfaced from the prior mcp.Load() call.
func loadScopeQuiet(scope string) (mcp.Config, error) {
	path, err := configPath(scope)
	if err != nil {
		return mcp.Config{}, err
	}
	return mcp.LoadFile(path)
}

func hasServer(cfg mcp.Config, name string) bool {
	_, ok := cfg.Servers[name]
	return ok
}

// parseHeaderFlags splits each "Header: value" string into a map entry.
// Header names are case-insensitive on the wire, but we keep whatever
// the user typed so the value carries through verbatim (useful for
// servers picky about, e.g., "X-API-Key" capitalisation).
func parseHeaderFlags(raw []string) (map[string]string, error) {
	out := map[string]string{}
	for _, h := range raw {
		i := strings.Index(h, ":")
		if i < 0 {
			return nil, fmt.Errorf("invalid header %q: expected `Name: value`", h)
		}
		name := strings.TrimSpace(h[:i])
		value := strings.TrimSpace(h[i+1:])
		if name == "" {
			return nil, fmt.Errorf("invalid header %q: empty name", h)
		}
		out[name] = value
	}
	return out, nil
}

// parseEnvFlags splits each "KEY=value" string into a map entry.
// `=` in the value is preserved (only the first `=` separates).
func parseEnvFlags(raw []string) map[string]string {
	out := map[string]string{}
	for _, e := range raw {
		i := strings.Index(e, "=")
		if i < 0 {
			// "KEY" with no value → empty string. Matches the
			// behaviour of `env` and most shells' export syntax.
			out[e] = ""
			continue
		}
		out[e[:i]] = e[i+1:]
	}
	return out
}
