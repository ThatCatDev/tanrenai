package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/PuerkitoBio/goquery"
)

// WebSearchTool searches the web for information.
type WebSearchTool struct{}

type webSearchArgs struct {
	Query      string `json:"query"`
	MaxResults int    `json:"max_results,omitempty"`
}

func (t *WebSearchTool) Name() string { return "web_search" }

func (t *WebSearchTool) Description() string {
	return "Search the web for information. Returns titles, URLs, and snippets from search results."
}

func (t *WebSearchTool) Parameters() json.RawMessage {
	return Schema{
		Type: "object",
		Properties: map[string]SchemaProperty{
			"query":       {Type: "string", Description: "The search query to use"},
			"max_results": {Type: "integer", Description: "Maximum number of results to return (default: 5)"},
		},
		Required: []string{"query"},
	}.MustMarshal()
}

func (t *WebSearchTool) Execute(ctx context.Context, arguments string) (*ToolResult, error) {
	var args webSearchArgs
	if err := json.Unmarshal([]byte(arguments), &args); err != nil {
		return ErrorResult(fmt.Sprintf("invalid arguments: %v", err)), nil
	}

	if args.Query == "" {
		return ErrorResult("query is required"), nil
	}

	if args.MaxResults <= 0 {
		args.MaxResults = 5
	}

	results, err := searchDuckDuckGo(ctx, args.Query, args.MaxResults)
	if err != nil {
		return ErrorResult(fmt.Sprintf("web search failed: %v", err)), nil
	}

	if len(results) == 0 {
		return &ToolResult{Output: fmt.Sprintf("No results found for %q", args.Query)}, nil
	}

	var out strings.Builder
	out.WriteString(fmt.Sprintf("Search results for %q:\n\n", args.Query))
	for i, r := range results {
		out.WriteString(fmt.Sprintf("%d. %s\n   %s\n", i+1, r.Title, r.URL))
		if r.Snippet != "" {
			out.WriteString(fmt.Sprintf("   %s\n", r.Snippet))
		}
		if i < len(results)-1 {
			out.WriteString("\n")
		}
	}

	return &ToolResult{Output: out.String()}, nil
}

// searchResult holds a single search result.
type searchResult struct {
	Title   string
	URL     string
	Snippet string
}

// searchDuckDuckGo queries DuckDuckGo Lite via POST, which avoids CAPTCHA
// challenges that the HTML endpoint returns for GET requests.
func searchDuckDuckGo(ctx context.Context, query string, maxResults int) ([]searchResult, error) {
	form := url.Values{"q": {query}}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://lite.duckduckgo.com/lite/", strings.NewReader(form.Encode()))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)")

	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	doc, err := goquery.NewDocumentFromReader(resp.Body)
	if err != nil {
		return nil, err
	}

	// DuckDuckGo Lite uses <a class="result-link"> for titles/URLs
	// followed by <td class="result-snippet"> for the description.
	var results []searchResult
	doc.Find("a.result-link").Each(func(i int, s *goquery.Selection) {
		if i >= maxResults {
			return
		}
		title := strings.TrimSpace(s.Text())
		href, exists := s.Attr("href")
		if !exists || title == "" {
			return
		}

		// The snippet is in the next <td class="result-snippet"> sibling
		snippet := ""
		snippetTd := s.Closest("tr").Next().Find("td.result-snippet")
		if snippetTd.Length() > 0 {
			snippet = strings.TrimSpace(snippetTd.Text())
		}

		results = append(results, searchResult{
			Title:   title,
			URL:     href,
			Snippet: snippet,
		})
	})

	return results, nil
}
