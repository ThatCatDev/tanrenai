package scrolls

import (
	"os"
	"path/filepath"
	"testing"
)

func writeScroll(t *testing.T, dir, filename, content string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, filename), []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
}

func TestParseValidScroll(t *testing.T) {
	dir := t.TempDir()
	writeScroll(t, dir, "deploy.md", `---
name: deploy-gpu-server
description: How to deploy the GPU server to vast.ai with Headscale
tags: [deploy, gpu, vastai, headscale]
---

## Steps
1. First step
2. Second step
`)

	s, err := parse(filepath.Join(dir, "deploy.md"), "project")
	if err != nil {
		t.Fatal(err)
	}
	if s.Name != "deploy-gpu-server" {
		t.Errorf("name = %q, want %q", s.Name, "deploy-gpu-server")
	}
	if s.Description != "How to deploy the GPU server to vast.ai with Headscale" {
		t.Errorf("description = %q", s.Description)
	}
	if len(s.Tags) != 4 {
		t.Errorf("tags = %v, want 4 tags", s.Tags)
	}
	if s.Source != "project" {
		t.Errorf("source = %q, want %q", s.Source, "project")
	}
	if s.Content == "" {
		t.Error("content is empty")
	}
}

func TestParseScrollMissingTags(t *testing.T) {
	dir := t.TempDir()
	writeScroll(t, dir, "notags.md", `---
name: no-tags-scroll
description: A scroll without tags
---

Some content here.
`)

	s, err := parse(filepath.Join(dir, "notags.md"), "global")
	if err != nil {
		t.Fatal(err)
	}
	if s.Name != "no-tags-scroll" {
		t.Errorf("name = %q", s.Name)
	}
	if len(s.Tags) != 0 {
		t.Errorf("tags = %v, want empty", s.Tags)
	}
}

func TestParseMalformedFrontmatter(t *testing.T) {
	dir := t.TempDir()
	writeScroll(t, dir, "bad.md", `No frontmatter here.
Just regular markdown.
`)

	_, err := parse(filepath.Join(dir, "bad.md"), "project")
	if err == nil {
		t.Error("expected error for missing frontmatter")
	}
}

func TestParseMissingName(t *testing.T) {
	dir := t.TempDir()
	writeScroll(t, dir, "noname.md", `---
description: Missing name field
tags: [test]
---

Content.
`)

	_, err := parse(filepath.Join(dir, "noname.md"), "project")
	if err == nil {
		t.Error("expected error for missing name")
	}
}

func TestMatchReturnsRelevant(t *testing.T) {
	all := []Scroll{
		{Name: "deploy-gpu-server", Description: "Deploy GPU to vast.ai", Tags: []string{"deploy", "gpu", "vastai"}},
		{Name: "setup-headscale", Description: "Configure Headscale tunnel", Tags: []string{"headscale", "network", "tunnel"}},
		{Name: "train-model", Description: "Fine-tune a model", Tags: []string{"train", "finetune", "model"}},
	}

	results := Match(all, "how do I deploy the gpu server?", 3)
	if len(results) == 0 {
		t.Fatal("expected at least one match")
	}
	if results[0].Name != "deploy-gpu-server" {
		t.Errorf("top match = %q, want %q", results[0].Name, "deploy-gpu-server")
	}
}

func TestMatchReturnsNothing(t *testing.T) {
	all := []Scroll{
		{Name: "deploy-gpu-server", Description: "Deploy GPU to vast.ai", Tags: []string{"deploy", "gpu", "vastai"}},
	}

	results := Match(all, "what is the meaning of life?", 3)
	if len(results) != 0 {
		t.Errorf("expected no matches, got %d", len(results))
	}
}

func TestProjectOverridesGlobal(t *testing.T) {
	globalDir := t.TempDir()
	projectDir := t.TempDir()

	writeScroll(t, globalDir, "deploy.md", `---
name: deploy-guide
description: Global deploy guide
tags: [deploy]
---

Global content.
`)

	writeScroll(t, projectDir, "deploy.md", `---
name: deploy-guide
description: Project deploy guide
tags: [deploy]
---

Project content.
`)

	scrolls, err := Load(projectDir, globalDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(scrolls) != 1 {
		t.Fatalf("expected 1 scroll, got %d", len(scrolls))
	}
	if scrolls[0].Source != "project" {
		t.Errorf("source = %q, want %q", scrolls[0].Source, "project")
	}
	if scrolls[0].Description != "Project deploy guide" {
		t.Errorf("description = %q, want project version", scrolls[0].Description)
	}
}

func TestMaxResultsCap(t *testing.T) {
	var all []Scroll
	for i := 0; i < 10; i++ {
		all = append(all, Scroll{
			Name:        "scroll",
			Description: "deploy gpu server vastai",
			Tags:        []string{"deploy", "gpu"},
		})
	}

	results := Match(all, "deploy gpu server", 2)
	if len(results) > 2 {
		t.Errorf("expected at most 2 results, got %d", len(results))
	}
}
