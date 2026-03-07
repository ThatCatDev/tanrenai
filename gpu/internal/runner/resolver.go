package runner

import (
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/ThatCatDev/tanrenai/gpu/internal/gguf"
	"github.com/ThatCatDev/tanrenai/gpu/internal/models"
)

// TemplateResolution holds the result of automatic template detection.
type TemplateResolution struct {
	// TemplatePath is the path to the generated/fetched template file.
	TemplatePath string

	// Source describes where the template came from.
	// Examples: "generated:qwen2", "huggingface:Qwen/Qwen2.5-32B-GGUF", "none"
	Source string

	// Cleanup removes the temporary template file. May be nil.
	Cleanup func()
}

// ResolveTemplate attempts to auto-detect and generate a chat template for the
// given GGUF model file. Returns nil (no error) if no template could be determined.
//
// Resolution chain:
//  1. Read GGUF metadata → match architecture/name to known family → generate template
//  2. Load .meta.json sidecar → fetch from HuggingFace if repo info available
//  3. Return nil if no strategy works
func ResolveTemplate(modelPath string) (*TemplateResolution, error) {
	// Step 1: Try GGUF metadata → family match → generate.
	meta, err := gguf.ReadMetadata(modelPath)
	if err != nil {
		log.Printf("template resolver: could not read GGUF metadata: %v", err)
		// Non-fatal — continue to fallback strategies.
	} else if meta != nil && meta.General.Architecture != "" {
		if cfg := MatchFamily(meta.General.Architecture, meta.General.Name); cfg != nil {
			tpl := GenerateChatML(*cfg)
			name := sanitizeName(meta.General.Architecture)
			path, err := WriteTemplateFile(name, tpl)
			if err != nil {
				return nil, fmt.Errorf("template resolver: write generated template: %w", err)
			}
			return &TemplateResolution{
				TemplatePath: path,
				Source:       "generated:" + meta.General.Architecture,
				Cleanup:      func() { os.Remove(path) },
			}, nil
		}
	}

	// Step 2: Try HuggingFace via .meta.json sidecar.
	modelMeta, err := models.LoadMetadata(modelPath)
	if err != nil {
		log.Printf("template resolver: could not load model metadata: %v", err)
	}
	if modelMeta != nil && modelMeta.HFRepo != "" {
		branch := modelMeta.HFBranch
		if branch == "" {
			branch = "main"
		}
		hf := models.NewHFClient()
		tpl, err := hf.FetchChatTemplate(modelMeta.HFRepo, branch)
		if err != nil {
			log.Printf("template resolver: HuggingFace fetch failed: %v", err)
		} else if tpl != "" {
			name := sanitizeName(modelMeta.HFRepo)
			path, err := WriteTemplateFile(name, tpl)
			if err != nil {
				return nil, fmt.Errorf("template resolver: write HF template: %w", err)
			}
			return &TemplateResolution{
				TemplatePath: path,
				Source:       "huggingface:" + modelMeta.HFRepo,
				Cleanup:      func() { os.Remove(path) },
			}, nil
		}
	}

	// Step 3: No template found — return nil (caller uses whatever is in the GGUF).
	return nil, nil
}

// sanitizeName produces a safe filename component from an architecture or repo name.
func sanitizeName(s string) string {
	r := strings.NewReplacer("/", "-", " ", "-", ".", "-")
	return strings.ToLower(r.Replace(s))
}
