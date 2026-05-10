// Package models owns client-side parsing of model identifiers — turning a
// bare GGUF name like `Qwen3.6-35B-A3B-Q4_K_M` into a pullable hf:// URI.
//
// Kept in sync with tanrenai-platform's internal/models/resolve.go so the
// CLI's --local path and the platform's auto-pull path agree on what a
// bare name means. Don't drift them; users would get different behavior
// from `tanrenai pull --local <name>` vs the same name through the hosted
// platform.
package models

import (
	"fmt"
	"regexp"
	"strings"
)

// unslothQuantSuffix matches a trailing GGUF quant suffix using the Unsloth
// naming convention. Captures the root (pre-quant part) and quant (with any
// `UD-` dynamic prefix kept so it can be passed verbatim in the hf:// URI).
//
//	Qwen3.5-122B-A10B-UD-Q4_K_XL   -> root=Qwen3.5-122B-A10B  quant=UD-Q4_K_XL
//	Qwen2.5-7B-Instruct-Q4_K_M     -> root=Qwen2.5-7B-Instruct quant=Q4_K_M
//	gemma-2-27b-it-BF16            -> root=gemma-2-27b-it      quant=BF16
var unslothQuantSuffix = regexp.MustCompile(`(?i)^(.+?)-((?:UD-)?(?:I?Q\d+(?:_\w+)*|F16|BF16|F32))$`)

// IsURI reports whether the input is already a pullable URI/URL rather
// than a bare model name needing resolution.
func IsURI(s string) bool {
	return strings.HasPrefix(s, "hf://") ||
		strings.HasPrefix(s, "https://") ||
		strings.HasPrefix(s, "http://")
}

// ResolveBareNameToURI guesses an hf:// URI for a bare model name using the
// Unsloth GGUF repo convention `<name>-GGUF/<quant>`. URIs/URLs pass through
// unchanged. Returns "" when the input has no recognizable quant suffix —
// callers should surface that as an error so the user gets a useful message
// instead of a silent misroute.
func ResolveBareNameToURI(name string) string {
	if IsURI(name) {
		return name
	}
	m := unslothQuantSuffix.FindStringSubmatch(name)
	if m == nil {
		return ""
	}
	root, quant := m[1], m[2]
	return fmt.Sprintf("hf://unsloth/%s-GGUF/%s", root, quant)
}
