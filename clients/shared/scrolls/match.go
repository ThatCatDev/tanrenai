package scrolls

import (
	"sort"
	"strings"
)

// Common English stop words to filter out during matching.
var stopWords = map[string]bool{
	"a": true, "an": true, "the": true, "is": true, "are": true,
	"was": true, "were": true, "be": true, "been": true, "being": true,
	"have": true, "has": true, "had": true, "do": true, "does": true,
	"did": true, "will": true, "would": true, "could": true, "should": true,
	"may": true, "might": true, "can": true, "shall": true,
	"i": true, "me": true, "my": true, "we": true, "our": true,
	"you": true, "your": true, "he": true, "she": true, "it": true,
	"they": true, "them": true, "their": true, "its": true,
	"this": true, "that": true, "these": true, "those": true,
	"in": true, "on": true, "at": true, "to": true, "for": true,
	"of": true, "with": true, "by": true, "from": true, "as": true,
	"into": true, "about": true, "between": true, "through": true,
	"and": true, "or": true, "but": true, "not": true, "no": true,
	"if": true, "then": true, "so": true, "than": true,
	"what": true, "how": true, "when": true, "where": true, "who": true,
	"which": true, "there": true, "here": true,
}

// Match scores each scroll against the input and returns the top matches.
func Match(scrolls []Scroll, input string, maxResults int) []Scroll {
	tokens := tokenize(input)
	if len(tokens) == 0 {
		return nil
	}

	type scored struct {
		scroll Scroll
		score  int
	}

	var results []scored
	for _, s := range scrolls {
		score := scoreScroll(s, tokens)
		if score >= 2 {
			results = append(results, scored{s, score})
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})

	if len(results) > maxResults {
		results = results[:maxResults]
	}

	out := make([]Scroll, len(results))
	for i, r := range results {
		out[i] = r.scroll
	}

	return out
}

func scoreScroll(s Scroll, tokens []string) int {
	score := 0

	tagSet := make(map[string]bool, len(s.Tags))
	for _, t := range s.Tags {
		tagSet[strings.ToLower(t)] = true
	}

	nameWords := make(map[string]bool)
	for _, w := range splitIdent(s.Name) {
		nameWords[strings.ToLower(w)] = true
	}

	descWords := make(map[string]bool)
	for _, w := range tokenize(s.Description) {
		descWords[w] = true
	}

	for _, tok := range tokens {
		if tagSet[tok] {
			score += 3
		}
		if nameWords[tok] {
			score += 1
		}
		if descWords[tok] {
			score += 1
		}
	}

	return score
}

// tokenize lowercases the input, splits on whitespace/punctuation, and removes stop words.
func tokenize(s string) []string {
	s = strings.ToLower(s)
	// Split on anything that's not a letter, digit, or hyphen.
	fields := strings.FieldsFunc(s, func(r rune) bool {
		return !isWordChar(r)
	})

	var tokens []string
	for _, f := range fields {
		if !stopWords[f] && len(f) > 1 {
			tokens = append(tokens, f)
		}
	}

	return tokens
}

// splitIdent splits an identifier like "deploy-gpu-server" into words.
func splitIdent(s string) []string {
	return strings.FieldsFunc(s, func(r rune) bool {
		return r == '-' || r == '_' || r == '.'
	})
}

func isWordChar(r rune) bool {
	return (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-'
}
