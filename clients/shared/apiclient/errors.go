package apiclient

import (
	"encoding/json"
	"errors"
	"fmt"
)

var ErrServerUnavailable = errors.New("server unavailable")

type StatusError struct {
	Code int
	Body string
}

// Error renders the server's structured error when present, so callers (the CLI
// run flow, agent-rpc, the VSCode extension that surfaces it) get a readable
// message instead of a raw status dump. The plan gate gets an actionable hint.
// Code and Body remain available for callers that match via errors.As.
func (e *StatusError) Error() string {
	var p struct {
		Error   string `json:"error"`
		Message string `json:"message"`
	}
	if json.Unmarshal([]byte(e.Body), &p) == nil {
		if p.Error == "no_plan" {
			msg := p.Message
			if msg == "" {
				msg = "You don't have an active plan."
			}
			return msg + "  Redeem an invite code:  tanrenai activate <code>"
		}
		if p.Message != "" {
			return fmt.Sprintf("%s (status %d)", p.Message, e.Code)
		}
	}
	return fmt.Sprintf("server returned status %d: %s", e.Code, e.Body)
}
