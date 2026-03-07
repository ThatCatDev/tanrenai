package apiclient

import (
	"errors"
	"fmt"
)

var ErrServerUnavailable = errors.New("server unavailable")

type StatusError struct {
	Code int
	Body string
}

func (e *StatusError) Error() string {
	return fmt.Sprintf("server returned status %d: %s", e.Code, e.Body)
}
