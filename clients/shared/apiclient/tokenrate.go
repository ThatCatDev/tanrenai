package apiclient

import (
	"sync"
	"time"
)

// TokenRateTracker measures generation throughput during a streaming
// completion. Callers Record() once per streamed content delta (each
// non-empty `Delta.Content` chunk from llama-server's SSE is approximately
// one token, which is the granularity llama.cpp users expect when comparing
// to its own `eval rate` reporting). The first Record() captures the
// stream start time, so subsequent rate calculations exclude prompt-eval
// latency and reflect just the generation phase.
//
// Safe for concurrent Record/Snapshot calls from a stream goroutine and a
// UI redraw goroutine — the TUI updates the status bar from a different
// thread than the stream reader.
type TokenRateTracker struct {
	mu         sync.Mutex
	firstToken time.Time
	lastToken  time.Time
	count      int
	now        func() time.Time // injectable for tests; nil ⇒ time.Now
}

// Record counts one more streamed token. The first call also stamps the
// generation start time; subsequent calls extend the end time. Safe to call
// from the stream-read goroutine.
func (t *TokenRateTracker) Record() {
	if t == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	now := t.timeNow()
	if t.firstToken.IsZero() {
		t.firstToken = now
	}
	t.lastToken = now
	t.count++
}

// Reset clears all state. Call at the start of each new generation so the
// rate isn't averaged across previous turns.
func (t *TokenRateTracker) Reset() {
	if t == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.firstToken = time.Time{}
	t.lastToken = time.Time{}
	t.count = 0
}

// minRateWindow is the minimum elapsed time between first and last tokens
// before a rate is considered meaningful. With shorter windows the math is
// dominated by SSE flush jitter (sub-millisecond intervals from llama-server
// produce thousands-of-t/s readings that mean nothing).
const minRateWindow = 100 * time.Millisecond

// Snapshot returns the current (tokens, tokens-per-second) pair. tps is 0
// until at least two tokens have arrived AND at least minRateWindow has
// elapsed — short responses (a 2-token "Hi!" delivered in microseconds)
// would otherwise produce nonsense rates like 80000 t/s. Callers should
// suppress display of the rate when tps == 0.
func (t *TokenRateTracker) Snapshot() (tokens int, tps float64) {
	if t == nil {
		return 0, 0
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.count < 2 || t.firstToken.IsZero() {
		return t.count, 0
	}
	elapsed := t.lastToken.Sub(t.firstToken)
	if elapsed < minRateWindow {
		return t.count, 0
	}
	// (count - 1) because we measure the interval between the first and
	// last tokens; the first token's wall clock is the start of the
	// window, not a unit of work done within it.
	return t.count, float64(t.count-1) / elapsed.Seconds()
}

func (t *TokenRateTracker) timeNow() time.Time {
	if t.now != nil {
		return t.now()
	}
	return time.Now()
}
