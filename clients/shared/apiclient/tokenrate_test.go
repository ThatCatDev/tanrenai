package apiclient

import (
	"sync"
	"testing"
	"time"
)

// fakeClock returns timestamps from a fixed slice, advancing one call at a
// time. Lets tests pin the elapsed-time arithmetic without sleeping.
type fakeClock struct {
	mu    sync.Mutex
	times []time.Time
	idx   int
}

func (f *fakeClock) Now() time.Time {
	f.mu.Lock()
	defer f.mu.Unlock()
	t := f.times[f.idx]
	f.idx++
	return t
}

func TestTokenRateTracker_ZeroBeforeSecondToken(t *testing.T) {
	tr := &TokenRateTracker{}
	tokens, tps := tr.Snapshot()
	if tokens != 0 || tps != 0 {
		t.Errorf("empty tracker: got tokens=%d tps=%v, want 0/0", tokens, tps)
	}

	// Single recorded token: count rises but rate stays 0 — one sample is
	// not enough to compute a rate. The TUI uses this to decide whether to
	// render the t/s field at all on the first-token frame.
	tr.Record()
	tokens, tps = tr.Snapshot()
	if tokens != 1 {
		t.Errorf("after one record: tokens = %d, want 1", tokens)
	}
	if tps != 0 {
		t.Errorf("after one record: tps = %v, want 0 (need ≥2 tokens for a rate)", tps)
	}
}

func TestTokenRateTracker_SuppressesSubMillisecondWindow(t *testing.T) {
	// Two tokens 100µs apart — typical for very short responses where
	// llama-server flushes both deltas in the same network packet. The
	// rate would be 10000 t/s if we computed it naively, which is
	// meaningless to display. Snapshot must report 0 tps until the
	// elapsed window exceeds minRateWindow (100ms).
	start := time.Unix(0, 0)
	clock := &fakeClock{times: []time.Time{
		start,
		start.Add(100 * time.Microsecond),
	}}
	tr := &TokenRateTracker{now: clock.Now}
	tr.Record()
	tr.Record()
	tokens, tps := tr.Snapshot()
	if tokens != 2 {
		t.Errorf("tokens = %d, want 2", tokens)
	}
	if tps != 0 {
		t.Errorf("tps = %v, want 0 (window too short to measure reliably)", tps)
	}
}

func TestTokenRateTracker_RateExcludesFirstTokenLatency(t *testing.T) {
	// First token at t=1.0s (prompt-eval finished), then four more tokens
	// 0.1s apart. We want the rate to reflect generation only — i.e.
	// 4 tokens / 0.4 elapsed = 10 t/s — not 5 / 1.4 ≈ 3.6.
	start := time.Unix(0, 0)
	clock := &fakeClock{times: []time.Time{
		start.Add(1000 * time.Millisecond),
		start.Add(1100 * time.Millisecond),
		start.Add(1200 * time.Millisecond),
		start.Add(1300 * time.Millisecond),
		start.Add(1400 * time.Millisecond),
	}}
	tr := &TokenRateTracker{now: clock.Now}
	for range 5 {
		tr.Record()
	}
	tokens, tps := tr.Snapshot()
	if tokens != 5 {
		t.Errorf("tokens = %d, want 5", tokens)
	}
	if got := round1(tps); got != 10.0 {
		t.Errorf("tps = %v, want 10.0 (4 tokens over 0.4s — excludes prompt-eval latency)", got)
	}
}

func TestTokenRateTracker_Reset(t *testing.T) {
	clock := &fakeClock{times: []time.Time{
		time.Unix(0, 0),
		time.Unix(1, 0),
		time.Unix(2, 0),
	}}
	tr := &TokenRateTracker{now: clock.Now}
	tr.Record()
	tr.Record()
	if tokens, _ := tr.Snapshot(); tokens != 2 {
		t.Fatalf("pre-reset tokens = %d, want 2", tokens)
	}

	tr.Reset()
	tokens, tps := tr.Snapshot()
	if tokens != 0 || tps != 0 {
		t.Errorf("after reset: tokens=%d tps=%v, want 0/0", tokens, tps)
	}

	// Reset must rearm the first-token stamp so a new generation isn't
	// charged with the previous one's wall clock — otherwise back-to-back
	// turns would show absurd t/s spikes.
	tr.Record()
	if !tr.firstToken.Equal(time.Unix(2, 0)) {
		t.Errorf("post-reset firstToken = %v, want %v", tr.firstToken, time.Unix(2, 0))
	}
}

func TestTokenRateTracker_NilSafe(t *testing.T) {
	// The TUI may hold a nil pointer briefly during startup; calls must
	// not panic so the renderer doesn't crash on the first frame.
	var tr *TokenRateTracker
	tr.Record()
	tr.Reset()
	tokens, tps := tr.Snapshot()
	if tokens != 0 || tps != 0 {
		t.Errorf("nil tracker: got tokens=%d tps=%v, want 0/0", tokens, tps)
	}
}

func round1(f float64) float64 {
	return float64(int(f*10+0.5)) / 10
}
