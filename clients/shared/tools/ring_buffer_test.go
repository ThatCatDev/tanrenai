package tools

import (
	"testing"
)

func TestRingBuffer_BasicWrite(t *testing.T) {
	rb := NewRingBuffer(16)
	rb.Write([]byte("hello"))
	if got := rb.String(); got != "hello" {
		t.Errorf("got %q, want %q", got, "hello")
	}
}

func TestRingBuffer_WrapAround(t *testing.T) {
	rb := NewRingBuffer(8)
	rb.Write([]byte("12345"))
	rb.Write([]byte("67890"))
	got := rb.String()
	// Buffer is 8 bytes; after writing 10 bytes, oldest 2 are lost
	if got != "34567890" {
		t.Errorf("got %q, want %q", got, "34567890")
	}
}

func TestRingBuffer_ExactFit(t *testing.T) {
	rb := NewRingBuffer(5)
	rb.Write([]byte("abcde"))
	if got := rb.String(); got != "abcde" {
		t.Errorf("got %q, want %q", got, "abcde")
	}
}

func TestRingBuffer_LargerThanBuffer(t *testing.T) {
	rb := NewRingBuffer(4)
	rb.Write([]byte("abcdefgh"))
	// Only last 4 bytes should be kept
	if got := rb.String(); got != "efgh" {
		t.Errorf("got %q, want %q", got, "efgh")
	}
}

func TestRingBuffer_Empty(t *testing.T) {
	rb := NewRingBuffer(8)
	if got := rb.String(); got != "" {
		t.Errorf("got %q, want empty string", got)
	}
}

func TestRingBuffer_MultipleSmallWrites(t *testing.T) {
	rb := NewRingBuffer(6)
	rb.Write([]byte("ab"))
	rb.Write([]byte("cd"))
	rb.Write([]byte("ef"))
	if got := rb.String(); got != "abcdef" {
		t.Errorf("got %q, want %q", got, "abcdef")
	}
	// One more write should wrap
	rb.Write([]byte("gh"))
	if got := rb.String(); got != "cdefgh" {
		t.Errorf("got %q, want %q", got, "cdefgh")
	}
}
