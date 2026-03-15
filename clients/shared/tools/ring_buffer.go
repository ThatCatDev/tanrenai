package tools

import "sync"

// RingBuffer is a thread-safe fixed-size circular buffer implementing io.Writer.
type RingBuffer struct {
	mu   sync.Mutex
	buf  []byte
	size int
	pos  int
	full bool
}

// NewRingBuffer creates a ring buffer with the given capacity in bytes.
func NewRingBuffer(size int) *RingBuffer {
	return &RingBuffer{
		buf:  make([]byte, size),
		size: size,
	}
}

// Write appends data to the ring buffer, wrapping around when full.
func (r *RingBuffer) Write(p []byte) (int, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	n := len(p)
	if n >= r.size {
		// Data larger than buffer — keep only the last `size` bytes.
		copy(r.buf, p[n-r.size:])
		r.pos = 0
		r.full = true
		return n, nil
	}
	end := r.pos + n
	if end <= r.size {
		copy(r.buf[r.pos:], p)
	} else {
		first := r.size - r.pos
		copy(r.buf[r.pos:], p[:first])
		copy(r.buf, p[first:])
	}
	r.pos = end % r.size
	if end >= r.size {
		r.full = true
	}
	return n, nil
}

// String returns the buffer contents in chronological order.
func (r *RingBuffer) String() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.full {
		return string(r.buf[:r.pos])
	}
	out := make([]byte, r.size)
	n := copy(out, r.buf[r.pos:])
	copy(out[n:], r.buf[:r.pos])
	return string(out)
}
