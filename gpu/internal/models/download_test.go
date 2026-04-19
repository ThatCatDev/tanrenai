package models

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDownload_NonGGUFURL(t *testing.T) {
	_, err := Download("http://example.com/model.bin", t.TempDir(), nil)
	if err == nil {
		t.Fatal("expected error for non-.gguf URL")
	}
}

func TestDownload_HappyPath(t *testing.T) {
	content := []byte("fake gguf content for testing")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(content)
	}))
	defer srv.Close()

	destDir := t.TempDir()
	url := srv.URL + "/model.gguf"

	var lastDownloaded, lastTotal int64
	progress := func(downloaded, total int64) {
		lastDownloaded = downloaded
		lastTotal = total
	}

	path, err := Download(url, destDir, progress)
	if err != nil {
		t.Fatalf("Download error: %v", err)
	}

	if path != filepath.Join(destDir, "model.gguf") {
		t.Errorf("path = %q, want %q", path, filepath.Join(destDir, "model.gguf"))
	}

	// File should exist and have the correct content
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if string(got) != string(content) {
		t.Errorf("file content = %q, want %q", string(got), string(content))
	}

	// Partial file should be cleaned up
	if _, err := os.Stat(path + ".partial"); !os.IsNotExist(err) {
		t.Error(".partial file should be removed after successful download")
	}

	// Progress should have been called
	if lastDownloaded == 0 {
		t.Error("progress callback was never called or downloaded=0")
	}
	_ = lastTotal
}

func TestDownload_NilProgress(t *testing.T) {
	content := []byte("small gguf data")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(content)
	}))
	defer srv.Close()

	destDir := t.TempDir()
	url := srv.URL + "/model.gguf"

	// progress=nil should not panic
	path, err := Download(url, destDir, nil)
	if err != nil {
		t.Fatalf("Download error: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Errorf("output file not found: %v", err)
	}
}

func TestDownload_ServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	destDir := t.TempDir()
	_, err := Download(srv.URL+"/model.gguf", destDir, nil)
	if err == nil {
		t.Fatal("expected error for server 500")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("error should mention status 500: %v", err)
	}
}

func TestDownload_PartialResume(t *testing.T) {
	// Simulate a partial download that resumes with 206 Partial Content
	fullContent := []byte("0123456789abcdefghij gguf file content here")
	existingBytes := fullContent[:10]
	remainingBytes := fullContent[10:]

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rangeHeader := r.Header.Get("Range")
		if rangeHeader == "bytes=10-" {
			w.WriteHeader(http.StatusPartialContent)
			_, _ = w.Write(remainingBytes)
		} else {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(fullContent)
		}
	}))
	defer srv.Close()

	destDir := t.TempDir()
	url := srv.URL + "/model.gguf"

	// Pre-create a .partial file simulating a partial download
	partialPath := filepath.Join(destDir, "model.gguf.partial")
	if err := os.WriteFile(partialPath, existingBytes, 0644); err != nil {
		t.Fatalf("setup partial file: %v", err)
	}

	path, err := Download(url, destDir, nil)
	if err != nil {
		t.Fatalf("Download error: %v", err)
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if string(got) != string(fullContent) {
		t.Errorf("resumed content = %q, want %q", string(got), string(fullContent))
	}
}

func TestDownload_PresignedURLWithQuery(t *testing.T) {
	// Presigned S3/R2 URLs have a long ?X-Amz-... query string. The
	// filename extraction must ignore it, otherwise the .gguf suffix
	// check rejects the URL even though the underlying object is valid.
	content := []byte("gguf bytes from r2")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(content)
	}))
	defer srv.Close()

	destDir := t.TempDir()
	// Mirror what modelcache.Cache.Lookup returns.
	url := srv.URL + "/models/unsloth/Qwen-GGUF/Q4_K_M/Qwen-Q4_K_M-00001-of-00003.gguf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=deadbeef"

	path, err := Download(url, destDir, nil)
	if err != nil {
		t.Fatalf("Download error on presigned URL: %v", err)
	}
	if !strings.HasSuffix(path, "Qwen-Q4_K_M-00001-of-00003.gguf") {
		t.Errorf("path should land without the query string: got %q", path)
	}
	if _, err := os.Stat(path); err != nil {
		t.Errorf("file missing after successful download: %v", err)
	}
}

func TestDownload_NetworkError(t *testing.T) {
	// Use a URL that will fail to connect
	_, err := Download("http://127.0.0.1:1/model.gguf", t.TempDir(), nil)
	if err == nil {
		t.Fatal("expected network error")
	}
}

func TestDownload_HFProvenance(t *testing.T) {
	content := []byte("gguf content")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Simulate HuggingFace URL pattern:
		// /Qwen/Qwen2.5-7B/resolve/main/model.gguf
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(content)
	}))
	defer srv.Close()

	destDir := t.TempDir()

	// Use a real HF-style URL so ParseHFURL returns true, but point at test server
	// We can't fully test HF provenance without a real HF URL, but we can test
	// that a non-HF URL doesn't error.
	url := srv.URL + "/model.gguf"
	path, err := Download(url, destDir, nil)
	if err != nil {
		t.Fatalf("Download error: %v", err)
	}
	if path == "" {
		t.Error("expected non-empty path")
	}
}

func TestDownload_LargeContent_ProgressTracking(t *testing.T) {
	// Generate content larger than the 32KB read buffer
	content := make([]byte, 100*1024) // 100KB
	for i := range content {
		content[i] = byte(i % 256)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "102400")
		w.WriteHeader(http.StatusOK)
		_, _ = io.Copy(w, strings.NewReader(string(content)))
	}))
	defer srv.Close()

	destDir := t.TempDir()
	url := srv.URL + "/bigmodel.gguf"

	callCount := 0
	progress := func(downloaded, total int64) {
		callCount++
	}

	path, err := Download(url, destDir, progress)
	if err != nil {
		t.Fatalf("Download error: %v", err)
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if len(got) != len(content) {
		t.Errorf("downloaded %d bytes, want %d", len(got), len(content))
	}
	if callCount == 0 {
		t.Error("progress callback never called")
	}
}
