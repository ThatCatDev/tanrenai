package models

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

// DownloadProgress is called periodically during a download.
type DownloadProgress func(downloaded, total int64)

// Download downloads a GGUF file from HuggingFace.
// url should be a direct download URL like:
//
//	https://huggingface.co/<repo>/resolve/main/<filename>.gguf
func Download(url, destDir string, progress DownloadProgress) (string, error) {
	// Extract filename from URL
	parts := strings.Split(url, "/")
	filename := parts[len(parts)-1]
	if !strings.HasSuffix(strings.ToLower(filename), ".gguf") {
		return "", fmt.Errorf("URL does not point to a .gguf file: %s", filename)
	}

	destPath := filepath.Join(destDir, filename)

	// Check for partial download (resume support)
	var startByte int64
	if info, err := os.Stat(destPath + ".partial"); err == nil {
		startByte = info.Size()
	}

	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}

	if startByte > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", startByte))
	}

	// Use HF token if available
	if token := os.Getenv("HF_TOKEN"); token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("download request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		return "", fmt.Errorf("download failed with status %d", resp.StatusCode)
	}

	totalSize := resp.ContentLength + startByte

	// Open file for writing (append if resuming)
	flags := os.O_CREATE | os.O_WRONLY
	if startByte > 0 {
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
	}

	partialPath := destPath + ".partial"
	f, err := os.OpenFile(partialPath, flags, 0644)
	if err != nil {
		return "", fmt.Errorf("open file: %w", err)
	}
	defer f.Close()

	// Copy with progress tracking
	buf := make([]byte, 32*1024)
	downloaded := startByte

	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, writeErr := f.Write(buf[:n]); writeErr != nil {
				return "", fmt.Errorf("write file: %w", writeErr)
			}
			downloaded += int64(n)
			if progress != nil {
				progress(downloaded, totalSize)
			}
		}
		if readErr != nil {
			if readErr == io.EOF {
				break
			}
			return "", fmt.Errorf("read body: %w", readErr)
		}
	}

	// Rename partial to final
	f.Close()
	if err := os.Rename(partialPath, destPath); err != nil {
		return "", fmt.Errorf("rename file: %w", err)
	}

	return destPath, nil
}
