package store

import (
	"os"
	"path/filepath"
	"strings"
)

// EnsureDir creates a directory if it doesn't exist
func EnsureDir(dir string) error {
	return os.MkdirAll(dir, 0755)
}

// WriteFile writes content to a file, creating parent directories
func WriteFile(path, content string) error {
	if err := EnsureDir(filepath.Dir(path)); err != nil {
		return err
	}
	return os.WriteFile(path, []byte(content), 0644)
}

// ReadFile reads content from a file
func ReadFile(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// FileExists checks if a file exists
func FileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// OpenAppend opens a file for appending, creating if needed
func OpenAppend(path string) (*os.File, error) {
	if err := EnsureDir(filepath.Dir(path)); err != nil {
		return nil, err
	}
	return os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
}

// SafeFileName sanitizes a file path for use as a filename component
func SafeFileName(filePath string) string {
	s := filePath
	s = strings.ReplaceAll(s, "\\", "_")
	s = strings.ReplaceAll(s, "/", "_")
	s = strings.ReplaceAll(s, ":", "_")
	s = strings.TrimLeft(s, "_")
	return s
}
