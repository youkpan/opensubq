package store

import (
	"crypto/md5"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"file-chat/model"
)

// ReadFileRegistry reads files.json, returns empty registry if not exists
func ReadFileRegistry(jobsDir string) (*model.FileRegistry, error) {
	path := filepath.Join(jobsDir, "files.json")
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return &model.FileRegistry{Files: make(map[string]*model.FileEntry)}, nil
		}
		return nil, fmt.Errorf("read files.json: %w", err)
	}
	var reg model.FileRegistry
	if err := json.Unmarshal(data, &reg); err != nil {
		return nil, fmt.Errorf("parse files.json: %w", err)
	}
	if reg.Files == nil {
		reg.Files = make(map[string]*model.FileEntry)
	}
	return &reg, nil
}

// WriteFileRegistry writes files.json
func WriteFileRegistry(jobsDir string, reg *model.FileRegistry) error {
	path := filepath.Join(jobsDir, "files.json")
	data, err := json.MarshalIndent(reg, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal files.json: %w", err)
	}
	return os.WriteFile(path, data, 0644)
}

// ComputeFileHash computes MD5 hash of a file
func ComputeFileHash(filePath string) (string, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := md5.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return fmt.Sprintf("%x", h.Sum(nil)), nil
}

// GetFileModTime returns file modification time as RFC3339 string
func GetFileModTime(filePath string) (string, error) {
	info, err := os.Stat(filePath)
	if err != nil {
		return "", err
	}
	return info.ModTime().Format("2006-01-02T15:04:05Z07:00"), nil
}

// GetFileSize returns file size
func GetFileSize(filePath string) (int64, error) {
	info, err := os.Stat(filePath)
	if err != nil {
		return 0, err
	}
	return info.Size(), nil
}
