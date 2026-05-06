package store

import (
	"fmt"
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

// JobPaths returns standard paths for a job
type JobPaths struct {
	Root         string // jobs/{id}
	Outline      string // jobs/{id}/outline
	Sources      string // jobs/{id}/sources
	Chunks       string // jobs/{id}/chunks
	OutlinesDir  string // jobs/{id}/outlines
	FilesSummary string // jobs/{id}/files_summary.xml
}

// GetJobPaths returns paths for a given job ID
func GetJobPaths(jobsDir, jobID string) JobPaths {
	root := filepath.Join(jobsDir, jobID)
	return JobPaths{
		Root:         root,
		Outline:      filepath.Join(root, "outline"),
		Sources:      filepath.Join(root, "sources"),
		Chunks:       filepath.Join(root, "chunks"),
		OutlinesDir:  filepath.Join(root, "outlines"),
		FilesSummary: filepath.Join(root, "files_summary.xml"),
	}
}

// InitJob creates the job directory structure
func InitJob(jobsDir, jobID string) (*JobPaths, error) {
	jp := GetJobPaths(jobsDir, jobID)
	if err := EnsureDir(jp.Sources); err != nil {
		return nil, fmt.Errorf("create sources dir: %w", err)
	}
	if err := EnsureDir(jp.Chunks); err != nil {
		return nil, fmt.Errorf("create chunks dir: %w", err)
	}
	if err := EnsureDir(jp.OutlinesDir); err != nil {
		return nil, fmt.Errorf("create outlines dir: %w", err)
	}
	return &jp, nil
}

// ListExistingDirs returns relative directory paths under the chunks directory
func ListExistingDirs(chunksDir string) []string {
	var dirs []string
	filepath.Walk(chunksDir, func(path string, info os.FileInfo, err error) error {
		if err != nil || !info.IsDir() || path == chunksDir {
			return nil
		}
		rel, _ := filepath.Rel(chunksDir, path)
		dirs = append(dirs, rel)
		return nil
	})
	return dirs
}

// SafeFileName sanitizes a file path for use as a filename component
func SafeFileName(filePath string) string {
	s := filePath
	// Replace both types of separators
	s = strings.ReplaceAll(s, "\\", "_")
	s = strings.ReplaceAll(s, "/", "_")
	// Remove drive letter colon (Windows)
	s = strings.ReplaceAll(s, ":", "_")
	s = strings.TrimLeft(s, "_")
	return s
}

// GetChunkFilePath returns the file path for a chunk
func GetChunkFilePath(chunksDir, filePath, chunkID string) string {
	return filepath.Join(chunksDir, SafeFileName(filePath)+"_"+chunkID)
}

// GetSourceFilePath returns the path for storing converted source text
func GetSourceFilePath(sourcesDir, originalPath string) string {
	ext := filepath.Ext(originalPath)
	safeName := SafeFileName(originalPath)
	return filepath.Join(sourcesDir, strings.TrimSuffix(safeName, ext)+".txt")
}

// GetPerFileOutlinePath returns the outline file path for a specific file
func GetPerFileOutlinePath(outlinesDir, filePath string) string {
	return filepath.Join(outlinesDir, SafeFileName(filePath))
}
