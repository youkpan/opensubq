package service

import (
	"fmt"
	"strconv"
	"strings"

	"file-chat/model"
	"file-chat/store"
)

// ReadOutline reads and parses the outline file for a job
func ReadOutline(outlinePath string) (*model.Outline, error) {
	if !store.FileExists(outlinePath) {
		return &model.Outline{}, nil
	}
	content, err := store.ReadFile(outlinePath)
	if err != nil {
		return nil, fmt.Errorf("read outline: %w", err)
	}
	return model.ParseOutline(content), nil
}

// AppendChunks appends chunk records to the outline file
func AppendChunks(outlinePath string, chunks []model.Chunk) error {
	var sb strings.Builder
	for _, c := range chunks {
		sb.WriteString(fmt.Sprintf("%s|%s|%s|%d|%d\n", c.ID, c.FilePath, c.Summary, c.StartLine, c.EndLine))
	}
	f, err := store.OpenAppend(outlinePath)
	if err != nil {
		return fmt.Errorf("open outline for append: %w", err)
	}
	defer f.Close()
	_, err = f.WriteString(sb.String())
	return err
}

// NextChunkID generates the next chunk ID based on existing chunks
func NextChunkID(chunks []model.Chunk) string {
	maxID := 0
	for _, c := range chunks {
		num, err := strconv.Atoi(strings.TrimPrefix(c.ID, "chunk_"))
		if err == nil && num > maxID {
			maxID = num
		}
	}
	return fmt.Sprintf("chunk_%03d", maxID+1)
}

// GetProcessedFiles returns the set of file paths already in the outline
func GetProcessedFiles(outline *model.Outline) map[string]bool {
	files := make(map[string]bool)
	for _, c := range outline.Chunks {
		files[c.FilePath] = true
	}
	return files
}

// WritePerFileOutline writes chunks for a single file to its own outline file
func WritePerFileOutline(outlinesDir string, chunks []model.Chunk) error {
	if len(chunks) == 0 {
		return nil
	}
	filePath := chunks[0].FilePath
	path := store.GetPerFileOutlinePath(outlinesDir, filePath)
	var sb strings.Builder
	for _, c := range chunks {
		sb.WriteString(fmt.Sprintf("%s|%s|%s|%d|%d\n", c.ID, c.FilePath, c.Summary, c.StartLine, c.EndLine))
	}
	return store.WriteFile(path, sb.String())
}

// ReadPerFileOutline reads the outline for a specific file
func ReadPerFileOutline(outlinesDir, filePath string) (*model.Outline, error) {
	path := store.GetPerFileOutlinePath(outlinesDir, filePath)
	return ReadOutline(path)
}

// RemoveChunksForFile removes all chunks for a file from the global outline
func RemoveChunksForFile(outlinePath, filePath string, outline *model.Outline) error {
	var remaining []model.Chunk
	for _, c := range outline.Chunks {
		if c.FilePath != filePath {
			remaining = append(remaining, c)
		}
	}
	outline.Chunks = remaining
	return store.WriteFile(outlinePath, outline.String())
}
