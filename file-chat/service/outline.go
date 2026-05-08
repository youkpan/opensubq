package service

import (
	"fmt"
	"strings"

	"file-chat/model"
	"file-chat/store"
)

// ReadOutline reads and parses an outline file (2-field format)
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

// BuildGlobalOutline reads all per-file outlines and assembles them into one Outline
func BuildGlobalOutline(dp *store.DataPaths, registry *model.FileRegistry) (*model.Outline, error) {
	var allChunks []model.Chunk
	for filePath := range registry.Files {
		outline, err := ReadPerFileOutline(dp, filePath)
		if err != nil {
			continue
		}
		// Set FilePath on each chunk (outline only stores chunk_id|summary)
		for i := range outline.Chunks {
			outline.Chunks[i].FilePath = filePath
		}
		allChunks = append(allChunks, outline.Chunks...)
	}
	return &model.Outline{Chunks: allChunks}, nil
}

// NextChunkID generates the next chunk ID based on existing chunks
func NextChunkID(chunks []model.Chunk) string {
	return fmt.Sprintf("chunk_%03d", nextChunkNum(chunks))
}

// GetProcessedFiles returns the set of file paths already in the outline
func GetProcessedFiles(outline *model.Outline) map[string]bool {
	files := make(map[string]bool)
	for _, c := range outline.Chunks {
		files[c.FilePath] = true
	}
	return files
}

// WritePerFileOutline writes chunks for a single file to its outline file (2-field format)
func WritePerFileOutline(dp *store.DataPaths, chunks []model.Chunk) error {
	if len(chunks) == 0 {
		return nil
	}
	filePath := chunks[0].FilePath
	path := dp.GetFileOutlinePath(filePath)
	var sb strings.Builder
	for _, c := range chunks {
		fmt.Fprintf(&sb, "%s|%s\n", c.ID, c.Summary)
	}
	return store.WriteFile(path, sb.String())
}

// ReadPerFileOutline reads the outline for a specific file
func ReadPerFileOutline(dp *store.DataPaths, filePath string) (*model.Outline, error) {
	path := dp.GetFileOutlinePath(filePath)
	return ReadOutline(path)
}

// RemoveChunksForFile removes all data for a file (just deletes the file dir)
func RemoveChunksForFile(dp *store.DataPaths, filePath string, outline *model.Outline) error {
	return dp.DeleteFileDir(filePath)
}
