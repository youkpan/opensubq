package store

import (
	"fmt"
	"os"

	"file-chat/model"
)

// ReadChunksJSON reads and parses chunks.json for a file
func ReadChunksJSON(dp *DataPaths, filePath string) (*model.ChunksFile, error) {
	path := dp.GetChunksJSONPath(filePath)
	if !FileExists(path) {
		return &model.ChunksFile{FilePath: filePath}, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read chunks.json: %w", err)
	}
	return model.ParseChunksFile(data)
}

// WriteChunksJSON serializes and writes chunks.json for a file
func WriteChunksJSON(dp *DataPaths, filePath string, cf *model.ChunksFile) error {
	data, err := model.SerializeChunksFile(cf)
	if err != nil {
		return fmt.Errorf("serialize chunks.json: %w", err)
	}
	return WriteFile(dp.GetChunksJSONPath(filePath), string(data))
}
