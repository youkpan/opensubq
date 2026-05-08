package store

import (
	"crypto/md5"
	"fmt"
	"os"
	"path/filepath"
)

// DataPaths manages all data storage paths under the data/ directory
type DataPaths struct {
	DataDir   string // data/
	FilesDir  string // data/files/
	ChatsDir  string // data/chats/
	GlobalDir string // data/global/
}

// NewDataPaths creates a new DataPaths instance
func NewDataPaths(dataDir string) *DataPaths {
	return &DataPaths{
		DataDir:   dataDir,
		FilesDir:  filepath.Join(dataDir, "files"),
		ChatsDir:  filepath.Join(dataDir, "chats"),
		GlobalDir: filepath.Join(dataDir, "global"),
	}
}

// InitDataDir creates the data directory structure and returns DataPaths
func InitDataDir(dataDir string) (*DataPaths, error) {
	dp := NewDataPaths(dataDir)
	for _, dir := range []string{dp.FilesDir, dp.ChatsDir, dp.GlobalDir} {
		if err := EnsureDir(dir); err != nil {
			return nil, fmt.Errorf("create data dir %s: %w", dir, err)
		}
	}
	return dp, nil
}

// pathHash computes MD5 hash of a file path (used for storage bucketing)
func pathHash(filePath string) string {
	h := md5.Sum([]byte(filePath))
	return fmt.Sprintf("%x", h)
}

// GetFileDir returns the storage directory for a file
// Format: data/files/{hash[:2]}/{hash[2:4]}/{SafeFileName}/
func (dp *DataPaths) GetFileDir(filePath string) string {
	h := pathHash(filePath)
	return filepath.Join(dp.FilesDir, h[:2], h[2:4], SafeFileName(filePath))
}

// GetFileChunksDir returns the chunks directory for a file
func (dp *DataPaths) GetFileChunksDir(filePath string) string {
	return filepath.Join(dp.GetFileDir(filePath), "chunks")
}

// GetFileOutlinePath returns the outline file path for a specific file
func (dp *DataPaths) GetFileOutlinePath(filePath string) string {
	return filepath.Join(dp.GetFileDir(filePath), "outline")
}

// GetFileSourcePath returns the source text path for a specific file
func (dp *DataPaths) GetFileSourcePath(filePath string) string {
	return filepath.Join(dp.GetFileDir(filePath), "source")
}

// GetGlobalOutlinePath returns the global outline path
func (dp *DataPaths) GetGlobalOutlinePath() string {
	return filepath.Join(dp.GlobalDir, "global_outline")
}

// GetGlobalSummaryPath returns the global files summary path
func (dp *DataPaths) GetGlobalSummaryPath() string {
	return filepath.Join(dp.GlobalDir, "global_files_summary.xml")
}

// GetChatFilesPath returns the chat files path for a conversation
func (dp *DataPaths) GetChatFilesPath(conversationID string) string {
	return filepath.Join(dp.ChatsDir, conversationID, "chat-files.json")
}

// GetRegistryPath returns the global registry path
func (dp *DataPaths) GetRegistryPath() string {
	return filepath.Join(dp.DataDir, "files.json")
}

// GetChunkFilePath returns the chunk file path for a specific file and chunk ID
func (dp *DataPaths) GetChunkFilePath(filePath, chunkID string) string {
	return filepath.Join(dp.GetFileChunksDir(filePath), chunkID)
}

// InitFileDir creates the directory structure for a file's storage
func (dp *DataPaths) InitFileDir(filePath string) error {
	return EnsureDir(dp.GetFileChunksDir(filePath))
}

// DeleteFileDir removes all stored data for a file
func (dp *DataPaths) DeleteFileDir(filePath string) error {
	return os.RemoveAll(dp.GetFileDir(filePath))
}
