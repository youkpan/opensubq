package store

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"file-chat/model"
)

// ReadChatFiles reads the chat files list for a conversation
func ReadChatFiles(dp *DataPaths, conversationID string) (*model.ChatFiles, error) {
	path := dp.GetChatFilesPath(conversationID)
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return &model.ChatFiles{
				ConversationID: conversationID,
			}, nil
		}
		return nil, fmt.Errorf("read chat files: %w", err)
	}
	var cf model.ChatFiles
	if err := json.Unmarshal(data, &cf); err != nil {
		return nil, fmt.Errorf("parse chat files: %w", err)
	}
	return &cf, nil
}

// WriteChatFiles writes the chat files list for a conversation
func WriteChatFiles(dp *DataPaths, cf *model.ChatFiles) error {
	path := dp.GetChatFilesPath(cf.ConversationID)
	if err := EnsureDir(filepath.Dir(path)); err != nil {
		return err
	}
	data, err := json.MarshalIndent(cf, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal chat files: %w", err)
	}
	return os.WriteFile(path, data, 0644)
}

// AddFileToChat adds a file to the conversation's file list
func AddFileToChat(dp *DataPaths, conversationID, filePath string) error {
	cf, err := ReadChatFiles(dp, conversationID)
	if err != nil {
		return err
	}
	for _, f := range cf.Files {
		if f == filePath {
			return nil
		}
	}
	cf.Files = append(cf.Files, filePath)
	cf.UpdatedAt = time.Now().Format("2006-01-02T15:04:05Z07:00")
	return WriteChatFiles(dp, cf)
}

// AddFolderToChat adds a folder to the conversation's folder list
func AddFolderToChat(dp *DataPaths, conversationID, folderPath string) error {
	cf, err := ReadChatFiles(dp, conversationID)
	if err != nil {
		return err
	}
	for _, f := range cf.Folders {
		if f == folderPath {
			return nil
		}
	}
	cf.Folders = append(cf.Folders, folderPath)
	cf.UpdatedAt = time.Now().Format("2006-01-02T15:04:05Z07:00")
	return WriteChatFiles(dp, cf)
}
