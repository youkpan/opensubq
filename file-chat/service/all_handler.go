package service

import (
	"fmt"
	"os"
	"strings"

	"file-chat/llm"
	"file-chat/model"
	"file-chat/store"
)

const allFilesRetrievePrompt = `你是一个文档检索助手。以下是系统中所有已处理文件的摘要：

%s

请根据用户的问题，列出最相关的文件路径，每行一个，最多 %d 个。
只输出文件路径，不要输出其他内容。`

// HandleAllFiles handles @全部 requests with two-level retrieval:
// 1. files_summary.xml → LLM selects relevant files
// 2. per-file outline → LLM selects relevant chunks
func HandleAllFiles(client *llm.Client, jp *store.JobPaths, query string, maxRetrieve int) ([]model.Chunk, error) {
	// 1. Read files_summary.xml
	summaryContent := ReadFilesSummary(jp.FilesSummary)
	if summaryContent == "" {
		return nil, nil
	}

	// 2. LLM selects relevant files
	fileSelectPrompt := fmt.Sprintf(allFilesRetrievePrompt, summaryContent, 5)
	relevantFilesResult, err := client.ChatSimple(fileSelectPrompt, query)
	if err != nil {
		return nil, fmt.Errorf("select files: %w", err)
	}

	// 3. Parse relevant file paths
	filePaths := parseRelevantFiles(relevantFilesResult)
	if len(filePaths) == 0 {
		return nil, nil
	}

	// 4. Load per-file outlines for relevant files
	var allRelevantChunks []model.Chunk
	for _, fp := range filePaths {
		fileOutline, err := ReadPerFileOutline(jp.OutlinesDir, fp)
		if err != nil {
			continue
		}
		allRelevantChunks = append(allRelevantChunks, fileOutline.Chunks...)
	}

	if len(allRelevantChunks) == 0 {
		return nil, nil
	}

	// 5. Chunk-level retrieval (reuse existing function)
	miniOutline := &model.Outline{Chunks: allRelevantChunks}
	selected, err := RetrieveChunks(client, miniOutline, query, maxRetrieve)
	if err != nil {
		return nil, fmt.Errorf("retrieve chunks: %w", err)
	}

	return selected, nil
}

// parseRelevantFiles extracts file paths from LLM output
func parseRelevantFiles(output string) []string {
	lines := strings.Split(output, "\n")
	var paths []string
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		// File paths typically start with / or drive letter
		if strings.HasPrefix(line, "/") || strings.Contains(line, ":/") || strings.Contains(line, ":\\") {
			paths = append(paths, line)
		}
	}
	return paths
}

// IsFileChanged checks if a file has changed compared to the registry entry
func IsFileChanged(filePath string, entry *model.FileEntry) bool {
	info, err := os.Stat(filePath)
	if err != nil {
		return true // file not accessible, treat as changed
	}
	// Quick check: size
	if info.Size() != entry.Size {
		return true
	}
	// Quick check: mod time
	currentModTime := info.ModTime().Format("2006-01-02T15:04:05Z07:00")
	if currentModTime == entry.ModTime {
		return false
	}
	// Mod time changed, verify with hash
	hash, err := store.ComputeFileHash(filePath)
	if err != nil {
		return true
	}
	return hash != entry.Hash
}
