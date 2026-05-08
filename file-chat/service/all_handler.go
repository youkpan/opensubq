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

<files_summary>
%s
</files_summary>

请根据用户的问题，列出最相关的文件路径，每行一个，最多 %d 个。
只输出文件路径，不要输出其他内容。`

const maxOutlineSize = 1048576 // 1MB

// HandleAllFiles handles @全部 requests with two-level retrieval:
// 1. global_files_summary.xml → LLM selects relevant files
// 2. per-file outline → LLM selects relevant chunks
func HandleAllFiles(client *llm.Client, dp *store.DataPaths, query string, maxRetrieve int) ([]model.Chunk, error) {
	// 1. Read global summary
	summaryContent := store.ReadGlobalSummary(dp)
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
		fileOutline, err := ReadPerFileOutline(dp, fp)
		if err != nil {
			continue
		}
		allRelevantChunks = append(allRelevantChunks, fileOutline.Chunks...)
	}

	if len(allRelevantChunks) == 0 {
		return nil, nil
	}

	// 5. Chunk-level retrieval
	miniOutline := &model.Outline{Chunks: allRelevantChunks}
	selected, err := RetrieveChunks(client, miniOutline, query, maxRetrieve)
	if err != nil {
		return nil, fmt.Errorf("retrieve chunks: %w", err)
	}

	return selected, nil
}

// HandleAllFilesFromOutline uses global outline directly for retrieval
// Falls back to this when global summary is empty or insufficient
func HandleAllFilesFromOutline(client *llm.Client, dp *store.DataPaths, query string, maxRetrieve int) ([]model.Chunk, error) {
	// Read global outline
	outlineContent, err := store.ReadGlobalOutline(dp)
	if err != nil || outlineContent == "" {
		return nil, nil
	}

	// Truncate if exceeds 1MB
	if len(outlineContent) > maxOutlineSize {
		outlineContent = store.TrimOutlineToSize(outlineContent, maxOutlineSize)
	}

	outline := model.ParseOutline(outlineContent)
	if len(outline.Chunks) == 0 {
		return nil, nil
	}

	return RetrieveChunks(client, outline, query, maxRetrieve)
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
		return true
	}
	if info.Size() != entry.Size {
		return true
	}
	currentModTime := info.ModTime().Format("2006-01-02T15:04:05Z07:00")
	if currentModTime == entry.ModTime {
		return false
	}
	hash, err := store.ComputeFileHash(filePath)
	if err != nil {
		return true
	}
	return hash != entry.Hash
}
