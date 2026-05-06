package service

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"file-chat/llm"
	"file-chat/model"
	"file-chat/store"
)

const retrievalSystemPrompt = `你是一个文档检索助手。以下是文档的索引大纲：

%s

请根据用户的问题，按相关性从高到低列出最相关的片段ID。
只输出chunk_id，每行一个，最多 %d 个。
不要输出任何其他内容。`

// RetrieveChunks selects the most relevant chunks for a query
func RetrieveChunks(client *llm.Client, outline *model.Outline, query string, maxRetrieve int) ([]model.Chunk, error) {
	if len(outline.Chunks) == 0 {
		return nil, nil
	}

	systemPrompt := fmt.Sprintf(retrievalSystemPrompt, outline.String(), maxRetrieve)
	result, err := client.ChatSimple(systemPrompt, query)
	if err != nil {
		return nil, fmt.Errorf("retrieve: %w", err)
	}

	// Parse chunk IDs from result
	orderedIDs := parseChunkIDs(result, maxRetrieve)

	// Build lookup
	chunkMap := make(map[string]model.Chunk)
	for _, c := range outline.Chunks {
		chunkMap[c.ID] = c
	}

	// Collect selected chunks, preserving relevance order
	var selected []model.Chunk
	for _, id := range orderedIDs {
		if c, ok := chunkMap[id]; ok {
			selected = append(selected, c)
		}
	}

	return selected, nil
}

// BuildContext reads selected chunks with expanded context and builds final context string
func BuildContext(chunksDir, sourcesDir string, selected []model.Chunk) (string, error) {
	if len(selected) == 0 {
		return "", nil
	}

	// Sort by file path then start line for consistent ordering
	sorted := make([]model.Chunk, len(selected))
	copy(sorted, selected)
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].FilePath != sorted[j].FilePath {
			return sorted[i].FilePath < sorted[j].FilePath
		}
		return sorted[i].StartLine < sorted[j].StartLine
	})

	var sb strings.Builder
	sb.WriteString("<context>\n")

	prevFile := ""
	prevEnd := 0

	for i, c := range sorted {
		// Read chunk content
		chunkPath := store.GetChunkFilePath(chunksDir, c.FilePath, c.ID)
		chunkContent, err := store.ReadFile(chunkPath)
		if err != nil {
			continue
		}

		// Determine if we need a chunk wrapper (non-adjacent)
		needWrapper := i == 0 || c.FilePath != prevFile || c.StartLine > prevEnd+1

		// Add expanded context from source file
		expandedContent := expandContext(sourcesDir, c, chunkContent)

		if needWrapper {
			if i > 0 {
				sb.WriteString(fmt.Sprintf("</%s>\n", sorted[i-1].ID))
			}
			sb.WriteString(fmt.Sprintf("<%s>\n", c.ID))
		}

		sb.WriteString(expandedContent)
		sb.WriteString("\n")

		prevFile = c.FilePath
		prevEnd = c.EndLine

		// Close last tag
		if i == len(sorted)-1 {
			sb.WriteString(fmt.Sprintf("</%s>\n", c.ID))
		}
	}

	sb.WriteString("</context>")
	return sb.String(), nil
}

// expandContext adds surrounding context from the original source file
func expandContext(sourcesDir string, chunk model.Chunk, chunkContent string) string {
	sourcePath := store.GetSourceFilePath(sourcesDir, chunk.FilePath)
	data, err := os.ReadFile(sourcePath)
	if err != nil {
		return chunkContent
	}
	sourceContent := string(data)
	lines := strings.Split(sourceContent, "\n")

	// Get expanded content: chunk + surrounding context
	startIdx := chunk.StartLine - 1 // 0-indexed
	endIdx := chunk.EndLine

	// Expand before: up to 500-1200 bytes
	beforeStart := startIdx
	byteCount := 0
	for beforeStart > 0 {
		byteCount += len(lines[beforeStart-1]) + 1 // +1 for newline
		if byteCount > 1200 {
			break
		}
		if byteCount > 500 {
			// Try to break at paragraph boundary
			if strings.TrimSpace(lines[beforeStart-1]) == "" {
				break
			}
		}
		if byteCount > 800 {
			// Try to break at line boundary
			if strings.TrimSpace(lines[beforeStart-1]) == "" || strings.HasSuffix(lines[beforeStart-1], "。") || strings.HasSuffix(lines[beforeStart-1], ".") {
				break
			}
		}
		beforeStart--
	}

	// Expand after: up to 500-1200 bytes
	afterEnd := endIdx
	byteCount = 0
	for afterEnd < len(lines) {
		byteCount += len(lines[afterEnd]) + 1
		if byteCount > 1200 {
			break
		}
		if byteCount > 500 {
			if strings.TrimSpace(lines[afterEnd]) == "" {
				break
			}
		}
		if byteCount > 800 {
			if strings.TrimSpace(lines[afterEnd]) == "" || strings.HasSuffix(lines[afterEnd], "。") || strings.HasSuffix(lines[afterEnd], ".") {
				break
			}
		}
		afterEnd++
	}

	var sb strings.Builder
	if beforeStart < startIdx {
		sb.WriteString(strings.Join(lines[beforeStart:startIdx], "\n"))
		sb.WriteString("\n")
	}
	sb.WriteString(chunkContent)
	if endIdx < afterEnd {
		sb.WriteString("\n")
		sb.WriteString(strings.Join(lines[endIdx:afterEnd], "\n"))
	}
	return sb.String()
}

// parseChunkIDs parses chunk IDs from LLM output
func parseChunkIDs(output string, max int) []string {
	lines := strings.Split(output, "\n")
	var ids []string
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "chunk_") {
			ids = append(ids, line)
			if len(ids) >= max {
				break
			}
		}
	}
	return ids
}

// resolvePathForStore is a helper to get relative path for store operations
func resolvePathForStore(baseDir, absPath string) string {
	rel, err := filepath.Rel(baseDir, absPath)
	if err != nil {
		return absPath
	}
	return rel
}
