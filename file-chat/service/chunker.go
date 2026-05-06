package service

import (
	"fmt"
	"os"
	"strings"

	"file-chat/llm"
	"file-chat/model"
	"file-chat/store"
)

const chunkingSystemPrompt = `你是一个文本分片助手。请将提供的文本按语义逻辑拆分为 1~3 个片段。

输出格式（每行一条，不要有其他内容）：
chunk_id|文件路径|片段摘要|起始行号|结束行号

然后空一行，输出每个片段的内容：
===chunk_id===
片段文本内容...

规则：
1. 按语义完整性和逻辑进行拆分
2. 每个片段应包含一个完整的概念或主题
3. 摘要 30~150 字，准确概括片段内容
4. 行号为原始文本中的实际行号（从1开始）
5. chunk_id 从 %s 开始递增（格式 chunk_NNN）
6. 相对路径使用提供的文件路径`

// ProcessFile processes a single file: convert if needed, then chunk
func ProcessFile(client *llm.Client, jp *store.JobPaths, filePath, markitdownCmd string, smallFileSize int64, outline *model.Outline) ([]model.Chunk, error) {
	// Check if already processed
	relPath := filePath
	processed := GetProcessedFiles(outline)
	if processed[relPath] {
		return nil, nil // already processed
	}

	// Get file info
	info, err := os.Stat(filePath)
	if err != nil {
		return nil, fmt.Errorf("stat file %s: %w", filePath, err)
	}

	// Read or convert content
	var content string
	if IsDocumentFile(filePath) {
		content, err = ConvertDocument(markitdownCmd, filePath)
		if err != nil {
			return nil, fmt.Errorf("convert %s: %w", filePath, err)
		}
	} else {
		data, err := os.ReadFile(filePath)
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", filePath, err)
		}
		content = string(data)
	}

	// Save source text for context expansion
	sourcePath := store.GetSourceFilePath(jp.Sources, relPath)
	if err := store.WriteFile(sourcePath, content); err != nil {
		return nil, fmt.Errorf("save source: %w", err)
	}

	// Small file: store as single chunk
	if info.Size() < smallFileSize {
		return storeSmallFile(jp, relPath, content, outline)
	}

	// Large file: LLM semantic chunking
	return chunkLargeFile(client, jp, relPath, content, outline)
}

func storeSmallFile(jp *store.JobPaths, relPath, content string, outline *model.Outline) ([]model.Chunk, error) {
	chunkID := NextChunkID(outline.Chunks)
	lines := strings.Split(content, "\n")

	chunk := model.Chunk{
		ID:        chunkID,
		FilePath:  relPath,
		Summary:   content,
		StartLine: 1,
		EndLine:   len(lines),
	}
	// Truncate summary for outline
	if len(chunk.Summary) > 150 {
		chunk.Summary = chunk.Summary[:150] + "..."
	}

	// Write chunk file
	chunkPath := store.GetChunkFilePath(jp.Chunks, relPath, chunkID)
	if err := store.WriteFile(chunkPath, content); err != nil {
		return nil, fmt.Errorf("write chunk: %w", err)
	}

	return []model.Chunk{chunk}, nil
}

func chunkLargeFile(client *llm.Client, jp *store.JobPaths, relPath, content string, outline *model.Outline) ([]model.Chunk, error) {
	lines := strings.Split(content, "\n")
	totalLines := len(lines)

	// Approximate tokens: ~4 chars per token for Chinese, ~1.5 for English
	// Use ~3 chars per token as rough average
	charsPerToken := 3
	chunkSizeChars := 2000 * charsPerToken // ~6000 chars per chunk window

	var allChunks []model.Chunk
	startLine := 0 // 0-indexed

	for startLine < totalLines {
		endLine := startLine
		charCount := 0
		for endLine < totalLines && charCount < chunkSizeChars {
			charCount += len(lines[endLine])
			endLine++
		}

		// Build text with line numbers
		var sb strings.Builder
		for i := startLine; i < endLine; i++ {
			sb.WriteString(fmt.Sprintf("%d|%s\n", i+1, lines[i]))
		}
		windowText := sb.String()

		// Prepare prompt
		nextID := NextChunkID(append(outline.Chunks, allChunks...))
		systemPrompt := fmt.Sprintf(chunkingSystemPrompt, nextID)

		// List existing dirs for context
		existingDirs := store.ListExistingDirs(jp.Chunks)
		dirsInfo := "（无已创建目录）"
		if len(existingDirs) > 0 {
			dirsInfo = strings.Join(existingDirs, "\n")
		}

		userMsg := fmt.Sprintf("文件路径：%s\n已有目录：\n%s\n\n待处理文本（第 %d ~ %d 行）：\n%s",
			relPath, dirsInfo, startLine+1, endLine, windowText)

		// Call LLM
		result, err := client.ChatSimple(systemPrompt, userMsg)
		if err != nil {
			// Fallback: treat the window as a single chunk
			chunk := model.Chunk{
				ID:        nextID,
				FilePath:  relPath,
				Summary:   fmt.Sprintf("第%d~%d行的内容", startLine+1, endLine),
				StartLine: startLine + 1,
				EndLine:   endLine,
			}
			chunkContent := strings.Join(lines[startLine:endLine], "\n")
			chunkPath := store.GetChunkFilePath(jp.Chunks, relPath, chunk.ID)
			if writeErr := store.WriteFile(chunkPath, chunkContent); writeErr != nil {
				return allChunks, fmt.Errorf("write fallback chunk: %w", writeErr)
			}
			allChunks = append(allChunks, chunk)
			startLine = endLine
			continue
		}

		// Parse LLM output
		chunks := parseChunkOutput(result, relPath)
		if len(chunks) == 0 {
			// Fallback
			chunk := model.Chunk{
				ID:        nextID,
				FilePath:  relPath,
				Summary:   fmt.Sprintf("第%d~%d行的内容", startLine+1, endLine),
				StartLine: startLine + 1,
				EndLine:   endLine,
			}
			chunkContent := strings.Join(lines[startLine:endLine], "\n")
			chunkPath := store.GetChunkFilePath(jp.Chunks, relPath, chunk.ID)
			store.WriteFile(chunkPath, chunkContent)
			allChunks = append(allChunks, chunk)
			startLine = endLine
			continue
		}

		// Write chunk files
		for _, c := range chunks {
			// Extract chunk content from source lines
			cStart := c.StartLine - 1 // convert to 0-indexed
			cEnd := c.EndLine
			if cStart < 0 {
				cStart = 0
			}
			if cEnd > totalLines {
				cEnd = totalLines
			}
			chunkContent := strings.Join(lines[cStart:cEnd], "\n")
			chunkPath := store.GetChunkFilePath(jp.Chunks, relPath, c.ID)
			if err := store.WriteFile(chunkPath, chunkContent); err != nil {
				return allChunks, fmt.Errorf("write chunk %s: %w", c.ID, err)
			}
			allChunks = append(allChunks, c)

			// Update startLine to continue from where this chunk ended
			if cEnd > startLine {
				startLine = cEnd
			}
		}

		// If no progress was made, advance by the window size
		if startLine == 0 || startLine <= (allChunks[0].StartLine-1) {
			startLine = endLine
		}
	}

	return allChunks, nil
}

// parseChunkOutput parses LLM chunk output into Chunk structs
func parseChunkOutput(output, defaultPath string) []model.Chunk {
	lines := strings.Split(output, "\n")
	var chunks []model.Chunk
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "===") {
			continue
		}
		parts := strings.SplitN(line, "|", 5)
		if len(parts) != 5 {
			continue
		}
		start, err1 := parseLineNum(parts[3])
		end, err2 := parseLineNum(parts[4])
		if err1 != nil || err2 != nil {
			continue
		}
		chunk := model.Chunk{
			ID:        parts[0],
			FilePath:  parts[1],
			Summary:   parts[2],
			StartLine: start,
			EndLine:   end,
		}
		if chunk.FilePath == "" {
			chunk.FilePath = defaultPath
		}
		// Validate chunk ID format
		if !strings.HasPrefix(chunk.ID, "chunk_") {
			continue
		}
		chunks = append(chunks, chunk)
	}
	return chunks
}

func parseLineNum(s string) (int, error) {
	s = strings.TrimSpace(s)
	var n int
	_, err := fmt.Sscanf(s, "%d", &n)
	return n, err
}
