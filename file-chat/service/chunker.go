package service

import (
	"fmt"
	"log"
	"os"
	"strings"
	"sync"

	"file-chat/llm"
	"file-chat/model"
	"file-chat/store"
)

const (
	segmentSizeChars = 30 * 1024 // 30KB per segment
	maxConcurrency   = 20        // max concurrent goroutines
)

const chunkingSystemPrompt = `你是一个文本分片助手。请将提供的文本按语义逻辑拆分为 1~3 个片段。

输出格式（每行一条，不要有其他内容）：
chunk_id|文件路径|片段摘要|起始行号|结束行号

规则：
1. 按语义完整性和逻辑进行拆分
2. 每个片段应包含一个完整的概念或主题
3. 摘要 30~150 字，准确概括片段内容
4. 行号为原始文本中的实际行号
5. 相对路径使用提供的文件路径
6. chunk_id 格式为 temp_NNN（N从1递增）`

// ProcessFile processes a single file: convert if needed, then chunk
func ProcessFile(client *llm.Client, jp *store.JobPaths, filePath, markitdownCmd string, smallFileSize int64, outline *model.Outline) ([]model.Chunk, error) {
	relPath := filePath
	processed := GetProcessedFiles(outline)
	if processed[relPath] {
		return nil, nil
	}

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

	// Save source text
	sourcePath := store.GetSourceFilePath(jp.Sources, relPath)
	if err := store.WriteFile(sourcePath, content); err != nil {
		return nil, fmt.Errorf("save source: %w", err)
	}

	// Small file: single chunk
	if info.Size() < smallFileSize {
		return storeSmallFile(client, jp, relPath, content, outline)
	}

	// Large file: parallel LLM chunking
	return chunkLargeFileParallel(client, jp, relPath, content, outline)
}

func storeSmallFile(client *llm.Client, jp *store.JobPaths, relPath, content string, outline *model.Outline) ([]model.Chunk, error) {
	chunkID := NextChunkID(outline.Chunks)
	lines := strings.Split(content, "\n")

	// Use LLM to generate summary
	summary, err := generateChunkSummary(client, relPath, content)
	if err != nil {
		log.Printf("generate chunk summary for %s: %v", relPath, err)
		// Fallback: truncate content
		summary = strings.ReplaceAll(content, "\n", " ")
		summary = strings.ReplaceAll(summary, "\r", " ")
		if len(summary) > 150 {
			summary = summary[:150] + "..."
		}
	}

	chunk := model.Chunk{
		ID:        chunkID,
		FilePath:  relPath,
		Summary:   summary,
		StartLine: 1,
		EndLine:   len(lines),
	}

	chunkPath := store.GetChunkFilePath(jp.Chunks, relPath, chunkID)
	if err := store.WriteFile(chunkPath, content); err != nil {
		return nil, fmt.Errorf("write chunk: %w", err)
	}

	return []model.Chunk{chunk}, nil
}

// segment represents a line range in a file
type segment struct {
	startLine int // 0-indexed
	endLine   int // exclusive
	index     int // segment order
}

// chunkResult holds the result of processing a segment
type chunkResult struct {
	index  int
	chunks []model.Chunk
	err    error
}

// chunkLargeFileParallel splits a large file into ~30KB segments and processes them concurrently
func chunkLargeFileParallel(client *llm.Client, jp *store.JobPaths, relPath, content string, outline *model.Outline) ([]model.Chunk, error) {
	lines := strings.Split(content, "\n")
	totalLines := len(lines)

	// Split into ~30KB segments by line boundaries
	segments := splitSegments(lines, segmentSizeChars)
	if len(segments) == 0 {
		return nil, nil
	}

	// Process segments concurrently
	results := make([]chunkResult, len(segments))
	sem := make(chan struct{}, maxConcurrency)
	var wg sync.WaitGroup

	for i, seg := range segments {
		wg.Add(1)
		sem <- struct{}{}
		go func(idx int, s segment) {
			defer wg.Done()
			defer func() { <-sem }()
			results[idx] = processSegment(client, jp, relPath, lines, s, totalLines)
		}(i, seg)
	}
	wg.Wait()

	// Assemble results in segment order, then renumber chunk IDs
	nextIDNum := nextChunkNum(outline.Chunks)
	var allChunks []model.Chunk

	for _, r := range results {
		if r.err != nil {
			log.Printf("segment %d error: %v", r.index, r.err)
			// Fallback: create a single chunk for the failed segment's range
			seg := segments[r.index]
			summary := fmt.Sprintf("第%d~%d行的内容", seg.startLine+1, seg.endLine)
			c := model.Chunk{
				ID:        fmt.Sprintf("chunk_%03d", nextIDNum),
				FilePath:  relPath,
				Summary:   summary,
				StartLine: seg.startLine + 1,
				EndLine:   seg.endLine,
			}
			chunkContent := strings.Join(lines[seg.startLine:seg.endLine], "\n")
			chunkPath := store.GetChunkFilePath(jp.Chunks, relPath, c.ID)
			store.WriteFile(chunkPath, chunkContent)
			allChunks = append(allChunks, c)
			nextIDNum++
			continue
		}
		for _, c := range r.chunks {
			// Renumber with sequential IDs
			c.ID = fmt.Sprintf("chunk_%03d", nextIDNum)
			// Write chunk file
			cStart := c.StartLine - 1
			cEnd := c.EndLine
			if cStart < 0 {
				cStart = 0
			}
			if cEnd > totalLines {
				cEnd = totalLines
			}
			chunkContent := strings.Join(lines[cStart:cEnd], "\n")
			chunkPath := store.GetChunkFilePath(jp.Chunks, relPath, c.ID)
			store.WriteFile(chunkPath, chunkContent)
			allChunks = append(allChunks, c)
			nextIDNum++
		}
	}

	return allChunks, nil
}

// processSegment processes a single segment with LLM
func processSegment(client *llm.Client, jp *store.JobPaths, relPath string, lines []string, seg segment, totalLines int) chunkResult {
	// Build text with line numbers
	var sb strings.Builder
	for i := seg.startLine; i < seg.endLine; i++ {
		fmt.Fprintf(&sb, "%d|%s\n", i+1, lines[i])
	}
	windowText := sb.String()

	existingDirs := store.ListExistingDirs(jp.Chunks)
	dirsInfo := "（无已创建目录）"
	if len(existingDirs) > 0 {
		dirsInfo = strings.Join(existingDirs, "\n")
	}

	userMsg := fmt.Sprintf("文件路径：%s\n已有目录：\n%s\n\n待处理文本（第 %d ~ %d 行）：\n%s",
		relPath, dirsInfo, seg.startLine+1, seg.endLine, windowText)

	result, err := client.ChatSimple(chunkingSystemPrompt, userMsg)
	if err != nil {
		return chunkResult{index: seg.index, err: err}
	}

	chunks := parseChunkOutput(result, relPath)
	if len(chunks) == 0 {
		// Return as fallback
		summary := strings.ReplaceAll(
			strings.Join(lines[seg.startLine:min(seg.startLine+5, seg.endLine)], " "),
			"\n", " ",
		)
		if len(summary) > 150 {
			summary = summary[:150] + "..."
		}
		chunks = []model.Chunk{{
			ID:        "temp_001",
			FilePath:  relPath,
			Summary:   summary,
			StartLine: seg.startLine + 1,
			EndLine:   seg.endLine,
		}}
	}

	return chunkResult{index: seg.index, chunks: chunks}
}

// splitSegments splits lines into segments of approximately maxSize bytes
func splitSegments(lines []string, maxSize int) []segment {
	var segments []segment
	start := 0
	currentSize := 0

	for i, line := range lines {
		lineSize := len(line) + 1 // +1 for newline
		if currentSize+lineSize > maxSize && i > start {
			segments = append(segments, segment{startLine: start, endLine: i, index: len(segments)})
			start = i
			currentSize = 0
		}
		currentSize += lineSize
	}

	if start < len(lines) {
		segments = append(segments, segment{startLine: start, endLine: len(lines), index: len(segments)})
	}

	return segments
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
		if !strings.HasPrefix(chunk.ID, "chunk_") && !strings.HasPrefix(chunk.ID, "temp_") {
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

// nextChunkNum returns the next chunk number (int) based on existing chunks
func nextChunkNum(chunks []model.Chunk) int {
	maxID := 0
	for _, c := range chunks {
		if strings.HasPrefix(c.ID, "chunk_") {
			num, err := parseLineNum(strings.TrimPrefix(c.ID, "chunk_"))
			if err == nil && num > maxID {
				maxID = num
			}
		}
	}
	return maxID + 1
}

// generateChunkSummary uses LLM to generate a chunk summary (50-150 chars)
func generateChunkSummary(client *llm.Client, filePath, content string) (string, error) {
	// Truncate content to avoid too large prompt
	truncated := content
	if len(truncated) > 3000 {
		truncated = truncated[:3000] + "..."
	}
	prompt := fmt.Sprintf(
		"请用50-150字概括以下文件内容的核心要点，直接输出摘要，不要其他内容。\n\n文件：%s\n\n%s",
		filePath, truncated,
	)
	return client.ChatSimple("", prompt)
}
