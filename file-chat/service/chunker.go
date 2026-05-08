package service

import (
	"bytes"
	"encoding/hex"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"sync"

	"file-chat/llm"
	"file-chat/model"
	"file-chat/store"
)

const (
	segmentSizeChars     = 30 * 1024 // 30KB per segment
	maxConcurrency       = 20        // max concurrent goroutines
	maxChunkBytesNew     = 8000      // new file split threshold
	maxChunkBytesUpdate  = 6000      // incremental update split threshold
)

const chunkingSystemPrompt = `你是一个文本分片助手。请将提供的文本按语义逻辑拆分为 1~3 个片段。

输出格式（每行一条，不要有其他内容）：
chunk_id|片段摘要|起始行号|结束行号

规则：
1. 按语义完整性和逻辑进行拆分
2. 每个片段应包含一个完整的概念或主题
3. 摘要 30~150 字，准确概括片段内容
4. 行号为原始文本中的实际行号
5. chunk_id 格式为 temp_NNN（N从1递增）
6. 注意：<content> 标签内的文本才是需要处理的内容`

// linesToByteOffsets builds a mapping from 1-based line number to byte offset
func linesToByteOffsets(lines []string) []int64 {
	offsets := make([]int64, len(lines)+2) // 1-indexed, +1 for end-of-file
	var off int64
	for i, line := range lines {
		offsets[i+1] = off
		off += int64(len(line) + 1) // +1 for \n
	}
	offsets[len(lines)+1] = off // past-end offset
	return offsets
}

// ProcessFile processes a single file: convert if needed, then chunk
func ProcessFile(client *llm.Client, dp *store.DataPaths, filePath, markitdownCmd string, smallFileSize int64, outline *model.Outline) ([]model.Chunk, []model.ChunkMeta, error) {
	processed := GetProcessedFiles(outline)
	if processed[filePath] {
		return nil, nil, nil
	}

	info, err := os.Stat(filePath)
	if err != nil {
		return nil, nil, fmt.Errorf("stat file %s: %w", filePath, err)
	}

	// Read or convert content
	var content string
	if IsDocumentFile(filePath) {
		content, err = ConvertDocument(markitdownCmd, filePath)
		if err != nil {
			return nil, nil, fmt.Errorf("convert %s: %w", filePath, err)
		}
	} else {
		data, err := os.ReadFile(filePath)
		if err != nil {
			return nil, nil, fmt.Errorf("read %s: %w", filePath, err)
		}
		content = string(data)
	}

	// Create file storage directory
	if err := dp.InitFileDir(filePath); err != nil {
		return nil, nil, fmt.Errorf("init file dir: %w", err)
	}

	// Save source only for document files (non-plain-text)
	if !IsPlainTextFile(filePath) {
		sourcePath := dp.GetFileSourcePath(filePath)
		if err := store.WriteFile(sourcePath, content); err != nil {
			return nil, nil, fmt.Errorf("save source: %w", err)
		}
	}

	// Small file: single chunk
	if info.Size() < smallFileSize {
		return processSmallFile(client, dp, filePath, content, outline)
	}

	// Large file: parallel LLM chunking
	return processLargeFileParallel(client, dp, filePath, content, outline)
}

func processSmallFile(client *llm.Client, dp *store.DataPaths, filePath, content string, outline *model.Outline) ([]model.Chunk, []model.ChunkMeta, error) {
	nextID := nextChunkNum(outline.Chunks)
	chunkID := fmt.Sprintf("chunk_%03d", nextID)
	nextID++

	summary, err := generateChunkSummary(client, filePath, content)
	if err != nil {
		log.Printf("generate chunk summary for %s: %v", filePath, err)
		summary = strings.ReplaceAll(content, "\n", " ")
		summary = strings.ReplaceAll(summary, "\r", " ")
		if len(summary) > 150 {
			summary = summary[:150] + "..."
		}
	}

	startByte := int64(0)
	endByte := int64(len(content))
	chunkContent := []byte(content)

	chunk := model.Chunk{
		ID:        chunkID,
		FilePath:  filePath,
		Summary:   summary,
		StartByte: startByte,
		EndByte:   endByte,
	}

	metas := maybeSplitChunks([]model.ChunkMeta{
		chunk.ToMeta(chunkContent),
	}, []byte(content), &nextID, maxChunkBytesNew)

	return []model.Chunk{chunk}, metas, nil
}

// segment represents a line range in a file
type segment struct {
	startLine int // 0-indexed
	endLine   int // exclusive
	index     int // segment order
}

// lineChunk is intermediate: LLM output with line numbers, before converting to byte offsets
type lineChunk struct {
	ID        string
	Summary   string
	StartLine int // 1-based
	EndLine   int // 1-based, inclusive
}

// lineChunkResult holds the result of processing a segment
type lineChunkResult struct {
	index  int
	chunks []lineChunk
	err    error
}

// processLargeFileParallel splits a large file into ~30KB segments and processes them concurrently
func processLargeFileParallel(client *llm.Client, dp *store.DataPaths, filePath, content string, outline *model.Outline) ([]model.Chunk, []model.ChunkMeta, error) {
	lines := strings.Split(content, "\n")
	byteOffsets := linesToByteOffsets(lines)

	segments := splitSegments(lines, segmentSizeChars)
	if len(segments) == 0 {
		return nil, nil, nil
	}

	results := make([]lineChunkResult, len(segments))
	sem := make(chan struct{}, maxConcurrency)
	var wg sync.WaitGroup

	for i, seg := range segments {
		wg.Add(1)
		sem <- struct{}{}
		go func(idx int, s segment) {
			defer wg.Done()
			defer func() { <-sem }()
			results[idx] = processSegment(client, filePath, lines, s)
		}(i, seg)
	}
	wg.Wait()

	nextIDNum := nextChunkNum(outline.Chunks)
	var allChunks []model.Chunk
	var allMetas []model.ChunkMeta
	contentBytes := []byte(content)

	for ri, r := range results {
		if r.err != nil {
			log.Printf("segment %d error: %v", r.index, r.err)
			seg := segments[ri]
			summary := fmt.Sprintf("第%d~%d行的内容", seg.startLine+1, seg.endLine)
			startByte := byteOffsets[seg.startLine+1]
			endByte := byteOffsets[seg.endLine+1]
			c := model.Chunk{
				ID:        fmt.Sprintf("chunk_%03d", nextIDNum),
				FilePath:  filePath,
				Summary:   summary,
				StartByte: startByte,
				EndByte:   endByte,
			}
			chunkContent := contentBytes[startByte:endByte]
			allMetas = append(allMetas, maybeSplitChunks([]model.ChunkMeta{c.ToMeta(chunkContent)}, contentBytes, &nextIDNum, maxChunkBytesNew)...)
			allChunks = append(allChunks, c)
			nextIDNum++
			continue
		}
		for _, lc := range r.chunks {
			cID := fmt.Sprintf("chunk_%03d", nextIDNum)
			startIdx := lc.StartLine
			if startIdx < 1 {
				startIdx = 1
			}
			if startIdx > len(lines) {
				startIdx = len(lines)
			}
			startByte := byteOffsets[startIdx]

			endIdx := lc.EndLine + 1
			if endIdx < 1 {
				endIdx = 1
			}
			if endIdx > len(lines)+1 {
				endIdx = len(lines) + 1
			}
			endByte := byteOffsets[endIdx]

			chunkContent := contentBytes[startByte:endByte]

			c := model.Chunk{
				ID:        cID,
				FilePath:  filePath,
				Summary:   lc.Summary,
				StartByte: startByte,
				EndByte:   endByte,
			}
			allMetas = append(allMetas, maybeSplitChunks([]model.ChunkMeta{c.ToMeta(chunkContent)}, contentBytes, &nextIDNum, maxChunkBytesNew)...)
			allChunks = append(allChunks, c)
			nextIDNum++
		}
	}

	return allChunks, allMetas, nil
}

// processSegment processes a single segment with LLM, returns line-based chunks
func processSegment(client *llm.Client, filePath string, lines []string, seg segment) lineChunkResult {
	var sb strings.Builder
	for i := seg.startLine; i < seg.endLine; i++ {
		fmt.Fprintf(&sb, "%d|%s\n", i+1, lines[i])
	}
	windowText := sb.String()

	userMsg := fmt.Sprintf("文件路径：%s\n\n<content>\n待处理文本（第 %d ~ %d 行）：\n%s\n</content>",
		filePath, seg.startLine+1, seg.endLine, windowText)

	result, err := client.ChatSimple(chunkingSystemPrompt, userMsg)
	if err != nil {
		return lineChunkResult{index: seg.index, err: err}
	}

	lcs := parseLineChunkOutput(result)
	if len(lcs) == 0 {
		summary := strings.ReplaceAll(
			strings.Join(lines[seg.startLine:min(seg.startLine+5, seg.endLine)], " "),
			"\n", " ",
		)
		if len(summary) > 150 {
			summary = summary[:150] + "..."
		}
		lcs = []lineChunk{{
			ID:        "temp_001",
			Summary:   summary,
			StartLine: seg.startLine + 1,
			EndLine:   seg.endLine,
		}}
	}

	return lineChunkResult{index: seg.index, chunks: lcs}
}

// parseLineChunkOutput parses LLM output into lineChunk structs (4-field: id|summary|start|end)
func parseLineChunkOutput(output string) []lineChunk {
	lines := strings.Split(output, "\n")
	var chunks []lineChunk
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "===") {
			continue
		}
		parts := strings.SplitN(line, "|", 4)
		if len(parts) != 4 {
			continue
		}
		start, err1 := parseLineNum(parts[2])
		end, err2 := parseLineNum(parts[3])
		if err1 != nil || err2 != nil {
			continue
		}
		chunk := lineChunk{
			ID:        parts[0],
			Summary:   parts[1],
			StartLine: start,
			EndLine:   end,
		}
		if !strings.HasPrefix(chunk.ID, "chunk_") && !strings.HasPrefix(chunk.ID, "temp_") {
			continue
		}
		chunks = append(chunks, chunk)
	}
	return chunks
}

// maybeSplitChunks splits oversized chunks into new sequential chunks
func maybeSplitChunks(metas []model.ChunkMeta, content []byte, nextIDNum *int, maxSize int) []model.ChunkMeta {
	var result []model.ChunkMeta
	for _, m := range metas {
		chunkSize := m.EndByte - m.StartByte
		if chunkSize <= int64(maxSize) {
			result = append(result, m)
			continue
		}
		// Split into new sequential chunks
		offset := m.StartByte
		for offset < m.EndByte {
			end := offset + int64(maxSize)
			if end > m.EndByte {
				end = m.EndByte
			}
			// Try to find a newline boundary in the last 500 bytes of the window
			if end < m.EndByte {
				searchStart := end - 500
				if searchStart < offset {
					searchStart = offset
				}
				for i := end; i > searchStart; i-- {
					if content[i-1] == '\n' {
						end = i
						break
					}
				}
			}
			subContent := content[offset:end]
			subMeta := model.ChunkMeta{
				ID:        fmt.Sprintf("chunk_%03d", *nextIDNum),
				StartByte: offset,
				EndByte:   end,
				Head30:    model.ComputeHead30(subContent),
				Tail30:    model.ComputeTail30(subContent),
				Hash:      model.ComputeChunkHash(subContent),
				Summary:   m.Summary,
			}
			result = append(result, subMeta)
			offset = end
			(*nextIDNum)++
		}
	}
	return result
}

// splitSegments splits lines into segments of approximately maxSize bytes
func splitSegments(lines []string, maxSize int) []segment {
	var segments []segment
	start := 0
	currentSize := 0

	for i, line := range lines {
		lineSize := len(line) + 1
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
			numStr := strings.TrimPrefix(c.ID, "chunk_")
			num, err := parseLineNum(numStr)
			if err == nil && num > maxID {
				maxID = num
			}
		}
	}
	return maxID + 1
}

// generateChunkSummary uses LLM to generate a chunk summary (50-150 chars)
func generateChunkSummary(client *llm.Client, filePath, content string) (string, error) {
	truncated := content
	if len(truncated) > 3000 {
		truncated = truncated[:3000] + "..."
	}
	prompt := fmt.Sprintf(
		"请用50-150字概括以下文件内容的核心要点，直接输出摘要，不要其他内容。\n\n文件：%s\n\n<content>\n%s\n</content>",
		filePath, truncated,
	)
	return client.ChatSimple("", prompt)
}

// IncrementalUpdateFile handles incremental update when a file has changed
func IncrementalUpdateFile(client *llm.Client, dp *store.DataPaths, filePath, markitdownCmd string, smallFileSize int64) ([]model.Chunk, []model.ChunkMeta, error) {
	// Read old chunks
	oldCF, err := store.ReadChunksJSON(dp, filePath)
	if err != nil {
		log.Printf("read old chunks.json for incremental update: %v", err)
		return ProcessFile(client, dp, filePath, markitdownCmd, smallFileSize, &model.Outline{})
	}
	if len(oldCF.Chunks) == 0 {
		return ProcessFile(client, dp, filePath, markitdownCmd, smallFileSize, &model.Outline{})
	}

	// Read current content
	var content string
	if IsDocumentFile(filePath) {
		content, err = ConvertDocument(markitdownCmd, filePath)
		if err != nil {
			return nil, nil, fmt.Errorf("convert %s: %w", filePath, err)
		}
	} else {
		data, err := os.ReadFile(filePath)
		if err != nil {
			return nil, nil, fmt.Errorf("read %s: %w", filePath, err)
		}
		content = string(data)
	}

	// Save source for document files
	if !IsPlainTextFile(filePath) {
		sourcePath := dp.GetFileSourcePath(filePath)
		store.WriteFile(sourcePath, content)
	}

	contentBytes := []byte(content)

	// Incremental alignment: check each old chunk against new content
	var result []model.ChunkMeta
	i := 0
	totalShift := 0 // cumulative ID shift

	for i < len(oldCF.Chunks) {
		old := oldCF.Chunks[i]
		currentID := shiftChunkID(old.ID, totalShift)

		// Check if content at old byte range still matches
		if old.EndByte <= int64(len(contentBytes)) && old.StartByte >= 0 {
			currentHash := model.ComputeChunkHash(contentBytes[old.StartByte:old.EndByte])
			if currentHash == old.Hash {
				// Unchanged: keep with adjusted ID
				adjusted := old
				adjusted.ID = currentID
				result = append(result, adjusted)
				i++
				continue
			}
		}

		// Hash mismatch: find change point
		changeStart := old.StartByte

		// Try to align subsequent chunks by searching for their Head30
		alignIdx, alignPos := findAlignment(oldCF.Chunks, i+1, contentBytes)

		if alignIdx >= 0 {
			// Aligned: reprocess [changeStart, alignPos)
			newMetas := reprocessRegion(client, filePath, contentBytes, changeStart, alignPos, currentID)

			// newMetas took N IDs starting from currentID
			shift := len(newMetas) - 1
			totalShift += shift

			result = append(result, newMetas...)

			// Adjust remaining chunks: byte offset + ID shift
			delta := alignPos - oldCF.Chunks[alignIdx].StartByte
			for j := alignIdx; j < len(oldCF.Chunks); j++ {
				adjusted := oldCF.Chunks[j]
				adjusted.StartByte += delta
				adjusted.EndByte += delta
				adjusted.ID = shiftChunkID(adjusted.ID, totalShift)
				result = append(result, adjusted)
			}

			// Continue checking from aligned chunk's next
			// Re-verify the adjusted chunks in case more changed
			// For now, trust alignment and break the loop
			break
		} else {
			// No alignment found: reprocess from changeStart to end
			newMetas := reprocessToEnd(client, dp, filePath, contentBytes, changeStart, currentID)
			result = append(result, newMetas...)
			break
		}
	}

	// Build runtime chunks
	var allChunks []model.Chunk
	for _, m := range result {
		allChunks = append(allChunks, model.Chunk{
			ID:        m.ID,
			FilePath:  filePath,
			Summary:   m.Summary,
			StartByte: m.StartByte,
			EndByte:   m.EndByte,
		})
	}

	return allChunks, result, nil
}

// findAlignment searches for a subsequent chunk's Head30 in content to find alignment point
func findAlignment(oldChunks []model.ChunkMeta, from int, content []byte) (int, int64) {
	for j := from; j < len(oldChunks); j++ {
		head30Bytes, err := hex.DecodeString(oldChunks[j].Head30)
		if err != nil || len(head30Bytes) == 0 {
			continue
		}
		pos := bytes.Index(content, head30Bytes)
		if pos >= 0 {
			return j, int64(pos)
		}
	}
	return -1, -1
}

// shiftChunkID shifts a chunk ID by a given amount: "chunk_010" + 2 → "chunk_012"
func shiftChunkID(id string, shift int) string {
	numStr := strings.TrimPrefix(id, "chunk_")
	num, err := strconv.Atoi(numStr)
	if err != nil {
		return id
	}
	return fmt.Sprintf("chunk_%03d", num+shift)
}

// reprocessRegion reprocesses a byte region [start, end) with LLM summary generation
func reprocessRegion(client *llm.Client, filePath string, content []byte, start, end int64, startID string) []model.ChunkMeta {
	region := content[start:end]
	startNum := 0
	fmt.Sscanf(strings.TrimPrefix(startID, "chunk_"), "%d", &startNum)

	var metas []model.ChunkMeta
	nextID := startNum

	// If region is small enough, single chunk
	if len(region) <= maxChunkBytesUpdate {
		summary, err := generateChunkSummary(client, filePath, string(region))
		if err != nil {
			summary = fmt.Sprintf("第%d-%d字节的内容", start, end)
		}
		metas = append(metas, model.ChunkMeta{
			ID:        fmt.Sprintf("chunk_%03d", nextID),
			StartByte: start,
			EndByte:   end,
			Head30:    model.ComputeHead30(region),
			Tail30:    model.ComputeTail30(region),
			Hash:      model.ComputeChunkHash(region),
			Summary:   summary,
		})
		return metas
	}

	// Split by line boundaries and generate summaries
	offset := start
	for offset < end {
		chunkEnd := offset + int64(maxChunkBytesUpdate)
		if chunkEnd > end {
			chunkEnd = end
		}
		// Find newline boundary
		if chunkEnd < end {
			searchStart := chunkEnd - 500
			if searchStart < offset {
				searchStart = offset
			}
			for i := chunkEnd; i > searchStart; i-- {
				if content[i-1] == '\n' {
					chunkEnd = i
					break
				}
			}
		}

		chunkContent := content[offset:chunkEnd]
		summary, err := generateChunkSummary(client, filePath, string(chunkContent))
		if err != nil {
			summary = fmt.Sprintf("第%d-%d字节的内容", offset, chunkEnd)
		}
		metas = append(metas, model.ChunkMeta{
			ID:        fmt.Sprintf("chunk_%03d", nextID),
			StartByte: offset,
			EndByte:   chunkEnd,
			Head30:    model.ComputeHead30(chunkContent),
			Tail30:    model.ComputeTail30(chunkContent),
			Hash:      model.ComputeChunkHash(chunkContent),
			Summary:   summary,
		})
		offset = chunkEnd
		nextID++
	}
	return metas
}

// reprocessToEnd reprocesses from start to end of file using full LLM chunking
func reprocessToEnd(client *llm.Client, dp *store.DataPaths, filePath string, content []byte, start int64, startID string) []model.ChunkMeta {
	startNum := 0
	fmt.Sscanf(strings.TrimPrefix(startID, "chunk_"), "%d", &startNum)

	region := string(content[start:])
	lines := strings.Split(region, "\n")

	// Build byte offsets relative to original content
	segments := splitSegments(lines, segmentSizeChars)
	if len(segments) == 0 {
		return nil
	}

	results := make([]lineChunkResult, len(segments))
	sem := make(chan struct{}, maxConcurrency)
	var wg sync.WaitGroup

	for i, seg := range segments {
		wg.Add(1)
		sem <- struct{}{}
		go func(idx int, s segment) {
			defer wg.Done()
			defer func() { <-sem }()
			results[idx] = processSegment(client, filePath, lines, s)
		}(i, seg)
	}
	wg.Wait()

	nextID := startNum
	var metas []model.ChunkMeta

	// Build absolute byte offsets
	absOffsets := make([]int64, len(lines)+1)
	absOffsets[0] = start
	for i, line := range lines {
		absOffsets[i+1] = absOffsets[i] + int64(len(line)+1)
	}

	for ri, r := range results {
		if r.err != nil {
			seg := segments[ri]
			summary := fmt.Sprintf("第%d~%d行的内容", seg.startLine+1, seg.endLine)
			sByte := absOffsets[seg.startLine]
			eByte := absOffsets[seg.endLine]
			chunkContent := content[sByte:eByte]
			metas = append(metas, model.ChunkMeta{
				ID:        fmt.Sprintf("chunk_%03d", nextID),
				StartByte: sByte,
				EndByte:   eByte,
				Head30:    model.ComputeHead30(chunkContent),
				Tail30:    model.ComputeTail30(chunkContent),
				Hash:      model.ComputeChunkHash(chunkContent),
				Summary:   summary,
			})
			nextID++
			continue
		}
		for _, lc := range r.chunks {
			startIdx := lc.StartLine
			if startIdx < 1 {
				startIdx = 1
			}
			if startIdx > len(lines) {
				startIdx = len(lines)
			}
			sByte := absOffsets[startIdx-1]

			endIdx := lc.EndLine + 1
			if endIdx < 1 {
				endIdx = 1
			}
			if endIdx > len(lines) {
				endIdx = len(lines)
			}
			eByte := absOffsets[endIdx-1]

			chunkContent := content[sByte:eByte]
			metas = append(metas, model.ChunkMeta{
				ID:        fmt.Sprintf("chunk_%03d", nextID),
				StartByte: sByte,
				EndByte:   eByte,
				Head30:    model.ComputeHead30(chunkContent),
				Tail30:    model.ComputeTail30(chunkContent),
				Hash:      model.ComputeChunkHash(chunkContent),
				Summary:   lc.Summary,
			})
			nextID++
		}
	}

	return metas
}
