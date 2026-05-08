package service

import (
	"fmt"
	"os"
	"sort"
	"strings"

	"file-chat/llm"
	"file-chat/model"
	"file-chat/store"
)

const retrievalSystemPrompt = `你是一个文档检索助手。以下是文档的索引大纲：

<outline>
%s
</outline>

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

	orderedIDs := parseChunkIDs(result, maxRetrieve)

	chunkMap := make(map[string]model.Chunk)
	for _, c := range outline.Chunks {
		chunkMap[c.ID] = c
	}

	var selected []model.Chunk
	for _, id := range orderedIDs {
		if c, ok := chunkMap[id]; ok {
			selected = append(selected, c)
		}
	}

	return selected, nil
}

// BuildContext reads selected chunks using byte offsets from chunks.json and builds final context string
func BuildContext(dp *store.DataPaths, selected []model.Chunk) (string, error) {
	if len(selected) == 0 {
		return "", nil
	}

	sorted := make([]model.Chunk, len(selected))
	copy(sorted, selected)
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].FilePath != sorted[j].FilePath {
			return sorted[i].FilePath < sorted[j].FilePath
		}
		return sorted[i].StartByte < sorted[j].StartByte
	})

	// Load chunks.json for each unique file to get byte offsets
	type fileData struct {
		metas map[string]model.ChunkMeta
	}
	fileMetaCache := make(map[string]*fileData)
	loadFileMetas := func(filePath string) *fileData {
		if fd, ok := fileMetaCache[filePath]; ok {
			return fd
		}
		cf, err := store.ReadChunksJSON(dp, filePath)
		if err != nil {
			return nil
		}
		metas := make(map[string]model.ChunkMeta)
		for _, m := range cf.Chunks {
			metas[m.ID] = m
		}
		fd := &fileData{metas: metas}
		fileMetaCache[filePath] = fd
		return fd
	}

	var sb strings.Builder
	sb.WriteString("<context>\n")

	prevFile := ""
	prevEndByte := int64(-1)

	for i, c := range sorted {
		needWrapper := i == 0 || c.FilePath != prevFile || c.StartByte > prevEndByte+1

		fd := loadFileMetas(c.FilePath)
		var chunkContent string
		if fd != nil {
			if meta, ok := fd.metas[c.ID]; ok {
				content, err := ReadChunkContent(dp, &meta, c.FilePath)
				if err == nil {
					expanded := expandContextByByte(dp, c.FilePath, c.StartByte, c.EndByte, content)
					chunkContent = expanded
				}
			}
		}
		if chunkContent == "" {
			// Fallback: read directly from source
			srcPath := GetSourcePath(dp, c.FilePath)
			data, err := os.ReadFile(srcPath)
			if err == nil && c.EndByte <= int64(len(data)) {
				chunkContent = string(data[c.StartByte:c.EndByte])
			}
		}

		if needWrapper {
			if i > 0 {
				fmt.Fprintf(&sb, "</%s>\n", sorted[i-1].ID)
			}
			fmt.Fprintf(&sb, "<%s>\n", c.ID)
		}

		sb.WriteString(chunkContent)
		sb.WriteString("\n")

		prevFile = c.FilePath
		prevEndByte = c.EndByte

		if i == len(sorted)-1 {
			fmt.Fprintf(&sb, "</%s>\n", c.ID)
		}
	}

	sb.WriteString("</context>")
	return sb.String(), nil
}

// GetSourcePath returns the path to read source content from:
// - plain text files: read directly from original path
// - document files: read from stored source copy
func GetSourcePath(dp *store.DataPaths, filePath string) string {
	if IsPlainTextFile(filePath) {
		return filePath
	}
	return dp.GetFileSourcePath(filePath)
}

// ReadChunkContent reads chunk content from source file using byte offsets
func ReadChunkContent(dp *store.DataPaths, meta *model.ChunkMeta, filePath string) (string, error) {
	srcPath := GetSourcePath(dp, filePath)
	f, err := os.Open(srcPath)
	if err != nil {
		return "", fmt.Errorf("open source %s: %w", srcPath, err)
	}
	defer f.Close()

	size := meta.EndByte - meta.StartByte
	buf := make([]byte, size)
	_, err = f.ReadAt(buf, meta.StartByte)
	if err != nil {
		return "", fmt.Errorf("read chunk bytes: %w", err)
	}
	return string(buf), nil
}

// expandContextByByte adds surrounding context from the source file using byte offsets
func expandContextByByte(dp *store.DataPaths, filePath string, startByte, endByte int64, chunkContent string) string {
	srcPath := GetSourcePath(dp, filePath)
	data, err := os.ReadFile(srcPath)
	if err != nil {
		return chunkContent
	}

	// Expand backward
	beforeStart := startByte
	beforeBytes := 0
	for beforeStart > 0 {
		beforeStart--
		beforeBytes++
		if beforeBytes > 1200 {
			break
		}
		if beforeBytes > 500 && data[beforeStart] == '\n' {
			break
		}
		if beforeBytes > 800 {
			ch := data[beforeStart]
			if ch == '\n' || ch == '.' || (beforeStart >= 3 && data[beforeStart] == 0x80 && data[beforeStart-1] == 0x80 && data[beforeStart-2] == 0xe3 && data[beforeStart-3] == 0xe3) {
				// Chinese period 。 is e3 80 82 in UTF-8
				break
			}
		}
	}

	// Expand forward
	afterEnd := endByte
	afterBytes := 0
	for afterEnd < int64(len(data)) {
		afterBytes++
		afterEnd++
		if afterBytes > 1200 {
			break
		}
		if afterBytes > 500 && data[afterEnd-1] == '\n' {
			break
		}
		if afterBytes > 800 {
			ch := data[afterEnd-1]
			if ch == '\n' || ch == '.' {
				break
			}
		}
	}

	var sb strings.Builder
	if beforeStart < startByte {
		sb.Write(data[beforeStart:startByte])
		sb.WriteByte('\n')
	}
	sb.WriteString(chunkContent)
	if endByte < afterEnd {
		sb.WriteByte('\n')
		sb.Write(data[endByte:afterEnd])
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
