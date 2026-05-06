package service

import (
	"crypto/md5"
	"fmt"
	"log"
	"strings"
	"time"

	"file-chat/llm"
	"file-chat/model"
	"file-chat/store"
)

type ChatService struct {
	Client        *llm.Client
	JobsDir       string
	MarkitdownCmd string
	ChunkTokens   int
	MaxRetrieve   int
	SmallFileSize int64
}

func NewChatService(client *llm.Client, jobsDir, markitdownCmd string, chunkTokens, maxRetrieve int, smallFileSize int64) *ChatService {
	return &ChatService{
		Client:        client,
		JobsDir:       jobsDir,
		MarkitdownCmd: markitdownCmd,
		ChunkTokens:   chunkTokens,
		MaxRetrieve:   maxRetrieve,
		SmallFileSize: smallFileSize,
	}
}

// ProcessRequest handles a full chat request: extract paths, process files, build context
func (s *ChatService) ProcessRequest(messages []model.Message, conversationID string) ([]llm.ChatMessage, error) {
	// 1. Determine job ID
	jobID := conversationID
	if jobID == "" {
		for _, m := range messages {
			if m.Role == "user" {
				jobID = fmt.Sprintf("%x", md5.Sum([]byte(m.Content)))[:6]
				break
			}
		}
	}
	if jobID == "" {
		jobID = "default"
	}

	// 2. Extract @paths and @全部 from all user messages
	var allPaths []string
	var hasAll bool
	var cleanMessages []llm.ChatMessage
	for _, m := range messages {
		paths, h := ExtractPaths(m.Content)
		if h {
			hasAll = true
		}
		allPaths = append(allPaths, paths...)
		cleanContent := CleanPaths(m.Content)
		cleanMessages = append(cleanMessages, llm.ChatMessage{
			Role:    m.Role,
			Content: cleanContent,
		})
	}

	// 3. If no paths and no @全部, return clean messages directly
	if len(allPaths) == 0 && !hasAll {
		return cleanMessages, nil
	}

	// 4. Init job
	jp, err := store.InitJob(s.JobsDir, jobID)
	if err != nil {
		return nil, fmt.Errorf("init job: %w", err)
	}

	// 5. Load global file registry
	registry, err := store.ReadFileRegistry(s.JobsDir)
	if err != nil {
		log.Printf("read registry: %v", err)
		registry = &model.FileRegistry{Files: make(map[string]*model.FileEntry)}
	}

	// 6. Load outline
	outline, err := ReadOutline(jp.Outline)
	if err != nil {
		return nil, fmt.Errorf("read outline: %w", err)
	}

	// 7. Process each path with change detection
	registryChanged := false
	for _, rawPath := range allPaths {
		absPath, isDir := ResolvePath(rawPath)
		if absPath == "" {
			log.Printf("path not found: %s", rawPath)
			continue
		}

		var files []string
		if isDir {
			files = ListFiles(absPath)
		} else {
			files = []string{absPath}
		}

		for _, f := range files {
			relPath := f

			// Check registry for change detection
			entry := registry.Files[relPath]
			if entry != nil && !IsFileChanged(f, entry) {
				// File unchanged, skip processing
				continue
			}

			// File changed or new: cleanup old data if changed
			if entry != nil {
				log.Printf("file changed, re-processing: %s", relPath)
				CleanupFileChunks(jp, relPath, outline)
				// Reload outline after cleanup
				outline, _ = ReadOutline(jp.Outline)
			}

			// Process file
			chunks, err := ProcessFile(s.Client, jp, f, s.MarkitdownCmd, s.SmallFileSize, outline)
			if err != nil {
				log.Printf("process file %s: %v", f, err)
				continue
			}
			if len(chunks) == 0 {
				continue
			}

			// Append to global outline
			if err := AppendChunks(jp.Outline, chunks); err != nil {
				log.Printf("append chunks: %v", err)
				continue
			}
			outline.Chunks = append(outline.Chunks, chunks...)

			// Write per-file outline
			if err := WritePerFileOutline(jp.OutlinesDir, chunks); err != nil {
				log.Printf("write per-file outline: %v", err)
			}

			// Generate file summary and update files_summary.xml
			summary, err := GenerateFileSummary(s.Client, chunks)
			if err != nil {
				log.Printf("generate summary: %v", err)
				summary = fmt.Sprintf("文件 %s 的内容", relPath)
			}
			if err := AppendFileSummary(jp.FilesSummary, relPath, summary); err != nil {
				log.Printf("append file summary: %v", err)
			}

			// Update registry
			hash, _ := store.ComputeFileHash(f)
			modTime, _ := store.GetFileModTime(f)
			size, _ := store.GetFileSize(f)
			registry.Files[relPath] = &model.FileEntry{
				Hash:        hash,
				ModTime:     modTime,
				Size:        size,
				ProcessedAt: time.Now().Format("2006-01-02T15:04:05Z07:00"),
			}
			registryChanged = true
		}
	}

	// 8. Save registry if changed
	if registryChanged {
		if err := store.WriteFileRegistry(s.JobsDir, registry); err != nil {
			log.Printf("write registry: %v", err)
		}
	}

	// 9. Reload outline for retrieval
	outline, err = ReadOutline(jp.Outline)
	if err != nil {
		return nil, fmt.Errorf("reload outline: %w", err)
	}

	if len(outline.Chunks) == 0 {
		return cleanMessages, nil
	}

	// 10. Get last user message as query
	var query string
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			query = CleanPaths(messages[i].Content)
			break
		}
	}

	// 11. Retrieve relevant chunks
	var selected []model.Chunk
	if hasAll {
		selected, err = HandleAllFiles(s.Client, jp, query, s.MaxRetrieve)
		if err != nil {
			log.Printf("handle all files: %v", err)
		}
	}
	if len(selected) == 0 && len(allPaths) > 0 {
		selected, err = RetrieveChunks(s.Client, outline, query, s.MaxRetrieve)
		if err != nil {
			log.Printf("retrieve chunks: %v", err)
			return cleanMessages, nil
		}
	}
	if len(selected) == 0 {
		return cleanMessages, nil
	}

	// 12. Build context
	context, err := BuildContext(jp.Chunks, jp.Sources, selected)
	if err != nil {
		log.Printf("build context: %v", err)
		return cleanMessages, nil
	}

	// 13. Construct final messages with context
	finalMessages := []llm.ChatMessage{
		{
			Role: "system",
			Content: "你是一个文档问答助手。请根据提供的文档内容回答用户的问题。" +
				"如果文档中没有相关信息，请如实说明。" +
				"引用文档内容时请注明来源片段。",
		},
	}

	if context != "" {
		finalMessages = append(finalMessages, llm.ChatMessage{
			Role:    "user",
			Content: fmt.Sprintf("以下是相关文档内容：\n\n%s", context),
		})
		finalMessages = append(finalMessages, llm.ChatMessage{
			Role:    "assistant",
			Content: "好的，我已经阅读了文档内容，请提出你的问题。",
		})
	}

	for _, m := range cleanMessages {
		if strings.TrimSpace(m.Content) == "" {
			continue
		}
		finalMessages = append(finalMessages, m)
	}

	return finalMessages, nil
}
