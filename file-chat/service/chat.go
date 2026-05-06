package service

import (
	"crypto/md5"
	"fmt"
	"log"
	"strings"
	"sync"
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

// fileTask holds a file to process and its result
type fileTask struct {
	filePath string
	chunks   []model.Chunk
	err      error
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

	// 2. Extract @paths and @全部
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

	if len(allPaths) == 0 && !hasAll {
		return cleanMessages, nil
	}

	// 3. Init job
	jp, err := store.InitJob(s.JobsDir, jobID)
	if err != nil {
		return nil, fmt.Errorf("init job: %w", err)
	}

	// 4. Load registry and outline
	registry, err := store.ReadFileRegistry(s.JobsDir)
	if err != nil {
		log.Printf("read registry: %v", err)
		registry = &model.FileRegistry{Files: make(map[string]*model.FileEntry)}
	}

	outline, err := ReadOutline(jp.Outline)
	if err != nil {
		return nil, fmt.Errorf("read outline: %w", err)
	}

	// 5. Resolve all paths to file lists, filter unchanged
	var filesToProcess []string
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
			entry := registry.Files[f]
			if entry != nil && !IsFileChanged(f, entry) {
				continue // unchanged, skip
			}
			// File changed: cleanup old data
			if entry != nil {
				log.Printf("file changed, re-processing: %s", f)
				CleanupFileChunks(jp, f, outline)
				outline, _ = ReadOutline(jp.Outline)
			}
			filesToProcess = append(filesToProcess, f)
		}
	}

	// 6. Process files in parallel (up to 20 concurrent)
	if len(filesToProcess) > 0 {
		tasks := make([]fileTask, len(filesToProcess))
		sem := make(chan struct{}, maxConcurrency)
		var wg sync.WaitGroup

		for i, f := range filesToProcess {
			wg.Add(1)
			sem <- struct{}{}
			go func(idx int, filePath string) {
				defer wg.Done()
				defer func() { <-sem }()
				chunks, err := ProcessFile(s.Client, jp, filePath, s.MarkitdownCmd, s.SmallFileSize, outline)
				tasks[idx] = fileTask{filePath: filePath, chunks: chunks, err: err}
			}(i, f)
		}
		wg.Wait()

		// 7. Sequential: assemble results into outline + registry
		registryChanged := false
		for _, t := range tasks {
			if t.err != nil {
				log.Printf("process file %s: %v", t.filePath, t.err)
				continue
			}
			if len(t.chunks) == 0 {
				continue
			}

			// Append to global outline
			if err := AppendChunks(jp.Outline, t.chunks); err != nil {
				log.Printf("append chunks: %v", err)
				continue
			}
			outline.Chunks = append(outline.Chunks, t.chunks...)

			// Write per-file outline
			if err := WritePerFileOutline(jp.OutlinesDir, t.chunks); err != nil {
				log.Printf("write per-file outline: %v", err)
			}

			// Generate file summary
			summary, err := GenerateFileSummary(s.Client, t.chunks)
			if err != nil {
				log.Printf("generate summary: %v", err)
				summary = fmt.Sprintf("文件 %s 的内容", t.filePath)
			}
			if err := AppendFileSummary(jp.FilesSummary, t.filePath, summary); err != nil {
				log.Printf("append file summary: %v", err)
			}

			// Update registry
			hash, _ := store.ComputeFileHash(t.filePath)
			modTime, _ := store.GetFileModTime(t.filePath)
			size, _ := store.GetFileSize(t.filePath)
			registry.Files[t.filePath] = &model.FileEntry{
				Hash:        hash,
				ModTime:     modTime,
				Size:        size,
				ProcessedAt: time.Now().Format("2006-01-02T15:04:05Z07:00"),
			}
			registryChanged = true
		}

		if registryChanged {
			store.WriteFileRegistry(s.JobsDir, registry)
		}
	}

	// 8. Reload outline for retrieval
	outline, err = ReadOutline(jp.Outline)
	if err != nil {
		return nil, fmt.Errorf("reload outline: %w", err)
	}

	if len(outline.Chunks) == 0 {
		return cleanMessages, nil
	}

	// 9. Get query
	var query string
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			query = CleanPaths(messages[i].Content)
			break
		}
	}

	// 10. Retrieve
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

	// 11. Build context
	context, err := BuildContext(jp.Chunks, jp.Sources, selected)
	if err != nil {
		log.Printf("build context: %v", err)
		return cleanMessages, nil
	}

	// 12. Construct final messages
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
