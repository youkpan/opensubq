package service

import (
	"crypto/md5"
	"fmt"
	"log"
	"path/filepath"
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
	startTime := time.Now()

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
	log.Printf("[Step 1] jobID=%s", jobID)

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
	log.Printf("[Step 2] 提取路径: paths=%v, hasAll=%v", allPaths, hasAll)

	if len(allPaths) == 0 && !hasAll {
		log.Printf("[Step 2] 无文件引用，透传请求")
		return cleanMessages, nil
	}

	// 3. Init job
	jp, err := store.InitJob(s.JobsDir, jobID)
	if err != nil {
		return nil, fmt.Errorf("init job: %w", err)
	}
	log.Printf("[Step 3] 初始化 job: %s", jp.Root)

	// 4. Load registry and outline
	registry, err := store.ReadFileRegistry(s.JobsDir)
	if err != nil {
		log.Printf("read registry: %v", err)
		registry = &model.FileRegistry{Files: make(map[string]*model.FileEntry)}
	}
	log.Printf("[Step 4] 加载注册表: 已有 %d 个文件, 已有 %d 条记录", len(registry.Files), len(registry.Files))

	outline, err := ReadOutline(jp.Outline)
	if err != nil {
		return nil, fmt.Errorf("read outline: %w", err)
	}
	log.Printf("[Step 4] 加载大纲: %d 个已有 chunks", len(outline.Chunks))

	// 5. Resolve all paths to file lists, filter unchanged
	var filesToProcess []string
	var filesSkipped []string
	for _, rawPath := range allPaths {
		absPath, isDir := ResolvePath(rawPath)
		if absPath == "" {
			log.Printf("[Step 5] 路径未找到: %s", rawPath)
			continue
		}

		var files []string
		if isDir {
			files = ListFiles(absPath)
			log.Printf("[Step 5] 目录解析: %s → %d 个文件", absPath, len(files))
		} else {
			files = []string{absPath}
		}

		for _, f := range files {
			entry := registry.Files[f]
			if entry != nil && !IsFileChanged(f, entry) {
				filesSkipped = append(filesSkipped, f)
				continue // unchanged, skip
			}
			// File changed: cleanup old data
			if entry != nil {
				log.Printf("[Step 5] 文件已变更，重新处理: %s", f)
				CleanupFileChunks(jp, f, outline)
				outline, _ = ReadOutline(jp.Outline)
			}
			filesToProcess = append(filesToProcess, f)
		}
	}
	log.Printf("[Step 5] 文件清单: 待处理=%d, 跳过(未变更)=%d", len(filesToProcess), len(filesSkipped))
	for _, f := range filesToProcess {
		log.Printf("[Step 5]   待处理: %s", f)
	}
	for _, f := range filesSkipped {
		log.Printf("[Step 5]   跳过: %s", f)
	}

	// 6. Process files in parallel (up to 20 concurrent)
	if len(filesToProcess) > 0 {
		log.Printf("[Step 6] 开始并行处理 %d 个文件 (并发上限=%d)", len(filesToProcess), maxConcurrency)
		processStart := time.Now()
		tasks := make([]fileTask, len(filesToProcess))
		sem := make(chan struct{}, maxConcurrency)
		var wg sync.WaitGroup

		for i, f := range filesToProcess {
			wg.Add(1)
			sem <- struct{}{}
			go func(idx int, filePath string) {
				defer wg.Done()
				defer func() { <-sem }()
				fileStart := time.Now()
				chunks, err := ProcessFile(s.Client, jp, filePath, s.MarkitdownCmd, s.SmallFileSize, outline)
				elapsed := time.Since(fileStart)
				if err != nil {
					log.Printf("[Step 6]   文件处理失败 [%s]: %v (%.1fs)", filePath, err, elapsed.Seconds())
				} else {
					ids := make([]string, len(chunks))
					for i, c := range chunks {
						ids[i] = c.ID
					}
					log.Printf("[Step 6]   文件处理完成 [%s]: %d chunks %v (%.1fs)", filePath, len(chunks), ids, elapsed.Seconds())
				}
				tasks[idx] = fileTask{filePath: filePath, chunks: chunks, err: err}
			}(i, f)
		}
		wg.Wait()
		log.Printf("[Step 6] 并行处理完成 (%.1fs)", time.Since(processStart).Seconds())

		// 7. Sequential: assemble results into outline + registry
		log.Printf("[Step 7] 组装结果: outline + per-file outline + summary + registry")
		registryChanged := false
		for _, t := range tasks {
			if t.err != nil {
				continue
			}
			if len(t.chunks) == 0 {
				continue
			}

			ids := make([]string, len(t.chunks))
			for i, c := range t.chunks {
				ids[i] = c.ID
			}
			log.Printf("[Step 7]   组装: %s → chunks %v", t.filePath, ids)

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
			log.Printf("[Step 7]   摘要: %s → %s", t.filePath, summary)
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
			log.Printf("[Step 7] 注册表已更新: %d 个文件", len(registry.Files))
		}
	}

	// 8. Reload outline for retrieval
	outline, err = ReadOutline(jp.Outline)
	if err != nil {
		return nil, fmt.Errorf("reload outline: %w", err)
	}
	log.Printf("[Step 8] 重载大纲: %d 个 chunks", len(outline.Chunks))

	if len(outline.Chunks) == 0 {
		log.Printf("[Step 8] 大纲为空，透传请求")
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
	log.Printf("[Step 9] 查询: %q", query)

	// 10. Retrieve
	log.Printf("[Step 10] 开始检索 (hasAll=%v, maxRetrieve=%d)", hasAll, s.MaxRetrieve)
	var selected []model.Chunk
	if hasAll {
		selected, err = HandleAllFiles(s.Client, jp, query, s.MaxRetrieve)
		if err != nil {
			log.Printf("[Step 10] @全部检索失败: %v", err)
		} else {
			log.Printf("[Step 10] @全部检索: %d 个 chunks", len(selected))
		}
	}
	if len(selected) == 0 && len(allPaths) > 0 {
		selected, err = RetrieveChunks(s.Client, outline, query, s.MaxRetrieve)
		if err != nil {
			log.Printf("[Step 10] 路径检索失败: %v", err)
			return cleanMessages, nil
		}
		log.Printf("[Step 10] 路径检索: %d 个 chunks", len(selected))
	}
	if len(selected) == 0 {
		log.Printf("[Step 10] 无检索结果，透传请求")
		return cleanMessages, nil
	}
	selectedIDs := make([]string, len(selected))
	for i, c := range selected {
		selectedIDs[i] = fmt.Sprintf("%s(%s:%d-%d)", c.ID, filepath.Base(c.FilePath), c.StartLine, c.EndLine)
	}
	log.Printf("[Step 10] 选中 chunks: %v", selectedIDs)

	// 11. Build context
	context, err := BuildContext(jp.Chunks, jp.Sources, selected)
	if err != nil {
		log.Printf("[Step 11] 构建上下文失败: %v", err)
		return cleanMessages, nil
	}
	log.Printf("[Step 11] 上下文构建完成: %d 字符, %d bytes", len(context), len(context))

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

	log.Printf("[Step 12] 最终消息: %d 条, 总耗时 %.1fs", len(finalMessages), time.Since(startTime).Seconds())

	return finalMessages, nil
}
