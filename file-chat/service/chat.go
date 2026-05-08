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
	DataDir       string
	MarkitdownCmd string
	ChunkTokens   int
	MaxRetrieve   int
	SmallFileSize int64
}

func NewChatService(client *llm.Client, dataDir, markitdownCmd string, chunkTokens, maxRetrieve int, smallFileSize int64) *ChatService {
	return &ChatService{
		Client:        client,
		DataDir:       dataDir,
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
	metas    []model.ChunkMeta
	err      error
}

// ProcessRequest handles a full chat request: extract paths, process files, build context
func (s *ChatService) ProcessRequest(messages []model.Message, conversationID string) ([]llm.ChatMessage, error) {
	startTime := time.Now()

	// 1. Determine conversation ID
	if conversationID == "" {
		for _, m := range messages {
			if m.Role == "user" {
				conversationID = fmt.Sprintf("%x", md5.Sum([]byte(m.Content)))[:6]
				break
			}
		}
	}
	if conversationID == "" {
		conversationID = "default"
	}
	log.Printf("[Step 1] conversationID=%s", conversationID)

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

	// 3. Initialize data directory (needed for chat-files.json)
	dp, err := store.InitDataDir(s.DataDir)
	if err != nil {
		return nil, fmt.Errorf("init data dir: %w", err)
	}
	log.Printf("[Step 3] 初始化数据目录: %s", dp.DataDir)

	if len(allPaths) == 0 && !hasAll {
		// No new file references — check if conversation has associated files
		chatFiles, _ := store.ReadChatFiles(dp, conversationID)
		if len(chatFiles.Files) == 0 {
			log.Printf("[Step 2] 无文件引用且对话无关联文件，透传请求")
			return cleanMessages, nil
		}
		log.Printf("[Step 2] 无新文件引用，使用对话关联文件: %d 个", len(chatFiles.Files))
	}

	// 4. Load global registry
	registry, err := store.ReadFileRegistry(dp)
	if err != nil {
		log.Printf("read registry: %v", err)
		registry = &model.FileRegistry{Files: make(map[string]*model.FileEntry)}
	}
	log.Printf("[Step 4] 加载注册表: %d 个文件", len(registry.Files))

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
			store.AddFolderToChat(dp, conversationID, absPath)
		} else {
			files = []string{absPath}
		}

		for _, f := range files {
			entry := registry.Files[f]
			if entry != nil && !IsFileChanged(f, entry) {
				filesSkipped = append(filesSkipped, f)
				continue
			}
			// File changed: will do incremental update
			if entry != nil {
				log.Printf("[Step 5] 文件已变更，增量更新: %s", f)
			}
			filesToProcess = append(filesToProcess, f)
		}
	}

	// @全部: also include files from registry that this conversation hasn't referenced yet
	if hasAll {
		chatFiles, _ := store.ReadChatFiles(dp, conversationID)
		chatFileSet := make(map[string]bool)
		for _, f := range chatFiles.Files {
			chatFileSet[f] = true
		}
		for f := range registry.Files {
			if !chatFileSet[f] {
				filesSkipped = append(filesSkipped, f)
			}
		}
	}

	log.Printf("[Step 5] 文件清单: 待处理=%d, 跳过(未变更)=%d", len(filesToProcess), len(filesSkipped))
	for _, f := range filesToProcess {
		log.Printf("[Step 5]   待处理: %s", f)
	}
n		// Add skipped (unchanged) files to chat-files.json so they are included in retrieval
		for _, f := range filesSkipped {
			store.AddFileToChat(dp, conversationID, f)
		}

	// 6. Process files in parallel
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
				var chunks []model.Chunk
				var metas []model.ChunkMeta
				var procErr error
				lockErr := store.WithFileLock(filePath, func() error {
					// Check if this is an incremental update (file exists in registry)
					if registry.Files[filePath] != nil {
						chunks, metas, procErr = IncrementalUpdateFile(s.Client, dp, filePath, s.MarkitdownCmd, s.SmallFileSize)
					} else {
						chunks, metas, procErr = ProcessFile(s.Client, dp, filePath, s.MarkitdownCmd, s.SmallFileSize, &model.Outline{})
					}
					return nil
				})
				if lockErr != nil {
					procErr = lockErr
				}
				elapsed := time.Since(fileStart)
				if procErr != nil {
					log.Printf("[Step 6]   文件处理失败 [%s]: %v (%.1fs)", filePath, procErr, elapsed.Seconds())
				} else {
					ids := make([]string, len(chunks))
					for i, c := range chunks {
						ids[i] = c.ID
					}
					log.Printf("[Step 6]   文件处理完成 [%s]: %d chunks %v (%.1fs)", filePath, len(chunks), ids, elapsed.Seconds())
				}
				tasks[idx] = fileTask{filePath: filePath, chunks: chunks, metas: metas, err: procErr}
			}(i, f)
		}
		wg.Wait()
		log.Printf("[Step 6] 并行处理完成 (%.1fs)", time.Since(processStart).Seconds())

		// 7. Sequential: assemble results
		log.Printf("[Step 7] 组装结果: per-file outline + summary + chunks.json + registry")
		registryChanged := false
		for _, t := range tasks {
			if t.err != nil || len(t.chunks) == 0 {
				continue
			}

			ids := make([]string, len(t.chunks))
			for i, c := range t.chunks {
				ids[i] = c.ID
			}
			log.Printf("[Step 7]   组装: %s → chunks %v", t.filePath, ids)

			// Write chunks.json
			cf := &model.ChunksFile{
				FilePath: t.filePath,
				Chunks:   t.metas,
			}
			if err := store.WriteChunksJSON(dp, t.filePath, cf); err != nil {
				log.Printf("write chunks.json: %v", err)
				continue
			}

			// Build outline chunks from metas (metas include sub-chunk IDs)
			outlineChunks := make([]model.Chunk, len(t.metas))
			for i, m := range t.metas {
				outlineChunks[i] = model.Chunk{
					ID:        m.ID,
					FilePath:  t.filePath,
					Summary:   m.Summary,
					StartByte: m.StartByte,
					EndByte:   m.EndByte,
				}
			}

			// Write per-file outline (2-field format, with sub-chunk IDs)
			if err := WritePerFileOutline(dp, outlineChunks); err != nil {
				log.Printf("write per-file outline: %v", err)
			}

			// Generate file summary and write to per-file summary.xml
			summary, err := GenerateFileSummary(s.Client, outlineChunks)
			if err != nil {
				log.Printf("generate summary: %v", err)
				summary = fmt.Sprintf("文件 %s 的内容", t.filePath)
			}
			log.Printf("[Step 7]   摘要: %s → %s", t.filePath, summary)
			if err := WritePerFileSummary(dp, t.filePath, summary); err != nil {
				log.Printf("write per-file summary: %v", err)
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
				ChunkCount:  len(t.metas),
				Summary:     summary,
			}
			registryChanged = true

			// Add file to chat-files.json
			store.AddFileToChat(dp, conversationID, t.filePath)
		}

		if registryChanged {
			store.WriteFileRegistry(dp, registry)
			log.Printf("[Step 7] 注册表已更新: %d 个文件", len(registry.Files))
		}
	}

	// 8. Build global outline from per-file outlines
	outline, err := BuildGlobalOutline(dp, registry)
	if err != nil {
		return nil, fmt.Errorf("build global outline: %w", err)
	}
	// Load byte offsets for all chunks
	for filePath := range registry.Files {
		cf, err := store.ReadChunksJSON(dp, filePath)
		if err != nil {
			continue
		}
		for i := range outline.Chunks {
			if outline.Chunks[i].FilePath != filePath {
				continue
			}
			for _, m := range cf.Chunks {
				if m.ID == outline.Chunks[i].ID {
					outline.Chunks[i].StartByte = m.StartByte
					outline.Chunks[i].EndByte = m.EndByte
					break
				}
			}
		}
	}
	log.Printf("[Step 8] 全局大纲: %d 个 chunks", len(outline.Chunks))

	// Filter outline to conversation scope if not @全部
	if !hasAll {
		chatFiles, _ := store.ReadChatFiles(dp, conversationID)
		chatFileSet := make(map[string]bool)
		for _, f := range chatFiles.Files {
			chatFileSet[f] = true
		}
		for _, p := range allPaths {
			chatFileSet[p] = true
		}
		var filtered []model.Chunk
		for _, c := range outline.Chunks {
			if chatFileSet[c.FilePath] {
				filtered = append(filtered, c)
			}
		}
		outline.Chunks = filtered
		log.Printf("[Step 8] 对话范围大纲: %d 个 chunks (过滤后)", len(outline.Chunks))
	}

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
		selected, err = HandleAllFiles(s.Client, dp, registry, query, s.MaxRetrieve)
		if err != nil {
			log.Printf("[Step 10] @全部两级检索失败: %v", err)
		}
		if len(selected) == 0 {
			selected, err = HandleAllFilesFromOutline(s.Client, dp, registry, query, s.MaxRetrieve)
			if err != nil {
				log.Printf("[Step 10] @全部大纲检索失败: %v", err)
			}
		}
		log.Printf("[Step 10] @全部检索: %d 个 chunks", len(selected))
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
		selectedIDs[i] = fmt.Sprintf("%s(%s:%d-%d)", c.ID, filepath.Base(c.FilePath), c.StartByte, c.EndByte)
	}
	log.Printf("[Step 10] 选中 chunks: %v", selectedIDs)

	// 11. Build context
	context, err := BuildContext(dp, selected)
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
			Content: fmt.Sprintf("以下是相关文档内容：\n\n<document_content>\n%s\n</document_content>", context),
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
