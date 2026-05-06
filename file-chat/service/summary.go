package service

import (
	"fmt"
	"os"
	"strings"

	"file-chat/llm"
	"file-chat/model"
	"file-chat/store"
)

const summaryPrompt = `你是一个文档摘要助手。请为以下文件生成一段简洁的摘要（50-150字），概括文件的主要内容。

<file_path>%s</file_path>
<chunk_summaries>
%s
</chunk_summaries>

请直接输出摘要文本，不要输出其他内容。`

// GenerateFileSummary uses LLM to generate a file-level summary from chunk summaries
func GenerateFileSummary(client *llm.Client, chunks []model.Chunk) (string, error) {
	if len(chunks) == 0 {
		return "", nil
	}
	var summaries []string
	for _, c := range chunks {
		summaries = append(summaries, c.Summary)
	}
	allSummaries := strings.Join(summaries, "\n")
	prompt := fmt.Sprintf(summaryPrompt, chunks[0].FilePath, allSummaries)
	return client.ChatSimple("", prompt)
}

// ReadFilesSummary reads files_summary.xml content
func ReadFilesSummary(path string) string {
	if !store.FileExists(path) {
		return ""
	}
	content, err := store.ReadFile(path)
	if err != nil {
		return ""
	}
	return content
}

// AppendFileSummary adds a file entry to files_summary.xml
func AppendFileSummary(path, filePath, summary string) error {
	entry := fmt.Sprintf("  <file path=\"%s\">%s</file>\n", escapeXML(filePath), escapeXML(summary))

	if !store.FileExists(path) {
		content := "<files>\n" + entry + "</files>\n"
		return store.WriteFile(path, content)
	}

	// Read existing, insert before </files>
	content, err := store.ReadFile(path)
	if err != nil {
		return err
	}
	// Remove old entry for this file if exists
	content = removeFileEntry(content, filePath)
	// Insert new entry before </files>
	content = strings.Replace(content, "</files>", entry+"</files>", 1)
	return store.WriteFile(path, content)
}

// RemoveFileFromSummary removes a file entry from files_summary.xml
func RemoveFileFromSummary(path, filePath string) error {
	if !store.FileExists(path) {
		return nil
	}
	content, err := store.ReadFile(path)
	if err != nil {
		return err
	}
	content = removeFileEntry(content, filePath)
	return store.WriteFile(path, content)
}

// removeFileEntry removes a <file path="...">...</file> line from XML content
func removeFileEntry(content, filePath string) string {
	lines := strings.Split(content, "\n")
	var result []string
	prefix := fmt.Sprintf("<file path=\"%s\">", escapeXML(filePath))
	for _, line := range lines {
		if !strings.Contains(line, prefix) {
			result = append(result, line)
		}
	}
	return strings.Join(result, "\n")
}

func escapeXML(s string) string {
	s = strings.ReplaceAll(s, "&", "&amp;")
	s = strings.ReplaceAll(s, "<", "&lt;")
	s = strings.ReplaceAll(s, ">", "&gt;")
	s = strings.ReplaceAll(s, "\"", "&quot;")
	return s
}

// CleanupFileChunks removes all data for a file from the job
func CleanupFileChunks(jp *store.JobPaths, filePath string, outline *model.Outline) {
	// 1. Delete chunk files
	for _, c := range outline.Chunks {
		if c.FilePath == filePath {
			chunkPath := store.GetChunkFilePath(jp.Chunks, c.FilePath, c.ID)
			os.Remove(chunkPath)
		}
	}

	// 2. Rewrite global outline without this file's chunks
	RemoveChunksForFile(jp.Outline, filePath, outline)

	// 3. Delete per-file outline
	perFilePath := store.GetPerFileOutlinePath(jp.OutlinesDir, filePath)
	os.Remove(perFilePath)

	// 4. Remove from files_summary.xml
	RemoveFileFromSummary(jp.FilesSummary, filePath)

	// 5. Delete source file
	sourcePath := store.GetSourceFilePath(jp.Sources, filePath)
	os.Remove(sourcePath)
}
