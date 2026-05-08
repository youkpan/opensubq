package service

import (
	"fmt"
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

// WritePerFileSummary writes a single file's summary.xml
func WritePerFileSummary(dp *store.DataPaths, filePath, summary string) error {
	path := dp.GetFileSummaryPath(filePath)
	content := fmt.Sprintf("<file_summary path=\"%s\">%s</file_summary>\n", escapeXML(filePath), escapeXML(summary))
	return store.WriteFile(path, content)
}

// ReadPerFileSummary reads a single file's summary.xml
func ReadPerFileSummary(dp *store.DataPaths, filePath string) string {
	path := dp.GetFileSummaryPath(filePath)
	if !store.FileExists(path) {
		return ""
	}
	content, err := store.ReadFile(path)
	if err != nil {
		return ""
	}
	return content
}

// BuildGlobalSummary assembles all per-file summaries into global XML format
func BuildGlobalSummary(dp *store.DataPaths, registry *model.FileRegistry) string {
	var sb strings.Builder
	sb.WriteString("<files>\n")
	for filePath := range registry.Files {
		summary := ReadPerFileSummary(dp, filePath)
		if summary != "" {
			sb.WriteString(summary)
		}
	}
	sb.WriteString("</files>")
	return sb.String()
}

// CleanupFileChunks removes all data for a file
func CleanupFileChunks(dp *store.DataPaths, filePath string, outline *model.Outline) {
	// Delete the entire file storage directory
	dp.DeleteFileDir(filePath)
}

func escapeXML(s string) string {
	s = strings.ReplaceAll(s, "&", "&amp;")
	s = strings.ReplaceAll(s, "<", "&lt;")
	s = strings.ReplaceAll(s, ">", "&gt;")
	s = strings.ReplaceAll(s, "\"", "&quot;")
	return s
}
