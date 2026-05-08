package model

import (
	"fmt"
	"strconv"
	"strings"
)

// Chunk represents a semantic chunk in the outline
type Chunk struct {
	ID        string // e.g. chunk_001
	FilePath  string // absolute path
	Summary   string // 30~150 chars
	StartLine int
	EndLine   int
}

// Outline represents the full outline
type Outline struct {
	Chunks []Chunk
}

// ParseOutline parses outline content from string
func ParseOutline(content string) *Outline {
	lines := strings.Split(strings.TrimSpace(content), "\n")
	var chunks []Chunk
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, "|", 5)
		if len(parts) != 5 {
			continue
		}
		start, err1 := strconv.Atoi(parts[3])
		end, err2 := strconv.Atoi(parts[4])
		if err1 != nil || err2 != nil {
			continue
		}
		chunks = append(chunks, Chunk{
			ID:        parts[0],
			FilePath:  parts[1],
			Summary:   parts[2],
			StartLine: start,
			EndLine:   end,
		})
	}
	return &Outline{Chunks: chunks}
}

// String serializes outline to string
func (o *Outline) String() string {
	var sb strings.Builder
	for _, c := range o.Chunks {
		sb.WriteString(fmt.Sprintf("%s|%s|%s|%d|%d\n", c.ID, c.FilePath, c.Summary, c.StartLine, c.EndLine))
	}
	return sb.String()
}

// ParsedPath holds extracted @path info
type ParsedPath struct {
	Original string // raw path from message
	IsDir    bool
}

// FileContent holds a file's content and metadata
type FileContent struct {
	Path    string
	Content string
	Size    int64
	Lines   []string // content split by lines
}

// FileEntry tracks a processed file in the global registry
type FileEntry struct {
	Hash        string `json:"hash"`
	ModTime     string `json:"mod_time"`
	Size        int64  `json:"size"`
	ProcessedAt string `json:"processed_at"`
	ChunkCount  int    `json:"chunk_count"`
	Summary     string `json:"summary"`
}

// FileRegistry is the global file registry stored in data/files.json
type FileRegistry struct {
	Files map[string]*FileEntry `json:"files"`
}

// ChatFiles tracks which files and folders are associated with a conversation
type ChatFiles struct {
	ConversationID string   `json:"conversation_id"`
	Files          []string `json:"files"`
	Folders        []string `json:"folders"`
	UpdatedAt      string   `json:"updated_at"`
}
