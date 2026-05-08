package model

import (
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
)

// ChunkMeta is stored in chunks.json — lightweight metadata for each chunk
type ChunkMeta struct {
	ID        string `json:"id"`         // "chunk_001", "chunk_071-1"
	StartByte int64  `json:"start_byte"` // inclusive byte offset in source
	EndByte   int64  `json:"end_byte"`   // exclusive byte offset in source
	Head30    string `json:"head_30"`    // first 30 bytes as hex fingerprint
	Tail30    string `json:"tail_30"`    // last 30 bytes as hex fingerprint
	Hash      string `json:"hash"`       // MD5 of chunk content
	Summary   string `json:"summary"`    // 30-150 char summary
}

// ChunksFile is the top-level chunks.json structure
type ChunksFile struct {
	FilePath string      `json:"file_path"`
	Chunks   []ChunkMeta `json:"chunks"`
}

// Chunk is the runtime struct used during retrieval
type Chunk struct {
	ID        string
	FilePath  string
	Summary   string
	StartByte int64
	EndByte   int64
}

// ToMeta converts a runtime Chunk + content into a ChunkMeta
func (c *Chunk) ToMeta(content []byte) ChunkMeta {
	return ChunkMeta{
		ID:        c.ID,
		StartByte: c.StartByte,
		EndByte:   c.EndByte,
		Head30:    ComputeHead30(content),
		Tail30:    ComputeTail30(content),
		Hash:      ComputeChunkHash(content),
		Summary:   c.Summary,
	}
}

// Outline represents the full outline
type Outline struct {
	Chunks []Chunk
}

// ParseOutline parses outline content from string (2-field format: chunk_id|summary)
func ParseOutline(content string) *Outline {
	lines := strings.Split(strings.TrimSpace(content), "\n")
	var chunks []Chunk
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, "|", 2)
		if len(parts) != 2 {
			continue
		}
		chunks = append(chunks, Chunk{
			ID:      parts[0],
			Summary: parts[1],
		})
	}
	return &Outline{Chunks: chunks}
}

// String serializes outline to string (2-field format)
func (o *Outline) String() string {
	var sb strings.Builder
	for _, c := range o.Chunks {
		sb.WriteString(fmt.Sprintf("%s|%s\n", c.ID, c.Summary))
	}
	return sb.String()
}

// ParseChunksFile parses chunks.json content
func ParseChunksFile(data []byte) (*ChunksFile, error) {
	var cf ChunksFile
	if err := json.Unmarshal(data, &cf); err != nil {
		return nil, err
	}
	return &cf, nil
}

// SerializeChunksFile serializes ChunksFile to JSON bytes
func SerializeChunksFile(cf *ChunksFile) ([]byte, error) {
	return json.MarshalIndent(cf, "", "  ")
}

// ComputeHead30 returns hex-encoded first 30 bytes (or fewer if content is shorter)
func ComputeHead30(content []byte) string {
	n := len(content)
	if n > 30 {
		n = 30
	}
	return hex.EncodeToString(content[:n])
}

// ComputeTail30 returns hex-encoded last 30 bytes (or fewer if content is shorter)
func ComputeTail30(content []byte) string {
	n := len(content)
	if n > 30 {
		n = 30
	}
	return hex.EncodeToString(content[len(content)-n:])
}

// ComputeChunkHash returns MD5 hex of content
func ComputeChunkHash(content []byte) string {
	h := md5.Sum(content)
	return hex.EncodeToString(h[:])
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
