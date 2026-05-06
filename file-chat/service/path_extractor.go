package service

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// Matches: @/abs/path, @C:\path, @.\rel, @..\rel, @folder/file.ext
var pathPattern = regexp.MustCompile(`@((?:[A-Za-z]:[\\/]|[.]{0,2}[\\/])\S+|\S+[\\/]\S+)`)

const AllFilesMarker = "@全部"
const AllFilesMarkerEn = "@all"

// ExtractPaths extracts @path references from message content.
// Returns (paths, hasAll) where hasAll indicates @全部 or @all was used.
func ExtractPaths(content string) ([]string, bool) {
	// Remove @全部/@all first so they don't interfere with path matching
	cleaned := strings.ReplaceAll(content, AllFilesMarker, "")
	cleaned = strings.ReplaceAll(cleaned, AllFilesMarkerEn+" ", " ")
	cleaned = strings.ReplaceAll(cleaned, AllFilesMarkerEn, "")
	hasAll := strings.Contains(content, AllFilesMarker) || strings.Contains(content, AllFilesMarkerEn+" ") || strings.HasSuffix(strings.TrimSpace(content), AllFilesMarkerEn)

	matches := pathPattern.FindAllStringSubmatch(cleaned, -1)
	seen := make(map[string]bool)
	var paths []string
	for _, m := range matches {
		p := m[1]
		// Skip Chinese-only strings (not file paths)
		if !strings.ContainsAny(p, "/\\.:") {
			continue
		}
		if !seen[p] {
			seen[p] = true
			paths = append(paths, p)
		}
	}
	return paths, hasAll
}

// CleanPaths removes @path references and @全部/@all from message content
func CleanPaths(content string) string {
	content = strings.ReplaceAll(content, AllFilesMarker, "")
	content = strings.ReplaceAll(content, AllFilesMarkerEn+" ", " ")
	content = strings.ReplaceAll(content, AllFilesMarkerEn, "")
	return pathPattern.ReplaceAllString(content, "")
}

// ResolvePath resolves and validates a path, returns cleaned absolute path
func ResolvePath(p string) (string, bool) {
	abs, err := filepath.Abs(p)
	if err != nil {
		return "", false
	}
	info, err := os.Stat(abs)
	if err != nil {
		return "", false
	}
	return abs, info.IsDir()
}

// ListFiles recursively lists all supported files under a path
func ListFiles(root string) []string {
	var files []string
	supported := map[string]bool{
		".txt": true, ".md": true, ".go": true, ".py": true, ".java": true,
		".js": true, ".ts": true, ".json": true, ".yaml": true, ".yml": true,
		".xml": true, ".html": true, ".css": true, ".sql": true, ".sh": true,
		".c": true, ".cpp": true, ".h": true, ".rs": true, ".toml": true,
		".ini": true, ".cfg": true, ".conf": true, ".log": true,
		".doc": true, ".docx": true, ".pdf": true,
		".xls": true, ".xlsx": true, ".ppt": true, ".pptx": true,
		".csv": true,
	}

	filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(path))
		if supported[ext] {
			files = append(files, path)
		}
		return nil
	})
	return files
}

// IsDocumentFile checks if a file needs markitdown conversion
func IsDocumentFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	return ext == ".doc" || ext == ".docx" || ext == ".pdf" ||
		ext == ".xls" || ext == ".xlsx" || ext == ".ppt" || ext == ".pptx"
}
