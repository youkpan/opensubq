package store

import (
	"fmt"
	"strings"
)

// ReadGlobalOutline reads the global outline content
func ReadGlobalOutline(dp *DataPaths) (string, error) {
	path := dp.GetGlobalOutlinePath()
	if !FileExists(path) {
		return "", nil
	}
	return ReadFile(path)
}

// AppendToGlobalOutline appends content to the global outline
func AppendToGlobalOutline(dp *DataPaths, content string) error {
	f, err := OpenAppend(dp.GetGlobalOutlinePath())
	if err != nil {
		return fmt.Errorf("open global outline: %w", err)
	}
	defer f.Close()
	_, err = f.WriteString(content)
	return err
}

// WriteGlobalOutline writes the full global outline (used when removing entries)
func WriteGlobalOutline(dp *DataPaths, content string) error {
	return WriteFile(dp.GetGlobalOutlinePath(), content)
}

// ReadGlobalSummary reads the global files summary
func ReadGlobalSummary(dp *DataPaths) string {
	path := dp.GetGlobalSummaryPath()
	if !FileExists(path) {
		return ""
	}
	content, err := ReadFile(path)
	if err != nil {
		return ""
	}
	return content
}

// TrimOutlineToSize trims outline content to maxBytes from the end, breaking at line boundary
func TrimOutlineToSize(content string, maxBytes int) string {
	if len(content) <= maxBytes {
		return content
	}
	// Take the last maxBytes
	truncated := content[len(content)-maxBytes:]
	// Break at first newline to ensure line-aligned
	if idx := strings.IndexByte(truncated, '\n'); idx >= 0 {
		truncated = truncated[idx+1:]
	}
	return truncated
}
