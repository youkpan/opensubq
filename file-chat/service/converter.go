package service

import (
	"fmt"
	"os/exec"
)

// ConvertDocument calls markitdown CLI to convert a document to text
func ConvertDocument(markitdownCmd, filePath string) (string, error) {
	cmd := exec.Command(markitdownCmd, filePath)
	output, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("markitdown failed for %s: %w", filePath, err)
	}
	return string(output), nil
}
