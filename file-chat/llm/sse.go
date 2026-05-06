package llm

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// StreamWriter writes SSE data to an http.ResponseWriter
type StreamWriter struct {
	w       http.ResponseWriter
	flusher http.Flusher
}

func NewStreamWriter(w http.ResponseWriter) (*StreamWriter, error) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return nil, fmt.Errorf("streaming not supported")
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	return &StreamWriter{w: w, flusher: flusher}, nil
}

// WriteChunk writes an SSE chunk to the response
func (sw *StreamWriter) WriteChunk(data interface{}) error {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return err
	}
	fmt.Fprintf(sw.w, "data: %s\n\n", string(jsonData))
	sw.flusher.Flush()
	return nil
}

// WriteDone writes the SSE [DONE] marker
func (sw *StreamWriter) WriteDone() {
	fmt.Fprintf(sw.w, "data: [DONE]\n\n")
	sw.flusher.Flush()
}

// ProxyStream proxies a streaming request from DeepSeek to the client
func ProxyStream(baseURL, apiKey string, reqBody interface{}, sw *StreamWriter) (string, error) {
	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("marshal: %w", err)
	}

	req, err := http.NewRequest("POST", baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	client := &http.Client{Timeout: 10 * time.Minute}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("API error %d: %s", resp.StatusCode, string(respBody))
	}

	var fullContent strings.Builder
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		// Forward the raw data to client
		fmt.Fprintf(sw.w, "data: %s\n\n", data)
		sw.flusher.Flush()

		// Parse to accumulate content
		var chunk StreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err == nil {
			if len(chunk.Choices) > 0 {
				fullContent.WriteString(chunk.Choices[0].Delta.Content)
			}
		}
	}

	return fullContent.String(), nil
}
