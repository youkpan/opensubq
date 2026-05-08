package handler

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"file-chat/llm"
	apimodel "file-chat/model"
	"file-chat/service"
)

type Config interface{}

type AppConfig struct {
	DeepSeekAPIKey  string
	DeepSeekBaseURL string
	Model           string
	DataDir         string
	MarkitdownCmd   string
	ChunkTokens     int
	MaxRetrieve     int
	SmallFileSize   int64
}

func ChatHandler(cfg AppConfig) http.HandlerFunc {
	// Initialize services
	client := &llm.Client{
		BaseURL: cfg.DeepSeekBaseURL,
		APIKey:  cfg.DeepSeekAPIKey,
		Model:   cfg.Model,
	}
	chatSvc := service.NewChatService(client, cfg.DataDir, cfg.MarkitdownCmd, cfg.ChunkTokens, cfg.MaxRetrieve, cfg.SmallFileSize)

	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "OPTIONS" {
			w.WriteHeader(200)
			return
		}
		if r.Method != "POST" {
			http.Error(w, "method not allowed", 405)
			return
		}

		// Parse request
		var req apimodel.ChatCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid request body", 400)
			return
		}

		// Get conversation ID from header
		conversationID := r.Header.Get("X-Conversation-Id")

		// Process files and build context
		finalMessages, err := chatSvc.ProcessRequest(req.Messages, conversationID)
		if err != nil {
			log.Printf("process request error: %v", err)
			finalMessages = convertMessages(req.Messages)
		}

		// Handle streaming
		if req.Stream {
			handleStream(w, client, req.Model, finalMessages)
			return
		}

		// Non-streaming
		handleNonStream(w, client, req.Model, finalMessages)
	}
}

func handleStream(w http.ResponseWriter, client *llm.Client, model string, messages []llm.ChatMessage) {
	sw, err := llm.NewStreamWriter(w)
	if err != nil {
		log.Printf("stream writer error: %v", err)
		http.Error(w, "streaming not supported", 500)
		return
	}

	chunkID := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())

	reqBody := map[string]interface{}{
		"model":       model,
		"messages":    messages,
		"stream":      true,
		"temperature": 1,
	}

	_, err = llm.ProxyStream(client.BaseURL, client.APIKey, reqBody, sw)
	if err != nil {
		log.Printf("proxy stream error: %v", err)
		errorChunk := apimodel.SSEChunk{
			ID:      chunkID,
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   model,
			Choices: []apimodel.Choice{{
				Index: 0,
				Delta: &apimodel.Delta{Content: fmt.Sprintf("\n\n[Error: %v]", err)},
			}},
		}
		sw.WriteChunk(errorChunk)
	}

	sw.WriteDone()
}

func handleNonStream(w http.ResponseWriter, client *llm.Client, model string, messages []llm.ChatMessage) {
	resp, err := client.Chat(messages)
	if err != nil {
		log.Printf("chat error: %v", err)
		http.Error(w, fmt.Sprintf("LLM error: %v", err), 500)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func convertMessages(msgs []apimodel.Message) []llm.ChatMessage {
	result := make([]llm.ChatMessage, len(msgs))
	for i, m := range msgs {
		content := m.Content
		if content == "" && m.Content == "" {
			// skip empty
		}
		content = service.CleanPaths(content)
		result[i] = llm.ChatMessage{
			Role:    m.Role,
			Content: content,
		}
	}
	return result
}

// extractTextFromContent extracts plain text from possibly multimodal content
func extractTextFromContent(content string) string {
	if strings.HasPrefix(content, "[") {
		var parts []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		}
		if err := json.Unmarshal([]byte(content), &parts); err == nil {
			var texts []string
			for _, p := range parts {
				if p.Type == "text" {
					texts = append(texts, p.Text)
				}
			}
			return strings.Join(texts, "\n")
		}
	}
	return content
}
