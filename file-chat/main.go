package main

import (
	"fmt"
	"log"
	"net/http"

	"file-chat/handler"
)

// corsMiddleware adds CORS headers and request logging
func corsMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		log.Printf("[REQUEST] %s %s from %s", r.Method, r.URL.Path, r.RemoteAddr)
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Conversation-Id")

		// Handle preflight
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next(w, r)
	}
}

func main() {
	config := LoadConfig()

	if config.DeepSeekAPIKey == "" {
		log.Fatal("DEEPSEEK_API_KEY is required")
	}

	appCfg := handler.AppConfig{
		DeepSeekAPIKey:  config.DeepSeekAPIKey,
		DeepSeekBaseURL: config.DeepSeekBaseURL,
		Model:           config.Model,
		JobsDir:         config.JobsDir,
		MarkitdownCmd:   config.MarkitdownCmd,
		ChunkTokens:     config.ChunkTokens,
		MaxRetrieve:     config.MaxRetrieve,
		SmallFileSize:   config.SmallFileSize,
	}

	// Register routes with CORS middleware (both /v1/ and non-/v1/ paths)
	chatH := corsMiddleware(handler.ChatHandler(appCfg))
	modelsH := corsMiddleware(handler.ModelsHandler(appCfg))
	http.HandleFunc("/v1/chat/completions", chatH)
	http.HandleFunc("/v1/models", modelsH)
	http.HandleFunc("/chat/completions", chatH)
	http.HandleFunc("/models", modelsH)

	addr := fmt.Sprintf(":%s", config.Port)
	log.Printf("file-chat server starting on %s", addr)
	log.Fatal(http.ListenAndServe(addr, nil))
}
