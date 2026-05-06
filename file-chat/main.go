package main

import (
	"fmt"
	"log"
	"net/http"

	"file-chat/handler"
)

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

	// Register routes
	http.HandleFunc("/v1/chat/completions", handler.ChatHandler(appCfg))
	http.HandleFunc("/v1/models", handler.ModelsHandler(appCfg))

	addr := fmt.Sprintf(":%s", config.Port)
	log.Printf("file-chat server starting on %s", addr)
	log.Fatal(http.ListenAndServe(addr, nil))
}
