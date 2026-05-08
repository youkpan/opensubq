package main

import (
	"os"
	"strconv"
)

type Config struct {
	DeepSeekAPIKey  string
	DeepSeekBaseURL string
	Model           string
	Port            string
	DataDir         string
	MarkitdownCmd   string
	ChunkTokens     int
	MaxRetrieve     int
	SmallFileSize   int64 // bytes
}

func LoadConfig() *Config {
	return &Config{
		DeepSeekAPIKey:  getEnv("DEEPSEEK_API_KEY", ""),
		DeepSeekBaseURL: getEnv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
		Model:           getEnv("MODEL", "deepseek-v4-flash"),
		Port:            getEnv("PORT", "8080"),
		DataDir:         getEnv("DATA_DIR", "./data"),
		MarkitdownCmd:   getEnv("MARKITDOWN_CMD", "markitdown"),
		ChunkTokens:     getEnvInt("CHUNK_TOKENS", 2000),
		MaxRetrieve:     getEnvInt("MAX_RETRIEVE", 20),
		SmallFileSize:   int64(getEnvInt("SMALL_FILE_SIZE", 15360)),
	}
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func getEnvInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			return i
		}
	}
	return fallback
}
