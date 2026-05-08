package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"file-chat/handler"
)

// corsMiddleware adds CORS headers, request logging and panic recovery
func corsMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if err := recover(); err != nil {
				log.Printf("[PANIC] %s %s: %v", r.Method, r.URL.Path, err)
				http.Error(w, `{"error":true,"message":"internal server error"}`, http.StatusInternalServerError)
			}
		}()

		log.Printf("[REQUEST] %s %s from %s", r.Method, r.URL.Path, r.RemoteAddr)
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Conversation-Id")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next(w, r)
	}
}

// spaFileHandler serves static files with SPA fallback to index.html
type spaFileHandler struct {
	root http.FileSystem
}

func (h *spaFileHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Path

	// Try to open the requested file
	f, err := h.root.Open(path)
	if err != nil {
		// File not found — serve index.html for SPA routing
		f, err = h.root.Open("/index.html")
		if err != nil {
			http.NotFound(w, r)
			return
		}
		defer f.Close()
		stat, err := f.Stat()
		if err != nil {
			http.NotFound(w, r)
			return
		}
		http.ServeContent(w, r, "index.html", stat.ModTime(), f)
		return
	}
	defer f.Close()

	stat, err := f.Stat()
	if err != nil {
		http.NotFound(w, r)
		return
	}

	// If path is a directory, try index.html inside it
	if stat.IsDir() {
		indexPath := strings.TrimSuffix(path, "/") + "/index.html"
		indexFile, err := h.root.Open(indexPath)
		if err != nil {
			// Directory without index.html — fallback to root index.html
			indexFile, err = h.root.Open("/index.html")
			if err != nil {
				http.NotFound(w, r)
				return
			}
			defer indexFile.Close()
			indexStat, err := indexFile.Stat()
			if err != nil {
				http.NotFound(w, r)
				return
			}
			http.ServeContent(w, r, "index.html", indexStat.ModTime(), indexFile)
			return
		}
		defer indexFile.Close()
		indexStat, err := indexFile.Stat()
		if err != nil {
			http.NotFound(w, r)
			return
		}
		http.ServeContent(w, r, indexPath, indexStat.ModTime(), indexFile)
		return
	}

	http.ServeContent(w, r, path, stat.ModTime(), f)
}

func openBrowser(url string) {
	var cmd string
	var args []string

	switch runtime.GOOS {
	case "windows":
		cmd = "rundll32"
		args = []string{"url.dll,FileProtocolHandler", url}
	case "darwin":
		cmd = "open"
		args = []string{url}
	default:
		cmd = "xdg-open"
		args = []string{url}
	}

	go func() {
		// Wait a moment for the server to start
		time.Sleep(800 * time.Millisecond)
		if err := exec.Command(cmd, args...).Start(); err != nil {
			log.Printf("[BROWSER] failed to open browser: %v", err)
		}
	}()
}

func getDistDir() string {
	// Try to find dist directory relative to the executable
	ex, err := os.Executable()
	if err != nil {
		return "./dist"
	}
	exDir := filepath.Dir(ex)

	// Check if dist exists next to the executable
	distPath := filepath.Join(exDir, "dist")
	if _, err := os.Stat(distPath); err == nil {
		return distPath
	}

	// Fallback to current working directory
	cwd, _ := os.Getwd()
	cwdDist := filepath.Join(cwd, "dist")
	if _, err := os.Stat(cwdDist); err == nil {
		return cwdDist
	}

	return "./dist"
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
		DataDir:         config.DataDir,
		MarkitdownCmd:   config.MarkitdownCmd,
		ChunkTokens:     config.ChunkTokens,
		MaxRetrieve:     config.MaxRetrieve,
		SmallFileSize:   config.SmallFileSize,
	}

	// Register API routes with CORS middleware
	chatH := corsMiddleware(handler.ChatHandler(appCfg))
	modelsH := corsMiddleware(handler.ModelsHandler(appCfg))
	http.HandleFunc("/v1/chat/completions", chatH)
	http.HandleFunc("/v1/models", modelsH)
	http.HandleFunc("/chat/completions", chatH)
	http.HandleFunc("/models", modelsH)

	// Serve static files (NextChat frontend)
	distDir := getDistDir()
	log.Printf("[STATIC] serving from %s", distDir)
	http.Handle("/", &spaFileHandler{root: http.Dir(distDir)})

	addr := fmt.Sprintf(":%s", config.Port)
	log.Printf("file-chat server starting on %s", addr)

	// Auto-open browser
	openBrowser(fmt.Sprintf("http://localhost:%s", config.Port))

	log.Fatal(http.ListenAndServe(addr, nil))
}
