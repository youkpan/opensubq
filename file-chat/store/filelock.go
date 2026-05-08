package store

import (
	"fmt"
	"os"
	"time"
)

// WithFileLock acquires a file-level lock and executes fn.
// Uses a .lock file with exclusive creation + polling.
func WithFileLock(filePath string, fn func() error) error {
	lockPath := filePath + ".lock"
	for i := 0; i < 100; i++ {
		f, err := os.OpenFile(lockPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0644)
		if err == nil {
			defer f.Close()
			defer os.Remove(lockPath)
			return fn()
		}
		time.Sleep(100 * time.Millisecond)
	}
	return fmt.Errorf("timeout acquiring lock for %s", filePath)
}
