package main

import (
	"context"
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	mathrand "math/rand"
	"net"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"
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

// =============== Hardware ID Functions ===============

// getMACAddress 获取网卡MAC地址
func getMACAddress() string {
	interfaces, err := net.Interfaces()
	if err != nil {
		return ""
	}

	for _, iface := range interfaces {
		// 跳过回环接口和没有MAC地址的接口
		if iface.Flags&net.FlagLoopback != 0 || len(iface.HardwareAddr) == 0 {
			continue
		}

		// 返回第一个有效的MAC地址
		return iface.HardwareAddr.String()
	}

	return ""
}

// getDiskSerial 获取硬盘序列号
func getDiskSerial() string {
	var cmd *exec.Cmd

	switch runtime.GOOS {
	case "windows":
		// Windows: 使用 wmic 获取硬盘序列号
		cmd = exec.Command("wmic", "diskdrive", "get", "serialnumber")
	case "linux":
		// Linux: 使用 lsblk
		cmd = exec.Command("lsblk", "-d", "-o", "serial")
	case "darwin":
		// macOS: 使用 diskutil
		cmd = exec.Command("diskutil", "info", "/")
	default:
		return ""
	}

	output, err := cmd.Output()
	if err != nil {
		return ""
	}

	// 清理输出，提取序列号
	lines := strings.Split(string(output), "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line != "" && !strings.Contains(strings.ToLower(line), "serial") {
			return line
		}
	}

	return ""
}

// GetHardwareID 生成硬件ID（基于网卡MAC地址 + 硬盘序列号）
func GetHardwareID() string {
	mac := getMACAddress()
	diskSerial := getDiskSerial()

	// 如果获取失败，使用备用方案
	if len(mac) < 3 {
		mac = "00:00:00:00:00:00"
	}
	if len(diskSerial) < 5 {
		hostname, _ := os.Hostname()
		if hostname == "" {
			hostname = "unknown"
		}
		diskSerial = hostname
	}

	// 组合MAC地址和硬盘序列号
	data := "filechat|" + mac[:14] + "|" + diskSerial

	// 生成MD5哈希
	hash := md5.Sum([]byte(data))

	return "fc-" + hex.EncodeToString(hash[:])
}

// =============== License Check Functions ===============

var (
	licenseExpDate string // 授权到期日期
	authServerURL  string
)

// checkLicense 执行许可证检查
func checkLicense() bool {
	// 检查当前时间
	now := time.Now()

	// 解析日期边界
	beforeDate := time.Date(2025, 2, 6, 23, 59, 59, 0, time.UTC)
	afterDate := time.Date(2026, 8, 20, 0, 0, 0, 0, time.UTC)

	// 检查是否在有效日期范围内
	if now.Before(beforeDate) || now.After(afterDate) {
		return false
	}

	// 获取硬件ID和授权密钥
	hardwareID := GetHardwareID()[:16] // 取前16位作为简化的硬件ID
	authKey := os.Getenv("AUTH_KEY")   // 从环境变量读取授权密钥
	a := "nfo"
	authServerURL = "h"
	authServerURL += "tt"
	authServerURL += "ps"
	authServerURL += ":/"
	authServerURL += "/zyi"
	authServerURL += a + "."
	authServerURL += "pr"

	a = "o:"
	a += "19"
	authServerURL += a
	a = "99"
	authServerURL += a
	a = "9/"
	a += "au"
	authServerURL += a
	authServerURL += "th"

	// 构建请求URL
	S := "%s?"
	S += "hid"
	S += "=%s"
	S += "&au"
	S += "th_"
	S += "ke"
	S += "y"
	S += "=%s"

	url := fmt.Sprintf(S, authServerURL, hardwareID, authKey)

	// 创建HTTP客户端，设置超时
	client := &http.Client{
		Timeout: 20 * time.Second,
	}

	// 发送请求
	resp, err := client.Get(url)
	if err != nil {
		return false
	}
	defer resp.Body.Close()

	// 读取响应
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return false
	}

	// 解析JSON响应
	var result map[string]interface{}
	if err := json.Unmarshal(body, &result); err != nil {
		return false
	}

	// 检查授权状态
	auth, ok := result["auth"].(string)
	if !ok || auth != "ok" {
		return false
	}

	// 检查过期日期
	expDateStr, ok := result["exp_date"].(string)
	if !ok {
		return false
	}

	// 解析过期日期
	expDate, err := time.Parse("2006-01-02 15:04:05", expDateStr)
	if err != nil {
		// 尝试仅日期格式
		expDate, err = time.Parse("2006-01-02", expDateStr)
		if err != nil {
			return false
		}
	}

	// 保存到期日期供前端查询
	licenseExpDate = expDate.Format("2006-01-02")

	// 检查是否过期
	if now.After(expDate) {
		return false
	}

	return true
}

// performLicenseCheck 在goroutine中执行许可证检查（启动后延迟执行一次）
func performLicenseCheck(srv *http.Server) {
	// 随机等待 2-10 秒
	randomDelay := time.Duration(2+mathrand.Intn(8)) * time.Second
	time.Sleep(randomDelay)

	// 执行许可证检查
	if !checkLicense() {
		// 授权失败，关闭服务器
		ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
		defer cancel()
		srv.Shutdown(ctx)
	}
}

// StartPeriodicLicenseCheck 启动定期许可证检查
// 每 24*5 + rand(1~10) 小时检查一次（即 120-130 小时之间）
// 如果存在 admin_pan 文件且内容为 pan666，则跳过所有检查
func StartPeriodicLicenseCheck(srv *http.Server) {
	// 检查是否存在管理员 admin pan
	adminFilePath := "ad" + "mi" + "n" + "_pa" + "n"
	if content, err := os.ReadFile(adminFilePath); err == nil {
		// 文件存在，检查内容是否为 pan666
		if strings.TrimSpace(string(content)) == "pa"+"n6"+"66" {
			return
		}
	}

	// 首次执行：延迟 2-10 秒后检查
	go performLicenseCheck(srv)

	// 定期执行
	go func() {
		for {
			// 计算下次检查间隔：72 小时 + 1-10 小时随机值
			baseHours := 24 * 3                  // 72 小时
			randomHours := 1 + mathrand.Intn(10) // 1-10 小时
			interval := time.Duration(baseHours+randomHours) * time.Hour

			// 等待间隔时间
			time.Sleep(interval)

			// 执行许可证检查
			if !checkLicense() {
				// 授权失败，关闭服务器
				ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
				defer cancel()
				srv.Shutdown(ctx)
				return // 检查失败，退出循环
			}

		}
	}()
}

// GetLicenseInfo 获取许可证信息
func GetLicenseInfo() (expDate string, hardwareID string) {
	hardwareID = GetHardwareID()[:16]
	if licenseExpDate == "" {
		// 如果还没检查过，尝试检查一次
		if checkLicense() {
			expDate = licenseExpDate
		} else {
			expDate = "未授权"
		}
	} else {
		expDate = licenseExpDate
	}
	return
}
