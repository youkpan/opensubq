package main

import (
	"fmt"
	"os"
)

func main() {
	data, err := os.ReadFile("/f/github/subq/file-chat/testnovel.txt")
	if err != nil {
		panic(err)
	}

	insertPos := 46575 // after chunk_009

	var insertText string
	insertText += "\n\n【以下是新插入的大量内容，用于测试增量更新功能】\n\n"
	for i := 0; i < 50; i++ {
		insertText += fmt.Sprintf("第%d章 测试插入内容\n", i+1)
		insertText += "这是一个用于测试文件增量更新功能的插入段落。"
		insertText += "我们需要在文件中间插入大量内容，以验证系统是否能够正确处理新增 chunk 的生成。"
		insertText += "这段文字描述了江湖中一位无名侠客的故事，他行侠仗义，仗义疏财，深受百姓爱戴。"
		insertText += "某日，他在一座小镇上遇到了一位神秘的老人，老人赠予他一本失传已久的武功秘籍。"
		insertText += "侠客日夜苦练，终于练成了绝世武功，成为了武林中的传奇人物。"
		insertText += "然而，他并没有因此骄傲自满，而是继续行侠仗义，帮助那些需要帮助的人。"
		insertText += "他的故事在民间广为流传，成为了后世侠客们学习的楷模。\n\n"
	}

	insertBytes := []byte(insertText)
	newData := make([]byte, 0, len(data)+len(insertBytes))
	newData = append(newData, data[:insertPos]...)
	newData = append(newData, insertBytes...)
	newData = append(newData, data[insertPos:]...)

	if err := os.WriteFile("/f/github/subq/file-chat/testnovel.txt", newData, 0644); err != nil {
		panic(err)
	}

	fmt.Printf("Original size: %d bytes\n", len(data))
	fmt.Printf("Insert size: %d bytes\n", len(insertBytes))
	fmt.Printf("New size: %d bytes\n", len(newData))
	fmt.Printf("Insert position: %d\n", insertPos)
}
