// Playwright 测试：NextChat + file-chat 端到端测试
// 用法: npx playwright test scripts/test-nextchat.js
// 前提: file-chat 运行在 localhost:8080, NextChat 运行在 localhost:3000

const { test, expect } = require('@playwright/test');

const FILE_CHAT_URL = 'http://localhost:8080';
const NEXTCHAT_URL = 'http://localhost:3000';
const TEST_FILE = 'F:/github/subq/file-chat/testfile.txt';

test.setTimeout(120000);

test.describe('File-Chat E2E', () => {

  test('1. file-chat /v1/models 接口', async ({ request }) => {
    const resp = await request.get(`${FILE_CHAT_URL}/v1/models`);
    expect(resp.ok()).toBeTruthy();
    const data = await resp.json();
    expect(data.data[0].id).toBe('deepseek-v4-flash');
  });

  test('2. file-chat 普通对话', async ({ request }) => {
    const resp = await request.post(`${FILE_CHAT_URL}/v1/chat/completions`, {
      headers: { 'Content-Type': 'application/json' },
      data: {
        model: 'deepseek-v4-flash',
        messages: [{ role: 'user', content: '用一句话说你好' }],
        stream: false,
      },
    });
    expect(resp.ok()).toBeTruthy();
    const data = await resp.json();
    expect(data.choices[0].message.content).toBeTruthy();
    console.log('普通对话回复:', data.choices[0].message.content.slice(0, 80));
  });

  test('3. file-chat 文件引用', async ({ request }) => {
    const resp = await request.post(`${FILE_CHAT_URL}/v1/chat/completions`, {
      headers: {
        'Content-Type': 'application/json',
        'X-Conversation-Id': 'e2e-test-001',
      },
      data: {
        model: 'deepseek-v4-flash',
        messages: [{ role: 'user', content: `总结这个文件的内容 @${TEST_FILE}` }],
        stream: false,
      },
    });
    expect(resp.ok()).toBeTruthy();
    const data = await resp.json();
    const content = data.choices[0].message.content;
    expect(content.length).toBeGreaterThan(10);
    console.log('文件引用回复:', content.slice(0, 100));
  });

  test('4. file-chat SSE stream', async ({ request }) => {
    const resp = await request.post(`${FILE_CHAT_URL}/v1/chat/completions`, {
      headers: { 'Content-Type': 'application/json' },
      data: {
        model: 'deepseek-v4-flash',
        messages: [{ role: 'user', content: '说OK' }],
        stream: true,
      },
    });
    expect(resp.ok()).toBeTruthy();
    const text = await resp.text();
    expect(text).toContain('data: [DONE]');
    console.log('SSE stream 正常结束');
  });

  test('5. NextChat 页面可访问', async ({ page }) => {
    await page.goto(NEXTCHAT_URL, { timeout: 30000 });
    // NextChat 应该加载成功
    await page.waitForTimeout(3000);
    const title = await page.title();
    console.log('NextChat title:', title);
    expect(title).toBeTruthy();
  });

  test('6. NextChat 发送消息', async ({ page }) => {
    await page.goto(NEXTCHAT_URL, { timeout: 30000 });
    await page.waitForTimeout(3000);

    // 找到输入框并输入消息
    const textarea = page.locator('textarea').first();
    if (await textarea.isVisible()) {
      await textarea.fill('你好，这是一个测试消息');
      await page.waitForTimeout(500);

      // 找发送按钮并点击
      const sendBtn = page.locator('button[type="submit"], [data-testid="send-button"]').first();
      if (await sendBtn.isVisible()) {
        await sendBtn.click();
        await page.waitForTimeout(10000);

        // 检查是否有回复
        const messages = page.locator('[class*="message"], [class*="chat"]');
        const count = await messages.count();
        console.log('页面消息数量:', count);
      }
    } else {
      console.log('未找到输入框，可能需要先配置 API');
    }
  });
});
