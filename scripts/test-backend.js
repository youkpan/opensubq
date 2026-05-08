// Playwright 快速测试：仅 file-chat 后端
// 用法: npx playwright test scripts/test-backend.js

const { test, expect } = require('@playwright/test');
const BASE = 'http://localhost:8880';
const FILE = 'F:/github/subq/file-chat/testfile.txt';
test.setTimeout(60000);

test('models API', async ({ request }) => {
  const r = await request.get(`${BASE}/v1/models`);
  expect(r.ok()).toBeTruthy();
  console.log('✓ /v1/models OK');
});

test('普通对话', async ({ request }) => {
  const r = await request.post(`${BASE}/v1/chat/completions`, {
    headers: { 'Content-Type': 'application/json' },
    data: { model: 'deepseek-v4-flash', messages: [{ role: 'user', content: '说OK' }], stream: false },
  });
  expect(r.ok()).toBeTruthy();
  const d = await r.json();
  console.log('✓ 普通对话:', d.choices[0].message.content.slice(0, 50));
});

test('文件引用', async ({ request }) => {
  const r = await request.post(`${BASE}/v1/chat/completions`, {
    headers: { 'Content-Type': 'application/json', 'X-Conversation-Id': 'pw-test-001' },
    data: { model: 'deepseek-v4-flash', messages: [{ role: 'user', content: `总结文件内容 @${FILE}` }], stream: false },
  });
  expect(r.ok()).toBeTruthy();
  const d = await r.json();
  console.log('✓ 文件引用:', d.choices[0].message.content.slice(0, 80));
});

test('追问', async ({ request }) => {
  const r = await request.post(`${BASE}/v1/chat/completions`, {
    headers: { 'Content-Type': 'application/json', 'X-Conversation-Id': 'pw-test-001' },
    data: {
      model: 'deepseek-v4-flash',
      messages: [
        { role: 'user', content: `总结文件内容 @${FILE}` },
        { role: 'assistant', content: '这是测试文件。' },
        { role: 'user', content: '性能指标是什么？' },
      ], stream: false,
    },
  });
  expect(r.ok()).toBeTruthy();
  const d = await r.json();
  console.log('✓ 追问:', d.choices[0].message.content.slice(0, 80));
});

test('SSE stream', async ({ request }) => {
  const r = await request.post(`${BASE}/v1/chat/completions`, {
    headers: { 'Content-Type': 'application/json' },
    data: { model: 'deepseek-v4-flash', messages: [{ role: 'user', content: '说OK' }], stream: true },
  });
  expect(r.ok()).toBeTruthy();
  const text = await r.text();
  expect(text).toContain('data: [DONE]');
  console.log('✓ SSE stream 正常');
});
