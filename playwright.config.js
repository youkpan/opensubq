const { defineConfig } = require('@playwright/test');
module.exports = defineConfig({
  testDir: './scripts',
  testMatch: 'test-*.js',
  use: { baseURL: 'http://localhost:8080' },
  reporter: 'list',
});
