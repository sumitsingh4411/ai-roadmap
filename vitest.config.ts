import { defineConfig } from 'vitest/config';

export default defineConfig({
  resolve: {
    alias: {
      'astro:content': new URL('./tests/stubs/astro-content.ts', import.meta.url).pathname,
    },
  },
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts'],
  },
});
