// @ts-check
import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://sumitsingh4411.github.io',
  base: '/ai-roadmap',
  trailingSlash: 'ignore',
  markdown: {
    shikiConfig: {
      themes: { light: 'github-light', dark: 'github-dark-dimmed' },
      wrap: true,
    },
  },
});
