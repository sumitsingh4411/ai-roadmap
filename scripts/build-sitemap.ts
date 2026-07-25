import { writeFileSync, mkdirSync } from 'node:fs';
import { readLessonFiles } from './lib/content-io';

const SITE = 'https://sumitsingh4411.github.io/ai-roadmap';

/** Emits a sitemap of every indexable page so search engines can crawl them. */
function buildSitemap(): string {
  const today = new Date().toISOString().slice(0, 10);
  const slugs = readLessonFiles().map((l) => l.slug);
  const urls = [
    { loc: `${SITE}/`, priority: '1.0', changefreq: 'weekly' },
    { loc: `${SITE}/roadmap`, priority: '0.9', changefreq: 'weekly' },
    { loc: `${SITE}/advanced`, priority: '0.8', changefreq: 'monthly' },
    ...slugs.map((slug) => ({ loc: `${SITE}/lessons/${slug}`, priority: '0.8', changefreq: 'monthly' })),
  ];

  const body = urls
    .map(
      (u) =>
        `  <url>\n    <loc>${u.loc}</loc>\n    <lastmod>${today}</lastmod>\n` +
        `    <changefreq>${u.changefreq}</changefreq>\n    <priority>${u.priority}</priority>\n  </url>`,
    )
    .join('\n');

  return `<?xml version="1.0" encoding="UTF-8"?>\n` +
    `<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${body}\n</urlset>\n`;
}

function main(): void {
  mkdirSync('public', { recursive: true });
  const xml = buildSitemap();
  writeFileSync('public/sitemap.xml', xml, 'utf8');
  const count = (xml.match(/<url>/g) || []).length;
  console.log(`✓ Wrote public/sitemap.xml (${count} URLs)`);
}

if (import.meta.url === `file://${process.argv[1]}`) main();
