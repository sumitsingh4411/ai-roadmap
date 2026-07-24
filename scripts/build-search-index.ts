import { writeFileSync, mkdirSync } from 'node:fs';
import { lessonSchema } from '../src/lib/lesson';
import { readLessonFiles, type LessonFile } from './lib/content-io';

export interface SearchDoc {
  slug: string;
  title: string;
  stage: number;
  summary: string;
  headings: string[];
  text: string;
}

/** Reduces Markdown to plain prose suitable for substring matching. */
export function stripMarkdown(markdown: string): string {
  return markdown
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`([^`]*)`/g, '$1')
    .replace(/!\[[^\]]*\]\([^)]*\)/g, ' ')
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/^\s{0,3}#{1,6}\s+/gm, '')
    .replace(/^\s{0,3}>\s?/gm, '')
    .replace(/[*_]{1,3}([^*_]+)[*_]{1,3}/g, '$1')
    .replace(/^\s*[-*+]\s+/gm, '')
    .replace(/\|/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function extractHeadings(markdown: string): string[] {
  return [...markdown.matchAll(/^##\s+(.+)$/gm)].map((m) => m[1].trim());
}

export function buildSearchIndex(lessons: LessonFile[]): SearchDoc[] {
  return lessons.flatMap((lesson) => {
    const parsed = lessonSchema.safeParse(lesson.frontmatter);
    if (!parsed.success) return [];
    return [
      {
        slug: lesson.slug,
        title: parsed.data.title,
        stage: parsed.data.stage,
        summary: parsed.data.summary,
        headings: extractHeadings(lesson.body),
        text: stripMarkdown(lesson.body).slice(0, 4000),
      },
    ];
  });
}

function main(): void {
  const docs = buildSearchIndex(readLessonFiles());
  mkdirSync('public', { recursive: true });
  writeFileSync('public/search-index.json', JSON.stringify(docs), 'utf8');
  console.log(`✓ Wrote public/search-index.json (${docs.length} documents)`);
}

if (import.meta.url === `file://${process.argv[1]}`) main();
