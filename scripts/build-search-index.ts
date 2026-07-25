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
    // Asterisk emphasis strips anywhere. Underscore emphasis only strips at
    // word boundaries (CommonMark's intraword rule) so identifiers like
    // `train_test_split` and `__init__` survive untouched — `\w` includes
    // `_`, so any underscore touching another underscore or an alnum on its
    // outer side is left alone, and multi-underscore runs (`__`, `___`)
    // never match at all.
    .replace(/\*{1,3}([^*]+)\*{1,3}/g, '$1')
    .replace(/(?<!\w)_(?!_)([^_]+)_(?!\w)/g, '$1')
    .replace(/^\s*[-*+]\s+/gm, '')
    .replace(/\|/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * Extracts `## ` headings, skipping any line inside a fenced code block
 * (``` or ```lang). Lessons embed fenced templates/snippets whose lines can
 * themselves start with `## ` (a README template, a commented-out Markdown
 * example) — those must not be mistaken for real headings.
 */
export function extractHeadings(markdown: string): string[] {
  const headings: string[] = [];
  let inFence = false;
  for (const line of markdown.split(/\r?\n/)) {
    if (/^\s{0,3}```/.test(line)) {
      inFence = !inFence;
      continue;
    }
    if (inFence) continue;
    const match = /^##\s+(.+)$/.exec(line);
    if (match) headings.push(match[1].trim());
  }
  return headings;
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
