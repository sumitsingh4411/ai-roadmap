import { describe, it, expect } from 'vitest';
import { stripMarkdown, buildSearchIndex } from '../../scripts/build-search-index';
import { readLessonFiles } from '../../scripts/lib/content-io';

describe('stripMarkdown', () => {
  it('removes fenced code blocks', () => {
    expect(stripMarkdown('before\n```py\nx = 1\n```\nafter')).toBe('before after');
  });

  it('removes heading markers but keeps the text', () => {
    expect(stripMarkdown('## Why this matters')).toBe('Why this matters');
  });

  it('keeps link text and drops the URL', () => {
    expect(stripMarkdown('see [the docs](https://example.com) now')).toBe('see the docs now');
  });

  it('removes emphasis markers', () => {
    expect(stripMarkdown('**bold** and _italic_')).toBe('bold and italic');
  });

  it('removes asterisk emphasis', () => {
    expect(stripMarkdown('*emphasis*')).toBe('emphasis');
  });

  it('strips word-bounded underscore emphasis', () => {
    expect(stripMarkdown('a _real_ word')).toBe('a real word');
  });

  it('does not mangle snake_case identifiers', () => {
    expect(stripMarkdown('train_test_split')).toBe('train_test_split');
  });

  it('does not mangle dunder identifiers', () => {
    expect(stripMarkdown('__init__')).toBe('__init__');
  });

  it('collapses whitespace', () => {
    expect(stripMarkdown('a\n\n\n   b')).toBe('a b');
  });
});

describe('buildSearchIndex', () => {
  const docs = buildSearchIndex(readLessonFiles());

  it('produces one document per lesson', () => {
    expect(docs.length).toBe(readLessonFiles().length);
  });

  it('captures the h2 headings of each lesson', () => {
    const doc = docs.find((d) => d.slug === 'what-is-ai')!;
    expect(doc.headings).toContain('Why this matters');
  });

  it('stores body text with no code fences', () => {
    for (const doc of docs) expect(doc.text).not.toContain('```');
  });

  it('carries the title and summary through', () => {
    const doc = docs.find((d) => d.slug === 'what-is-ai')!;
    expect(doc.title.length).toBeGreaterThan(0);
    expect(doc.summary.length).toBeGreaterThan(0);
  });
});
