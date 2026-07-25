import { describe, it, expect } from 'vitest';
import { stripMarkdown, buildSearchIndex, extractHeadings } from '../../scripts/build-search-index';
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

describe('extractHeadings', () => {
  it('extracts real h2 headings', () => {
    expect(extractHeadings('## Real Heading\ntext')).toEqual(['Real Heading']);
  });

  it('ignores lines that look like headings inside fenced code blocks', () => {
    const markdown = [
      '## Real Heading',
      'intro text',
      '```markdown',
      '## Not A Heading',
      '```',
      'more text',
    ].join('\n');
    expect(extractHeadings(markdown)).toEqual(['Real Heading']);
  });

  it('ignores fenced blocks that specify a language', () => {
    const markdown = ['```python', '## not a heading either', '```', '## Real One'].join('\n');
    expect(extractHeadings(markdown)).toEqual(['Real One']);
  });

  it('resumes extracting headings after a fence closes', () => {
    const markdown = ['```', '## inside fence', '```', '## Real Heading', 'text'].join('\n');
    expect(extractHeadings(markdown)).toEqual(['Real Heading']);
  });

  it('handles an odd number of fences by treating trailing content as still fenced', () => {
    const markdown = ['## Before', '```', '## After unterminated fence'].join('\n');
    expect(extractHeadings(markdown)).toEqual(['Before']);
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

  it('does not capture headings that live inside a fenced code block', () => {
    // lesson 01 has a fenced ```markdown template containing "## Week 01"
    const doc = docs.find((d) => d.slug === 'how-to-learn-ai')!;
    expect(doc.headings).not.toContain('Week 01');
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
