import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { roadmapSchema, type Roadmap } from '../../src/lib/roadmap';
import { stripOrderPrefix } from '../../src/lib/lesson';

export interface LessonFile {
  file: string;
  slug: string;
  frontmatter: Record<string, unknown>;
  body: string;
}

/**
 * Splits a flow-sequence's inner content on commas, but only outside of
 * quoted strings — so `"a, b", "c"` yields two items, not three. Supports
 * both `"` and `'` quoting, matching the scalar parser below. An unterminated
 * quote is a clear input error, so it throws rather than corrupting the split.
 */
function splitListItems(inner: string): string[] {
  const items: string[] = [];
  let current = '';
  let quote: '"' | "'" | null = null;

  for (const char of inner) {
    if (quote) {
      current += char;
      if (char === quote) quote = null;
      continue;
    }
    if (char === '"' || char === "'") {
      quote = char;
      current += char;
      continue;
    }
    if (char === ',') {
      items.push(current);
      current = '';
      continue;
    }
    current += char;
  }

  if (quote) throw new Error(`Unterminated quote (${quote}) in list value: ${inner}`);
  items.push(current);
  return items;
}

function parseScalar(raw: string): unknown {
  const value = raw.trim();
  if (value.startsWith('[') && value.endsWith(']')) {
    const inner = value.slice(1, -1).trim();
    if (inner === '') return [];
    return splitListItems(inner).map((item) => parseScalar(item));
  }
  if (/^".*"$/.test(value) || /^'.*'$/.test(value)) return value.slice(1, -1);
  if (/^-?\d+$/.test(value)) return Number(value);
  if (value === 'true') return true;
  if (value === 'false') return false;
  return value;
}

/** Splits `---` delimited YAML frontmatter from the Markdown body. */
export function parseFrontmatter(raw: string): {
  frontmatter: Record<string, unknown>;
  body: string;
} {
  const match = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/.exec(raw);
  if (!match) throw new Error('Missing YAML frontmatter');

  const frontmatter: Record<string, unknown> = {};
  for (const line of match[1].split(/\r?\n/)) {
    if (!line.trim() || line.trimStart().startsWith('#')) continue;
    const separator = line.indexOf(':');
    if (separator === -1) continue;
    frontmatter[line.slice(0, separator).trim()] = parseScalar(line.slice(separator + 1));
  }

  return { frontmatter, body: match[2] };
}

export function readRoadmap(root = process.cwd()): Roadmap {
  const raw = readFileSync(join(root, 'content/roadmap.json'), 'utf8');
  return roadmapSchema.parse(JSON.parse(raw));
}

export function readLessonFiles(root = process.cwd()): LessonFile[] {
  const dir = join(root, 'content/lessons');
  return readdirSync(dir)
    .filter((name) => name.endsWith('.md') && !name.startsWith('_'))
    .sort()
    .map((name) => {
      const raw = readFileSync(join(dir, name), 'utf8');
      const { frontmatter, body } = parseFrontmatter(raw);
      return { file: `content/lessons/${name}`, slug: stripOrderPrefix(name), frontmatter, body };
    });
}
