import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { roadmapSchema, type Roadmap } from '../../src/lib/roadmap';
import { stripOrderPrefix } from '../../src/content.config';

export interface LessonFile {
  file: string;
  slug: string;
  frontmatter: Record<string, unknown>;
  body: string;
}

function parseScalar(raw: string): unknown {
  const value = raw.trim();
  if (value.startsWith('[') && value.endsWith(']')) {
    const inner = value.slice(1, -1).trim();
    if (inner === '') return [];
    return inner.split(',').map((item) => parseScalar(item));
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
