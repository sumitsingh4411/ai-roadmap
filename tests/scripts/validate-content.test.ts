import { describe, it, expect } from 'vitest';
import { validateContent, isPendingLesson } from '../../scripts/validate-content';
import { parseFrontmatter, readRoadmap, readLessonFiles } from '../../scripts/lib/content-io';
import type { Roadmap } from '../../src/lib/roadmap';

const lesson = (slug: string, over: Record<string, unknown> = {}) => ({
  file: `content/lessons/00-${slug}.md`,
  slug,
  frontmatter: {
    title: 'T',
    stage: 0,
    order: 0,
    minutes: 10,
    difficulty: 'beginner',
    prerequisites: [],
    tags: [],
    summary: 's',
    ...over,
  },
  body: '# T',
});

const roadmap = (nodes: Roadmap['nodes']): Roadmap => ({
  stages: [{ id: 0, name: 'Orientation', color: 'violet' }],
  nodes,
});

const node = (id: string, prerequisites: string[] = [], x = 0) => ({
  id,
  lesson: `00-${id}`,
  stage: 0,
  grid: { x, y: 0 },
  prerequisites,
});

describe('parseFrontmatter', () => {
  it('splits YAML frontmatter from the body', () => {
    const { frontmatter, body } = parseFrontmatter('---\ntitle: "Hi"\nstage: 0\n---\n\n# Body\n');
    expect(frontmatter.title).toBe('Hi');
    expect(frontmatter.stage).toBe(0);
    expect(body.trim()).toBe('# Body');
  });

  it('parses list values', () => {
    const { frontmatter } = parseFrontmatter('---\ntags: ["a", "b"]\n---\nx');
    expect(frontmatter.tags).toEqual(['a', 'b']);
  });

  it('parses an empty list', () => {
    const { frontmatter } = parseFrontmatter('---\ntags: []\n---\nx');
    expect(frontmatter.tags).toEqual([]);
  });

  it('parses a list of bare numbers', () => {
    const { frontmatter } = parseFrontmatter('---\ntags: [1, 2]\n---\nx');
    expect(frontmatter.tags).toEqual([1, 2]);
  });

  it('parses a list of unquoted bare words', () => {
    const { frontmatter } = parseFrontmatter('---\ntags: [a, b]\n---\nx');
    expect(frontmatter.tags).toEqual(['a', 'b']);
  });

  it('tolerates surrounding whitespace around list items', () => {
    const { frontmatter } = parseFrontmatter('---\ntags: [ "a" ,  "b" ]\n---\nx');
    expect(frontmatter.tags).toEqual(['a', 'b']);
  });

  it('does not split a double-quoted item on an internal comma', () => {
    const { frontmatter } = parseFrontmatter('---\ntags: ["a, b", "c"]\n---\nx');
    expect(frontmatter.tags).toEqual(['a, b', 'c']);
  });

  it('does not split a single-quoted item on an internal comma', () => {
    const { frontmatter } = parseFrontmatter("---\ntags: ['a, b', 'c']\n---\nx");
    expect(frontmatter.tags).toEqual(['a, b', 'c']);
  });

  it('throws a clear error for an unterminated quote in a list', () => {
    expect(() => parseFrontmatter('---\ntags: ["a, b", "c]\n---\nx')).toThrow(/unterminated quote/i);
  });

  it('throws when frontmatter is missing', () => {
    expect(() => parseFrontmatter('# No frontmatter')).toThrow(/frontmatter/i);
  });
});

describe('validateContent', () => {
  it('returns no errors for a consistent pair', () => {
    expect(validateContent(roadmap([node('a')]), [lesson('a')])).toEqual([]);
  });

  it('reports a node whose lesson file is missing', () => {
    const errors = validateContent(roadmap([node('a'), node('b', [], 1)]), [lesson('a')]);
    expect(errors.join('\n')).toMatch(/b.*missing lesson/i);
  });

  it('reports an orphan lesson not referenced by any node', () => {
    const errors = validateContent(roadmap([node('a')]), [lesson('a'), lesson('b')]);
    expect(errors.join('\n')).toMatch(/orphan/i);
  });

  it('reports an unknown prerequisite id', () => {
    const errors = validateContent(roadmap([node('a', ['ghost'])]), [lesson('a')]);
    expect(errors.join('\n')).toMatch(/ghost/);
  });

  it('reports a prerequisite cycle', () => {
    const errors = validateContent(
      roadmap([node('a', ['b']), node('b', ['a'], 1)]),
      [lesson('a'), lesson('b')],
    );
    expect(errors.join('\n')).toMatch(/cycle/i);
  });

  it('reports duplicate grid positions', () => {
    const errors = validateContent(
      roadmap([node('a', [], 0), node('b', [], 0)]),
      [lesson('a'), lesson('b')],
    );
    expect(errors.join('\n')).toMatch(/grid position/i);
  });

  it('reports frontmatter failing the schema', () => {
    const errors = validateContent(roadmap([node('a')]), [lesson('a', { difficulty: 'expert' })]);
    expect(errors.join('\n')).toMatch(/difficulty/i);
  });

  it('reports a stage mismatch between node and frontmatter', () => {
    const errors = validateContent(roadmap([node('a')]), [lesson('a', { stage: 3 })]);
    expect(errors.join('\n')).toMatch(/stage/i);
  });

  it('reports a node whose stage id is not declared in roadmap.stages', () => {
    const errors = validateContent(roadmap([{ ...node('a'), stage: 9 }]), [lesson('a', { stage: 9 })]);
    expect(errors.join('\n')).toMatch(/unknown stage/i);
  });

  it('reports a lesson whose frontmatter prerequisites disagree with the graph', () => {
    const errors = validateContent(
      roadmap([node('a'), node('b', ['a'], 1)]),
      [lesson('a'), { ...lesson('b'), frontmatter: { ...lesson('b').frontmatter, prerequisites: ['ghost'] } }],
    );
    expect(errors.join('\n')).toMatch(/prerequisite/i);
  });

  it('reports a node whose lesson field does not resolve to its id', () => {
    const errors = validateContent(
      roadmap([{ ...node('numpy'), lesson: '03-wrong' }]),
      [lesson('numpy')],
    );
    expect(errors.join('\n')).toContain(
      'Node "numpy" lesson field "03-wrong" does not resolve to its id (got "wrong").',
    );
  });

  it('does not report a node whose lesson field correctly resolves to its id', () => {
    const errors = validateContent(roadmap([node('numpy')]), [lesson('numpy')]);
    expect(errors.join('\n')).not.toMatch(/does not resolve to its id/);
  });
});

describe('isPendingLesson', () => {
  it('recognises a missing-lesson error', () => {
    expect(isPendingLesson('Node "numpy" has a missing lesson file (expected slug "numpy").')).toBe(true);
  });

  it('does not treat other errors as pending', () => {
    expect(isPendingLesson('Orphan lesson content/lessons/00-x.md is not referenced by any roadmap node.')).toBe(false);
    expect(isPendingLesson('Prerequisite cycle detected: a -> b -> a.')).toBe(false);
  });
});

describe('the real content directory', () => {
  // roadmap.json declares all 34 lessons from Task 3 onward, but the lesson
  // files arrive incrementally in Tasks 11-15. A node without its file yet is
  // "pending", not corruption. Every OTHER rule must hold at all times.
  const errors = validateContent(readRoadmap(), readLessonFiles());

  it('has no errors other than lessons not yet written', () => {
    expect(errors.filter((e) => !isPendingLesson(e))).toEqual([]);
  });

  it('reports one pending error per unwritten lesson', () => {
    const written = readLessonFiles().length;
    expect(errors.filter(isPendingLesson)).toHaveLength(34 - written);
  });
});
