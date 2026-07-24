import { describe, it, expect } from 'vitest';
import { renderCurriculum } from '../../scripts/generate-curriculum';
import { readRoadmap, readLessonFiles } from '../../scripts/lib/content-io';
import type { Roadmap } from '../../src/lib/roadmap';
import type { LessonFile } from '../../scripts/lib/content-io';

describe('renderCurriculum', () => {
  const markdown = renderCurriculum(readRoadmap(), readLessonFiles());

  it('starts with a generated-file warning', () => {
    expect(markdown).toMatch(/generated/i);
    expect(markdown).toMatch(/do not edit/i);
  });

  it('includes a heading for every stage that has lessons', () => {
    expect(markdown).toContain('## Stage 0 · Orientation');
  });

  it('links every lesson by its relative file path', () => {
    expect(markdown).toContain('(content/lessons/00-what-is-ai.md)');
  });

  it('shows the estimated reading time for each lesson', () => {
    expect(markdown).toMatch(/20 min/);
  });

  it('lists lessons in prerequisite order', () => {
    const first = markdown.indexOf('00-what-is-ai.md');
    const second = markdown.indexOf('01-how-to-learn-ai.md');
    expect(first).toBeGreaterThan(-1);
    expect(second).toBeGreaterThan(first);
  });

  it('reports the total lesson count and total time', () => {
    expect(markdown).toMatch(/\*\*\d+ lessons\*\*/);
  });
});

describe('renderCurriculum with synthetic fixtures', () => {
  /** Minimal two-stage roadmap: stage 0 gets a lesson, stage 1 never does. */
  const roadmap: Roadmap = {
    stages: [
      { id: 0, name: 'Orientation', color: 'violet' },
      { id: 1, name: 'Python & Data', color: 'indigo' },
    ],
    nodes: [
      { id: 'what-is-ai', lesson: '00-what-is-ai', stage: 0, grid: { x: 0, y: 0 }, prerequisites: [] },
      { id: 'python-basics', lesson: '01-python-basics', stage: 1, grid: { x: 0, y: 1 }, prerequisites: [] },
    ],
  };

  function lessonFile(overrides: Partial<LessonFile['frontmatter']> = {}): LessonFile {
    return {
      file: 'content/lessons/00-what-is-ai.md',
      slug: 'what-is-ai',
      frontmatter: {
        title: 'What Is AI',
        stage: 0,
        order: 0,
        minutes: 20,
        difficulty: 'beginner',
        prerequisites: [],
        tags: [],
        summary: 'A short summary.',
        ...overrides,
      },
      body: '',
    };
  }

  it('emits neither a heading nor a table header for a stage with no renderable lessons', () => {
    // Only the stage-0 node has a matching lesson file; stage 1's node is declared
    // in the roadmap but has no lesson on disk, so it must render nothing at all.
    const markdown = renderCurriculum(roadmap, [lessonFile()]);
    expect(markdown).not.toContain('Stage 1');
    expect(markdown).not.toContain('Python & Data');
  });

  it('emits both a heading and a table header for a stage that has lessons', () => {
    const markdown = renderCurriculum(roadmap, [lessonFile()]);
    expect(markdown).toContain('## Stage 0 · Orientation');
    expect(markdown).toContain('| # | Lesson | Time | Level | What it covers |');
  });

  it('pluralizes "lesson" for a single lesson', () => {
    const markdown = renderCurriculum(roadmap, [lessonFile()]);
    expect(markdown).toMatch(/\*\*1 lesson\*\* /);
    expect(markdown).not.toMatch(/\*\*1 lessons\*\*/);
  });

  it('pluralizes "lessons" for more than one lesson', () => {
    const twoStageRoadmap: Roadmap = {
      stages: [{ id: 0, name: 'Orientation', color: 'violet' }],
      nodes: [
        { id: 'what-is-ai', lesson: '00-what-is-ai', stage: 0, grid: { x: 0, y: 0 }, prerequisites: [] },
        { id: 'how-to-learn-ai', lesson: '01-how-to-learn-ai', stage: 0, grid: { x: 1, y: 0 }, prerequisites: [] },
      ],
    };
    const markdown = renderCurriculum(twoStageRoadmap, [
      lessonFile(),
      {
        ...lessonFile({ title: 'How To Learn AI', order: 1 }),
        file: 'content/lessons/01-how-to-learn-ai.md',
        slug: 'how-to-learn-ai',
      },
    ]);
    expect(markdown).toMatch(/\*\*2 lessons\*\*/);
  });

  it('reports minutes, not "0 hours", for a total under an hour', () => {
    const markdown = renderCurriculum(roadmap, [lessonFile({ minutes: 20 })]);
    expect(markdown).toMatch(/roughly 20 minutes? of reading/);
    expect(markdown).not.toMatch(/0 hours/);
  });

  it('pluralizes "hour" for a total that rounds to exactly one hour', () => {
    const markdown = renderCurriculum(roadmap, [lessonFile({ minutes: 65 })]);
    expect(markdown).toMatch(/roughly 1 hour of reading/);
    expect(markdown).not.toMatch(/roughly 1 hours/);
  });

  it('pluralizes "hours" for a total of more than one hour', () => {
    const markdown = renderCurriculum(roadmap, [lessonFile({ minutes: 130 })]);
    expect(markdown).toMatch(/roughly 2 hours of reading/);
  });
});
