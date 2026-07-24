import { describe, it, expect } from 'vitest';
import { renderCurriculum } from '../../scripts/generate-curriculum';
import { readRoadmap, readLessonFiles } from '../../scripts/lib/content-io';

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
