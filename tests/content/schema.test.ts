import { describe, it, expect } from 'vitest';
import { lessonSchema, stripOrderPrefix } from '../../src/content.config';

const valid = {
  title: 'NumPy',
  stage: 1,
  order: 4,
  minutes: 40,
  difficulty: 'beginner',
  prerequisites: ['python-basics'],
  tags: ['python', 'data'],
  summary: 'Arrays, shapes, and vectorised thinking.',
};

describe('lessonSchema', () => {
  it('accepts a well-formed lesson', () => {
    expect(lessonSchema.parse(valid)).toMatchObject({ title: 'NumPy', stage: 1 });
  });

  it('defaults prerequisites and tags to empty arrays', () => {
    const { prerequisites, tags, ...rest } = valid;
    const parsed = lessonSchema.parse(rest);
    expect(parsed.prerequisites).toEqual([]);
    expect(parsed.tags).toEqual([]);
  });

  it('rejects an unknown difficulty', () => {
    expect(() => lessonSchema.parse({ ...valid, difficulty: 'expert' })).toThrow();
  });

  it('rejects a stage outside 0-6', () => {
    expect(() => lessonSchema.parse({ ...valid, stage: 7 })).toThrow();
  });

  it('rejects a missing title', () => {
    const { title, ...rest } = valid;
    expect(() => lessonSchema.parse(rest)).toThrow();
  });

  it('rejects a summary longer than 200 characters', () => {
    expect(() => lessonSchema.parse({ ...valid, summary: 'x'.repeat(201) })).toThrow();
  });
});

describe('stripOrderPrefix', () => {
  it('removes the numeric prefix and extension', () => {
    expect(stripOrderPrefix('04-numpy.md')).toBe('numpy');
  });

  it('leaves a file without a prefix alone', () => {
    expect(stripOrderPrefix('numpy.md')).toBe('numpy');
  });

  it('handles nested paths', () => {
    expect(stripOrderPrefix('extra/12-rag.md')).toBe('rag');
  });
});
