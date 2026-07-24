import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { stripOrderPrefix, lessonSchema, type LessonFrontmatter } from './lib/lesson';

export { stripOrderPrefix, lessonSchema, type LessonFrontmatter };

const lessons = defineCollection({
  loader: glob({
    pattern: '**/[^_]*.md',
    base: './content/lessons',
    generateId: ({ entry }) => stripOrderPrefix(entry),
  }),
  schema: lessonSchema,
});

export const collections = { lessons };
