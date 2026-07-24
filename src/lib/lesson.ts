import { z } from 'astro/zod';

/** Turns `content/lessons/04-numpy.md` into the slug `numpy`. */
export function stripOrderPrefix(entry: string): string {
  const file = entry.split('/').pop() ?? entry;
  return file.replace(/\.md$/, '').replace(/^\d+-/, '');
}

export const lessonSchema = z.object({
  title: z.string().min(1),
  stage: z.number().int().min(0).max(6),
  order: z.number().int().min(0),
  minutes: z.number().int().positive(),
  difficulty: z.enum(['beginner', 'intermediate', 'advanced']),
  prerequisites: z.array(z.string()).default([]),
  tags: z.array(z.string()).default([]),
  summary: z.string().min(1).max(200),
});

export type LessonFrontmatter = z.infer<typeof lessonSchema>;
