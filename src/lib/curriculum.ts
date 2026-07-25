import roadmapData from '../../content/roadmap.json';
import { roadmapSchema, topologicalOrder } from './roadmap';

export interface NavLesson {
  slug: string;
  title: string;
  order: number;
  minutes: number;
  difficulty: string;
  summary: string;
}

export interface NavStage {
  id: number;
  name: string;
  color: string;
  lessons: NavLesson[];
}

interface LessonEntry {
  id: string;
  data: {
    title: string;
    order: number;
    minutes: number;
    difficulty: string;
    summary: string;
    stage: number;
  };
}

/**
 * Groups the lessons collection into stages, in dependency order, for the
 * documentation nav and the curriculum landing page. Pure over roadmap.json —
 * never imports `astro:content`, so build scripts could reuse it too.
 */
export function buildCurriculum(lessons: LessonEntry[]): NavStage[] {
  const roadmap = roadmapSchema.parse(roadmapData);
  const ordered = topologicalOrder(roadmap.nodes);
  const byId = new Map(lessons.map((l) => [l.id, l]));

  const stages: NavStage[] = roadmap.stages.map((s) => ({ id: s.id, name: s.name, color: s.color, lessons: [] }));
  const stageById = new Map(stages.map((s) => [s.id, s]));

  for (const node of ordered) {
    const lesson = byId.get(node.id);
    if (!lesson) continue;
    stageById.get(node.stage)?.lessons.push({
      slug: lesson.id,
      title: lesson.data.title,
      order: lesson.data.order,
      minutes: lesson.data.minutes,
      difficulty: lesson.data.difficulty,
      summary: lesson.data.summary,
    });
  }

  return stages.filter((s) => s.lessons.length > 0);
}

/** Total lesson count across all stages. */
export function totalLessons(stages: NavStage[]): number {
  return stages.reduce((n, s) => n + s.lessons.length, 0);
}
