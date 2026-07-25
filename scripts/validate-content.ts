import { lessonSchema, stripOrderPrefix } from '../src/lib/lesson';
import { detectCycle, type Roadmap } from '../src/lib/roadmap';
import { readRoadmap, readLessonFiles, type LessonFile } from './lib/content-io';

/**
 * Every rule that must hold between roadmap.json and the lesson files.
 * Returns human-readable errors; an empty array means the content is sound.
 */
export function validateContent(roadmap: Roadmap, lessons: LessonFile[]): string[] {
  const errors: string[] = [];
  const nodeIds = new Set(roadmap.nodes.map((n) => n.id));
  const bySlug = new Map(lessons.map((l) => [l.slug, l]));
  const stageIds = new Set(roadmap.stages.map((s) => s.id));

  for (const node of roadmap.nodes) {
    if (!bySlug.has(node.id)) {
      errors.push(`Node "${node.id}" has a missing lesson file (expected slug "${node.id}").`);
    }
    if (!stageIds.has(node.stage)) {
      errors.push(`Node "${node.id}" references unknown stage ${node.stage}.`);
    }
    for (const prereq of node.prerequisites) {
      if (!nodeIds.has(prereq)) {
        errors.push(`Node "${node.id}" lists unknown prerequisite "${prereq}".`);
      }
    }
    const resolved = stripOrderPrefix(node.lesson);
    if (resolved !== node.id) {
      errors.push(
        `Node "${node.id}" lesson field "${node.lesson}" does not resolve to its id (got "${resolved}").`,
      );
    }
  }

  for (const lesson of lessons) {
    if (!nodeIds.has(lesson.slug)) {
      errors.push(`Orphan lesson ${lesson.file} is not referenced by any roadmap node.`);
    }
  }

  const seen = new Map<string, string>();
  for (const node of roadmap.nodes) {
    const key = `${node.grid.x},${node.grid.y}`;
    const previous = seen.get(key);
    if (previous) {
      errors.push(`Nodes "${previous}" and "${node.id}" share grid position ${key}.`);
    }
    seen.set(key, node.id);
  }

  const cycle = detectCycle(roadmap.nodes);
  if (cycle) errors.push(`Prerequisite cycle detected: ${cycle.join(' -> ')}.`);

  for (const lesson of lessons) {
    const parsed = lessonSchema.safeParse(lesson.frontmatter);
    if (!parsed.success) {
      for (const issue of parsed.error.issues) {
        errors.push(`${lesson.file}: frontmatter ${issue.path.join('.')} — ${issue.message}`);
      }
      continue;
    }

    const node = roadmap.nodes.find((n) => n.id === lesson.slug);
    if (!node) continue;

    if (parsed.data.stage !== node.stage) {
      errors.push(
        `${lesson.file}: frontmatter stage ${parsed.data.stage} does not match roadmap stage ${node.stage}.`,
      );
    }

    const declared = [...parsed.data.prerequisites].sort().join(',');
    const graph = [...node.prerequisites].sort().join(',');
    if (declared !== graph) {
      errors.push(
        `${lesson.file}: frontmatter prerequisites [${declared}] do not match roadmap prerequisites [${graph}].`,
      );
    }
  }

  return errors;
}

/**
 * A node whose lesson file has not been written yet.
 *
 * roadmap.json declares the whole 34-lesson curriculum up front, while the
 * lesson files land incrementally. That is a known-incomplete state, not
 * corruption, so it does not fail the build unless --strict is passed.
 */
export function isPendingLesson(error: string): boolean {
  return error.includes('has a missing lesson file');
}

function main(): void {
  const strict = process.argv.includes('--strict');
  const errors = validateContent(readRoadmap(), readLessonFiles());
  const pending = errors.filter(isPendingLesson);
  const hard = strict ? errors : errors.filter((e) => !isPendingLesson(e));

  if (!strict && pending.length > 0) {
    console.warn(`\n${pending.length} lesson(s) not written yet:\n`);
    for (const error of pending) console.warn(`  ⚠ ${error}`);
    console.warn('');
  }

  if (hard.length > 0) {
    console.error(`\nContent validation failed with ${hard.length} error(s):\n`);
    for (const error of hard) console.error(`  ✗ ${error}`);
    console.error('');
    process.exit(1);
  }

  console.log(
    pending.length > 0 && !strict
      ? `✓ Content validation passed (${pending.length} lesson(s) still pending).`
      : '✓ Content validation passed.',
  );
}

if (import.meta.url === `file://${process.argv[1]}`) main();
