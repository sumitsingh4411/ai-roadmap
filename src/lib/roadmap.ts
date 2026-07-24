import { z } from 'astro/zod';

export const roadmapStageSchema = z.object({
  id: z.number().int().min(0).max(6),
  name: z.string().min(1),
  color: z.string().min(1),
});

export const roadmapNodeSchema = z.object({
  id: z.string().min(1),
  lesson: z.string().min(1),
  stage: z.number().int().min(0).max(6),
  grid: z.object({ x: z.number().int(), y: z.number().int() }),
  prerequisites: z.array(z.string()).default([]),
});

export const roadmapSchema = z.object({
  stages: z.array(roadmapStageSchema).min(1),
  nodes: z.array(roadmapNodeSchema).min(1),
});

export type RoadmapStage = z.infer<typeof roadmapStageSchema>;
export type RoadmapNode = z.infer<typeof roadmapNodeSchema>;
export type Roadmap = z.infer<typeof roadmapSchema>;
export type NodeState = 'complete' | 'available' | 'locked';

/**
 * Depth-first search for a prerequisite cycle.
 * Returns the offending path (e.g. ['a', 'b', 'a']) or null when acyclic.
 */
export function detectCycle(nodes: RoadmapNode[]): string[] | null {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const visiting = new Set<string>();
  const visited = new Set<string>();
  const path: string[] = [];

  function walk(id: string): string[] | null {
    if (visiting.has(id)) return [...path.slice(path.indexOf(id)), id];
    if (visited.has(id)) return null;

    visiting.add(id);
    path.push(id);

    for (const prereq of byId.get(id)?.prerequisites ?? []) {
      if (!byId.has(prereq)) continue; // unknown ids are reported by validate-content
      const found = walk(prereq);
      if (found) return found;
    }

    path.pop();
    visiting.delete(id);
    visited.add(id);
    return null;
  }

  for (const n of nodes) {
    const found = walk(n.id);
    if (found) return found;
  }
  return null;
}

/**
 * Kahn's algorithm. Ties are broken by the node's position in the input array,
 * so the output is deterministic across runs.
 */
export function topologicalOrder(nodes: RoadmapNode[]): RoadmapNode[] {
  const cycle = detectCycle(nodes);
  if (cycle) {
    throw new Error(`Roadmap contains a prerequisite cycle: ${cycle.join(' -> ')}`);
  }

  const known = new Set(nodes.map((n) => n.id));
  const remaining = [...nodes];
  const done = new Set<string>();
  const ordered: RoadmapNode[] = [];

  while (remaining.length > 0) {
    const index = remaining.findIndex((n) =>
      n.prerequisites.every((p) => !known.has(p) || done.has(p)),
    );
    if (index === -1) {
      throw new Error('Roadmap contains a prerequisite cycle');
    }
    const [next] = remaining.splice(index, 1);
    done.add(next.id);
    ordered.push(next);
  }

  return ordered;
}

/** Ids the learner is allowed to start: every prerequisite is complete. */
export function unlockedIds(nodes: RoadmapNode[], completed: string[]): Set<string> {
  const done = new Set(completed);
  const unlocked = new Set<string>();
  for (const n of nodes) {
    if (n.prerequisites.every((p) => done.has(p))) unlocked.add(n.id);
  }
  return unlocked;
}

export function nodeState(
  node: RoadmapNode,
  completed: string[],
  unlocked: Set<string>,
): NodeState {
  if (completed.includes(node.id)) return 'complete';
  return unlocked.has(node.id) ? 'available' : 'locked';
}
