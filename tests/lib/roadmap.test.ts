import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import {
  detectCycle,
  topologicalOrder,
  unlockedIds,
  nodeState,
  roadmapSchema,
  type RoadmapNode,
} from '../../src/lib/roadmap';

const node = (id: string, prerequisites: string[] = []): RoadmapNode => ({
  id,
  lesson: id,
  stage: 0,
  grid: { x: 0, y: 0 },
  prerequisites,
});

describe('detectCycle', () => {
  it('returns null for an acyclic graph', () => {
    expect(detectCycle([node('a'), node('b', ['a']), node('c', ['b'])])).toBeNull();
  });

  it('finds a direct cycle', () => {
    const cycle = detectCycle([node('a', ['b']), node('b', ['a'])]);
    expect(cycle).not.toBeNull();
    expect(cycle!.length).toBeGreaterThan(1);
  });

  it('finds a longer cycle', () => {
    expect(detectCycle([node('a', ['c']), node('b', ['a']), node('c', ['b'])])).not.toBeNull();
  });
});

describe('topologicalOrder', () => {
  it('places prerequisites before dependants', () => {
    const ordered = topologicalOrder([node('c', ['b']), node('a'), node('b', ['a'])]);
    expect(ordered.map((n) => n.id)).toEqual(['a', 'b', 'c']);
  });

  it('throws when the graph has a cycle', () => {
    expect(() => topologicalOrder([node('a', ['b']), node('b', ['a'])])).toThrow(/cycle/i);
  });
});

describe('unlockedIds', () => {
  const graph = [node('a'), node('b', ['a']), node('c', ['a', 'b'])];

  it('unlocks nodes with no prerequisites', () => {
    expect(unlockedIds(graph, [])).toEqual(new Set(['a']));
  });

  it('unlocks a node once every prerequisite is complete', () => {
    expect(unlockedIds(graph, ['a'])).toEqual(new Set(['a', 'b']));
  });

  it('keeps a node locked while any prerequisite is missing', () => {
    expect(unlockedIds(graph, ['a']).has('c')).toBe(false);
    expect(unlockedIds(graph, ['a', 'b']).has('c')).toBe(true);
  });
});

describe('nodeState', () => {
  const graph = [node('a'), node('b', ['a'])];

  it('reports complete for a finished node', () => {
    expect(nodeState(graph[0], ['a'], unlockedIds(graph, ['a']))).toBe('complete');
  });

  it('reports available for an unlocked but unfinished node', () => {
    expect(nodeState(graph[1], ['a'], unlockedIds(graph, ['a']))).toBe('available');
  });

  it('reports locked when prerequisites are unmet', () => {
    expect(nodeState(graph[1], [], unlockedIds(graph, []))).toBe('locked');
  });
});

describe('content/roadmap.json', () => {
  const raw = JSON.parse(readFileSync('content/roadmap.json', 'utf8'));
  const roadmap = roadmapSchema.parse(raw);

  it('has 34 nodes', () => {
    expect(roadmap.nodes).toHaveLength(34);
  });

  it('is acyclic', () => {
    expect(detectCycle(roadmap.nodes)).toBeNull();
  });

  it('can be ordered topologically', () => {
    expect(topologicalOrder(roadmap.nodes)).toHaveLength(34);
  });

  it('has exactly one starting node', () => {
    const roots = roadmap.nodes.filter((n) => n.prerequisites.length === 0);
    expect(roots.map((n) => n.id)).toEqual(['what-is-ai']);
  });

  it('gives every node a unique grid position', () => {
    const positions = roadmap.nodes.map((n) => `${n.grid.x},${n.grid.y}`);
    expect(new Set(positions).size).toBe(positions.length);
  });
});
