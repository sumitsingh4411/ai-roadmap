# AI Learning Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a free, open-source AI learning roadmap whose 34 Markdown lessons read natively on github.com, published as a static Astro site with an isometric 3D roadmap, progress tracking, and search.

**Architecture:** Markdown files in `content/lessons/` are the canonical product. Astro reads them through a Content Layer `glob()` loader whose `base` points outside `src/`, generating one static page per lesson plus an isometric homepage driven by `content/roadmap.json`. All interactivity (progress, search, theme, tilt) is client-side over `localStorage` and a build-time JSON index — no backend, no network calls after load.

**Tech Stack:** Astro 7.1.3, TypeScript, Zod (via `astro/zod`), Vitest, tsx, Shiki (built into Astro), GitHub Actions → GitHub Pages.

## Global Constraints

- **Node `>=22.12.0`** — required by Astro 7. Local dev is on v22.22.1.
- **Astro `^7.1.3`** — Content Layer API only. The legacy content collections API was removed in v6; `type: 'content'` and `legacy.collections` do not exist.
- **Zod is imported from `astro/zod`**, never from a separate `zod` package. Applies to `src/content.config.ts`, `src/lib/*`, and `scripts/*`.
- **Content collection loader:** `glob()` imported from `astro/loaders`.
- **Rendering:** `render(entry)` imported from `astro:content`. There is no `entry.render()` method.
- **Lesson Markdown is portable only** — headings, fenced code blocks with language hints, tables, blockquotes, links, images. **No MDX, no Astro components, no raw HTML, no import statements.** Every lesson must render correctly on github.com.
- **Site config:** `site: 'https://sumitsingh4411.github.io'`, `base: '/ai-roadmap'`.
- **No hardcoded internal URLs.** Every internal link is built through `href()` from `src/lib/url.ts`, which reads `import.meta.env.BASE_URL`.
- **Repo:** `github.com/sumitsingh4411/ai-roadmap`. GitHub links in lesson pages point at `blob/main/content/lessons/<file>.md`.
- **Lesson slugs contain no numeric prefix.** File `04-numpy.md` produces slug `numpy`. Roadmap node `id` values must equal these slugs exactly.
- **`localStorage` key:** `ai-roadmap:progress`.
- **All animation respects `prefers-reduced-motion: reduce`.**
- **Commit after every task.** Conventional commit prefixes (`feat:`, `test:`, `chore:`, `docs:`, `content:`).

---

## File Structure

| Path | Responsibility |
|---|---|
| `content/roadmap.json` | The node graph — stages, nodes, grid positions, prerequisites. Hand-edited. |
| `content/lessons/*.md` | 34 canonical lessons. The product. |
| `CURRICULUM.md` | Generated ordered index for GitHub-only readers. Never hand-edited. |
| `README.md` | Project intro, "learn from GitHub" pitch, contribution notes. |
| `src/content.config.ts` | Collection definition + Zod frontmatter schema. |
| `src/lib/url.ts` | `href()` base-path helper. |
| `src/lib/roadmap.ts` | Graph types, cycle detection, topological order, unlock logic. Pure, no I/O. |
| `src/lib/progress.ts` | `localStorage` progress store. Injectable storage for testing. |
| `src/lib/search.ts` | Client-side query over the prebuilt index. |
| `src/styles/theme.css` | CSS custom properties for dark/light. |
| `src/styles/global.css` | Reset, typography, prose styles. |
| `src/layouts/BaseLayout.astro` | HTML shell, head, theme bootstrap, header/footer. |
| `src/components/IsometricBoard.astro` | The 3D roadmap board. |
| `src/components/RoadmapTile.astro` | A single extruded node tile. |
| `src/components/LessonCard.astro` | Depth-stack lesson header. |
| `src/components/SearchDialog.astro` | ⌘K overlay. |
| `src/components/ThemeToggle.astro` | Dark/light switch. |
| `src/pages/index.astro` | Homepage. |
| `src/pages/lessons/[slug].astro` | Lesson reader. |
| `src/pages/404.astro` | Not-found page. |
| `scripts/lib/content-io.ts` | Shared filesystem reads for scripts (roadmap + lesson files). |
| `scripts/validate-content.ts` | Integrity checks. Pure `validateContent()` + CLI wrapper. |
| `scripts/build-search-index.ts` | Emits `public/search-index.json`. |
| `scripts/generate-curriculum.ts` | Emits `CURRICULUM.md`. |
| `.github/workflows/deploy.yml` | Validate → test → build → deploy to Pages. |

Scripts keep their pure logic exported and their filesystem access in `scripts/lib/content-io.ts`, so every rule is unit-testable without touching disk.

---

### Task 1: Scaffold Astro and verify a clean build

**Files:**
- Create: `package.json`, `astro.config.mjs`, `tsconfig.json`, `vitest.config.ts`
- Create: `src/pages/index.astro`
- Create: `src/lib/url.ts`
- Test: `tests/lib/url.test.ts`

**Interfaces:**
- Consumes: nothing.
- Produces: `href(path: string): string` from `src/lib/url.ts` — joins `import.meta.env.BASE_URL` with a path, collapsing duplicate slashes. Every later task uses this for internal links.

- [ ] **Step 1: Create `package.json`**

```json
{
  "name": "ai-roadmap",
  "type": "module",
  "version": "0.1.0",
  "engines": { "node": ">=22.12.0" },
  "scripts": {
    "dev": "astro dev",
    "prebuild": "npm run validate && npm run gen:curriculum && npm run gen:search",
    "build": "astro build",
    "preview": "astro preview",
    "validate": "tsx scripts/validate-content.ts",
    "gen:curriculum": "tsx scripts/generate-curriculum.ts",
    "gen:search": "tsx scripts/build-search-index.ts",
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "dependencies": {
    "astro": "^7.1.3"
  },
  "devDependencies": {
    "tsx": "^4.19.2",
    "typescript": "^5.7.2",
    "vitest": "^2.1.8"
  }
}
```

- [ ] **Step 2: Install dependencies**

Run: `npm install`
Expected: completes with no `ERESOLVE` errors. `node_modules/astro/package.json` reports version 7.x.

Verify: `npx astro --version`
Expected: `7.1.3` or higher.

- [ ] **Step 3: Create `astro.config.mjs`**

```js
// @ts-check
import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://sumitsingh4411.github.io',
  base: '/ai-roadmap',
  trailingSlash: 'ignore',
  markdown: {
    shikiConfig: {
      themes: { light: 'github-light', dark: 'github-dark-dimmed' },
      wrap: true,
    },
  },
});
```

- [ ] **Step 4: Create `tsconfig.json`**

```json
{
  "extends": "astro/tsconfigs/strict",
  "include": [".astro/types.d.ts", "**/*"],
  "exclude": ["dist"],
  "compilerOptions": {
    "strictNullChecks": true,
    "allowJs": true
  }
}
```

- [ ] **Step 5: Create `vitest.config.ts`**

```ts
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts'],
  },
});
```

- [ ] **Step 6: Write the failing test for `href()`**

Create `tests/lib/url.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import { joinBase } from '../../src/lib/url';

describe('joinBase', () => {
  it('joins a base path and a route without doubling slashes', () => {
    expect(joinBase('/ai-roadmap/', '/lessons/numpy')).toBe('/ai-roadmap/lessons/numpy');
  });

  it('handles a base without a trailing slash', () => {
    expect(joinBase('/ai-roadmap', '/lessons/numpy')).toBe('/ai-roadmap/lessons/numpy');
  });

  it('handles a route without a leading slash', () => {
    expect(joinBase('/ai-roadmap/', 'lessons/numpy')).toBe('/ai-roadmap/lessons/numpy');
  });

  it('returns the base itself for the root route', () => {
    expect(joinBase('/ai-roadmap/', '/')).toBe('/ai-roadmap/');
  });

  it('works when deployed at the domain root', () => {
    expect(joinBase('/', '/lessons/numpy')).toBe('/lessons/numpy');
  });
});
```

- [ ] **Step 7: Run the test to verify it fails**

Run: `npx vitest run tests/lib/url.test.ts`
Expected: FAIL — `Failed to resolve import "../../src/lib/url"`.

- [ ] **Step 8: Implement `src/lib/url.ts`**

```ts
/**
 * Joins Astro's configured base path with an internal route.
 * Exported separately from `href` so it can be unit-tested without
 * `import.meta.env`, which only exists inside the Astro build.
 */
export function joinBase(base: string, path: string): string {
  const cleanBase = base.endsWith('/') ? base.slice(0, -1) : base;
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  if (cleanPath === '/') return `${cleanBase}/`;
  return `${cleanBase}${cleanPath}`;
}

/** Builds an internal link that is correct under the deployed base path. */
export function href(path: string): string {
  return joinBase(import.meta.env.BASE_URL, path);
}
```

- [ ] **Step 9: Run the test to verify it passes**

Run: `npx vitest run tests/lib/url.test.ts`
Expected: PASS — 5 tests.

- [ ] **Step 10: Create a placeholder `src/pages/index.astro`**

```astro
---
const title = 'AI Roadmap';
---
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
  </head>
  <body>
    <h1>AI Roadmap</h1>
  </body>
</html>
```

- [ ] **Step 11: Verify the build succeeds**

Run: `npx astro build`
Expected: `Complete!` with `dist/index.html` written and zero errors.

- [ ] **Step 12: Commit**

```bash
git add package.json package-lock.json astro.config.mjs tsconfig.json vitest.config.ts src tests
git commit -m "chore: scaffold Astro 7 project with base-path helper"
```

---

### Task 2: Content collection, frontmatter schema, and the two Stage 0 lessons

**Files:**
- Create: `src/content.config.ts`
- Create: `content/lessons/00-what-is-ai.md`
- Create: `content/lessons/01-how-to-learn-ai.md`
- Test: `tests/content/schema.test.ts`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: the `lessons` collection, queryable via `getCollection('lessons')`. Each entry has `id` (slug with the numeric prefix stripped, e.g. `what-is-ai`) and `data` matching `lessonSchema`. Also exports `lessonSchema` and the `LessonFrontmatter` type, which `scripts/*` reuse.

- [ ] **Step 1: Write the failing schema test**

Create `tests/content/schema.test.ts`:

```ts
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run tests/content/schema.test.ts`
Expected: FAIL — cannot resolve `../../src/content.config`.

- [ ] **Step 3: Implement `src/content.config.ts`**

```ts
import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
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

const lessons = defineCollection({
  loader: glob({
    pattern: '**/[^_]*.md',
    base: './content/lessons',
    generateId: ({ entry }) => stripOrderPrefix(entry),
  }),
  schema: lessonSchema,
});

export const collections = { lessons };
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run tests/content/schema.test.ts`
Expected: PASS — 9 tests.

Note: if Vitest fails resolving `astro:content`, that import is virtual and only exists during an Astro build. Fix by keeping the test importing only `lessonSchema` and `stripOrderPrefix` — both are defined before any `astro:content` value is used at module scope, so the import is hoisted but never evaluated at runtime. If resolution still fails, add to `vitest.config.ts`:

```ts
resolve: { alias: { 'astro:content': new URL('./tests/stubs/astro-content.ts', import.meta.url).pathname } }
```

and create `tests/stubs/astro-content.ts`:

```ts
export function defineCollection<T>(config: T): T {
  return config;
}
```

- [ ] **Step 5: Write `content/lessons/00-what-is-ai.md`**

This is the reference lesson — every later lesson copies its shape. Write it in full:

```markdown
---
title: "What AI, ML, Deep Learning and GenAI Actually Are"
stage: 0
order: 0
minutes: 20
difficulty: beginner
prerequisites: []
tags: ["foundations", "orientation"]
summary: "The four words everyone mixes up, sorted out once, with a mental model you can keep."
---

# What AI, ML, Deep Learning and GenAI Actually Are

## Why this matters

Almost every confusing article about AI is confusing because it uses four words
interchangeably that mean four different things. Sort them out now and the rest
of this roadmap stops feeling like jargon.

## The concept

Think of four nested boxes, each inside the one before it.

**Artificial Intelligence** is the outermost box: any program that does something
we would call "intelligent" if a person did it. A chess engine from 1997 counts.
A thermostat that learns your schedule counts.

**Machine Learning** sits inside AI. Instead of a human writing the rules, you
show the program examples and it derives the rules itself. You do not write
"if the email contains 'free money', mark it as spam." You show it 100,000 emails
labelled spam or not-spam, and it works out the pattern.

**Deep Learning** sits inside ML. It is machine learning done with neural
networks that have many layers stacked on top of each other. Each layer learns
something slightly more abstract than the one below it. In an image model, early
layers find edges, middle layers find shapes, late layers find faces.

**Generative AI** sits inside deep learning. These are models that produce new
content — text, images, audio, code — rather than just classifying or predicting
a number. ChatGPT and image generators live here.

| Term | What it is | Example |
|---|---|---|
| AI | Any "intelligent" program | Chess engine, route planner |
| ML | Learns rules from examples | Spam filter, price predictor |
| Deep Learning | ML with deep neural networks | Face recognition, speech-to-text |
| GenAI | Deep learning that creates content | ChatGPT, image generators |

The key jump is from AI to ML: **who writes the rules.** In classical AI, a human
does. In machine learning, the data does.

## In code

You do not need to understand this code yet. Read it as a picture of what
"learning from examples" means in practice.

```python
from sklearn.linear_model import LinearRegression

# Examples: house size in square metres -> price in thousands
sizes = [[50], [80], [110], [140], [170]]
prices = [150, 220, 300, 370, 450]

model = LinearRegression()
model.fit(sizes, prices)          # this line IS the "learning"

print(model.predict([[100]]))
```

```
[277.]
```

Nobody told the model that bigger houses cost more. It found that rule in the
five examples it was given. That is the entire idea of machine learning,
and everything later in this roadmap is a more powerful version of it.

## Build this

Change the five example prices so that price goes *down* as size goes up, then
re-run the code. Predict what the model will output for a 100 m² house before you
run it, then check whether you were right.

**Stretch:** add a sixth example that badly contradicts the others (a 60 m² house
at 900) and observe how much the prediction moves. You have just discovered why
data quality matters more than model choice.

## Go deeper

- [Google's Introduction to Machine Learning](https://developers.google.com/machine-learning/intro-to-ml) — 20 minutes, no maths.
- [3Blue1Brown: But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) — the best visual explanation of deep learning ever made.
- [Elements of AI](https://www.elementsofai.com/) — free university course for absolute beginners.

**Next:** [How to Learn AI Without Burning Out](01-how-to-learn-ai.md)
```

- [ ] **Step 6: Write `content/lessons/01-how-to-learn-ai.md`**

Same five-part structure. Frontmatter exactly:

```yaml
---
title: "How to Learn AI Without Burning Out"
stage: 0
order: 1
minutes: 15
difficulty: beginner
prerequisites: ["what-is-ai"]
tags: ["foundations", "study-plan"]
summary: "A realistic schedule, the order to learn things in, and the three traps that stop most beginners."
---
```

Body must cover: the maths-first trap (why to write code before proving theorems), how much time per week is realistic, why projects beat tutorials, the three common quitting points, and how to use this roadmap's prerequisite graph. Include a weekly-schedule table in the "The concept" section, and a "Build this" step that asks the reader to write their own 12-week plan into a file. End with a `**Next:**` link to `02-python-basics.md`.

- [ ] **Step 7: Verify the collection loads and builds**

Run: `npx astro build`
Expected: `Complete!`, and the build log reports the `lessons` collection syncing 2 entries with no schema errors.

- [ ] **Step 8: Verify GitHub rendering locally**

Run: `npx astro build 2>&1 | grep -i "error\|warn" || echo "clean"`
Expected: `clean`

Manually confirm both lesson files contain no HTML tags, no `import` statements, and no `{}` expressions — only portable Markdown.

- [ ] **Step 9: Commit**

```bash
git add src/content.config.ts content/lessons tests/content
git commit -m "feat: add lessons collection schema and Stage 0 lessons"
```

---

### Task 3: The roadmap graph — `roadmap.json` and `src/lib/roadmap.ts`

**Files:**
- Create: `content/roadmap.json`
- Create: `src/lib/roadmap.ts`
- Test: `tests/lib/roadmap.test.ts`

**Interfaces:**
- Consumes: lesson slugs produced by Task 2's `stripOrderPrefix`.
- Produces:
  - Types `RoadmapStage`, `RoadmapNode`, `Roadmap`, `NodeState`.
  - `roadmapSchema` (Zod) for validating `roadmap.json`.
  - `detectCycle(nodes: RoadmapNode[]): string[] | null` — returns the cycle path, or `null`.
  - `topologicalOrder(nodes: RoadmapNode[]): RoadmapNode[]` — throws on a cycle.
  - `unlockedIds(nodes: RoadmapNode[], completed: string[]): Set<string>`.
  - `nodeState(node: RoadmapNode, completed: string[], unlocked: Set<string>): NodeState`.

Tasks 5, 9, 10 and 14 all consume these exact names.

- [ ] **Step 1: Write the failing graph test**

Create `tests/lib/roadmap.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import {
  detectCycle,
  topologicalOrder,
  unlockedIds,
  nodeState,
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run tests/lib/roadmap.test.ts`
Expected: FAIL — cannot resolve `../../src/lib/roadmap`.

- [ ] **Step 3: Implement `src/lib/roadmap.ts`**

```ts
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run tests/lib/roadmap.test.ts`
Expected: PASS — 11 tests.

- [ ] **Step 5: Create `content/roadmap.json` with all 34 nodes**

`grid.y` is the row (roughly the stage), `grid.x` the column. Nodes in the same stage that can be studied in parallel share a row.

```json
{
  "stages": [
    { "id": 0, "name": "Orientation",            "color": "violet" },
    { "id": 1, "name": "Python & Data",          "color": "indigo" },
    { "id": 2, "name": "Math You Actually Need", "color": "sky" },
    { "id": 3, "name": "Classical ML",           "color": "cyan" },
    { "id": 4, "name": "Deep Learning",          "color": "teal" },
    { "id": 5, "name": "Generative AI & LLMs",   "color": "amber" },
    { "id": 6, "name": "Ship It",                "color": "rose" }
  ],
  "nodes": [
    { "id": "what-is-ai",          "lesson": "00-what-is-ai",          "stage": 0, "grid": { "x": 1, "y": 0 }, "prerequisites": [] },
    { "id": "how-to-learn-ai",     "lesson": "01-how-to-learn-ai",     "stage": 0, "grid": { "x": 2, "y": 0 }, "prerequisites": ["what-is-ai"] },

    { "id": "python-basics",       "lesson": "02-python-basics",       "stage": 1, "grid": { "x": 0, "y": 1 }, "prerequisites": ["how-to-learn-ai"] },
    { "id": "numpy",               "lesson": "03-numpy",               "stage": 1, "grid": { "x": 1, "y": 1 }, "prerequisites": ["python-basics"] },
    { "id": "pandas",              "lesson": "04-pandas",              "stage": 1, "grid": { "x": 2, "y": 1 }, "prerequisites": ["numpy"] },
    { "id": "data-visualization",  "lesson": "05-data-visualization",  "stage": 1, "grid": { "x": 3, "y": 1 }, "prerequisites": ["pandas"] },
    { "id": "real-datasets",       "lesson": "06-real-datasets",       "stage": 1, "grid": { "x": 4, "y": 1 }, "prerequisites": ["pandas"] },

    { "id": "linear-algebra",      "lesson": "07-linear-algebra",      "stage": 2, "grid": { "x": 0, "y": 2 }, "prerequisites": ["how-to-learn-ai"] },
    { "id": "calculus",            "lesson": "08-calculus",            "stage": 2, "grid": { "x": 1, "y": 2 }, "prerequisites": ["linear-algebra"] },
    { "id": "probability-stats",   "lesson": "09-probability-stats",   "stage": 2, "grid": { "x": 2, "y": 2 }, "prerequisites": ["how-to-learn-ai"] },

    { "id": "ml-fundamentals",     "lesson": "10-ml-fundamentals",     "stage": 3, "grid": { "x": 0, "y": 3 }, "prerequisites": ["numpy", "probability-stats"] },
    { "id": "regression",          "lesson": "11-regression",          "stage": 3, "grid": { "x": 1, "y": 3 }, "prerequisites": ["ml-fundamentals", "linear-algebra"] },
    { "id": "classification",      "lesson": "12-classification",      "stage": 3, "grid": { "x": 2, "y": 3 }, "prerequisites": ["regression"] },
    { "id": "model-evaluation",    "lesson": "13-model-evaluation",    "stage": 3, "grid": { "x": 3, "y": 3 }, "prerequisites": ["classification"] },
    { "id": "feature-engineering", "lesson": "14-feature-engineering", "stage": 3, "grid": { "x": 4, "y": 3 }, "prerequisites": ["pandas", "model-evaluation"] },
    { "id": "clustering-pca",      "lesson": "15-clustering-pca",      "stage": 3, "grid": { "x": 5, "y": 3 }, "prerequisites": ["ml-fundamentals", "linear-algebra"] },
    { "id": "trees-ensembles",     "lesson": "16-trees-ensembles",     "stage": 3, "grid": { "x": 6, "y": 3 }, "prerequisites": ["model-evaluation"] },
    { "id": "first-ml-project",    "lesson": "17-first-ml-project",    "stage": 3, "grid": { "x": 7, "y": 3 }, "prerequisites": ["feature-engineering", "trees-ensembles", "data-visualization", "real-datasets"] },

    { "id": "neural-networks",     "lesson": "18-neural-networks",     "stage": 4, "grid": { "x": 0, "y": 4 }, "prerequisites": ["first-ml-project", "calculus"] },
    { "id": "backprop-training",   "lesson": "19-backprop-training",   "stage": 4, "grid": { "x": 1, "y": 4 }, "prerequisites": ["neural-networks"] },
    { "id": "pytorch",             "lesson": "20-pytorch",             "stage": 4, "grid": { "x": 2, "y": 4 }, "prerequisites": ["backprop-training"] },
    { "id": "cnns-vision",         "lesson": "21-cnns-vision",         "stage": 4, "grid": { "x": 3, "y": 4 }, "prerequisites": ["pytorch"] },
    { "id": "sequence-models",     "lesson": "22-sequence-models",     "stage": 4, "grid": { "x": 4, "y": 4 }, "prerequisites": ["pytorch"] },
    { "id": "transformers",        "lesson": "23-transformers",        "stage": 4, "grid": { "x": 5, "y": 4 }, "prerequisites": ["sequence-models"] },

    { "id": "how-llms-work",       "lesson": "24-how-llms-work",       "stage": 5, "grid": { "x": 0, "y": 5 }, "prerequisites": ["transformers"] },
    { "id": "prompt-engineering",  "lesson": "25-prompt-engineering",  "stage": 5, "grid": { "x": 1, "y": 5 }, "prerequisites": ["how-llms-work"] },
    { "id": "embeddings",          "lesson": "26-embeddings",          "stage": 5, "grid": { "x": 2, "y": 5 }, "prerequisites": ["how-llms-work"] },
    { "id": "rag",                 "lesson": "27-rag",                 "stage": 5, "grid": { "x": 3, "y": 5 }, "prerequisites": ["embeddings", "prompt-engineering"] },
    { "id": "fine-tuning",         "lesson": "28-fine-tuning",         "stage": 5, "grid": { "x": 4, "y": 5 }, "prerequisites": ["how-llms-work", "pytorch"] },
    { "id": "ai-agents",           "lesson": "29-ai-agents",           "stage": 5, "grid": { "x": 5, "y": 5 }, "prerequisites": ["rag"] },
    { "id": "evals-guardrails",    "lesson": "30-evals-guardrails",    "stage": 5, "grid": { "x": 6, "y": 5 }, "prerequisites": ["ai-agents"] },

    { "id": "mlops-basics",        "lesson": "31-mlops-basics",        "stage": 6, "grid": { "x": 0, "y": 6 }, "prerequisites": ["first-ml-project"] },
    { "id": "deploying-models",    "lesson": "32-deploying-models",    "stage": 6, "grid": { "x": 1, "y": 6 }, "prerequisites": ["mlops-basics"] },
    { "id": "portfolio-career",    "lesson": "33-portfolio-career",    "stage": 6, "grid": { "x": 2, "y": 6 }, "prerequisites": ["deploying-models", "evals-guardrails"] }
  ]
}
```

- [ ] **Step 6: Add a test asserting the real roadmap file is acyclic and complete**

Append to `tests/lib/roadmap.test.ts`:

```ts
import { readFileSync } from 'node:fs';
import { roadmapSchema } from '../../src/lib/roadmap';

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
```

- [ ] **Step 7: Run the full suite**

Run: `npx vitest run`
Expected: PASS — all tests across `url`, `schema`, and `roadmap`.

- [ ] **Step 8: Commit**

```bash
git add content/roadmap.json src/lib/roadmap.ts tests/lib/roadmap.test.ts
git commit -m "feat: add roadmap graph with cycle detection and unlock logic"
```

---

### Task 4: Content validation script

**Files:**
- Create: `scripts/lib/content-io.ts`
- Create: `scripts/validate-content.ts`
- Test: `tests/scripts/validate-content.test.ts`

**Interfaces:**
- Consumes: `Roadmap`, `RoadmapNode`, `roadmapSchema`, `detectCycle` from `src/lib/roadmap.ts`; `lessonSchema`, `stripOrderPrefix` from `src/content.config.ts`.
- Produces:
  - `readRoadmap(root?: string): Roadmap` and `readLessonFiles(root?: string): LessonFile[]` from `scripts/lib/content-io.ts`, where `LessonFile = { file: string; slug: string; frontmatter: unknown; body: string }`. Tasks 5 and 6 reuse both.
  - `parseFrontmatter(raw: string): { frontmatter: Record<string, unknown>; body: string }`.
  - `validateContent(roadmap: Roadmap, lessons: LessonFile[]): string[]` — returns human-readable error strings, empty when valid.

- [ ] **Step 1: Write the failing validation test**

Create `tests/scripts/validate-content.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import { validateContent } from '../../scripts/validate-content';
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

  it('reports a lesson whose frontmatter prerequisites disagree with the graph', () => {
    const errors = validateContent(
      roadmap([node('a'), node('b', ['a'], 1)]),
      [lesson('a'), { ...lesson('b'), frontmatter: { ...lesson('b').frontmatter, prerequisites: ['ghost'] } }],
    );
    expect(errors.join('\n')).toMatch(/prerequisite/i);
  });
});

describe('the real content directory', () => {
  it('passes every validation rule', () => {
    expect(validateContent(readRoadmap(), readLessonFiles())).toEqual([]);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run tests/scripts/validate-content.test.ts`
Expected: FAIL — cannot resolve `../../scripts/validate-content`.

- [ ] **Step 3: Implement `scripts/lib/content-io.ts`**

Frontmatter is parsed with a small hand-rolled reader rather than a YAML dependency, because the schema is fixed and shallow — strings, numbers, and flat arrays only.

```ts
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { roadmapSchema, type Roadmap } from '../../src/lib/roadmap';
import { stripOrderPrefix } from '../../src/content.config';

export interface LessonFile {
  file: string;
  slug: string;
  frontmatter: Record<string, unknown>;
  body: string;
}

function parseScalar(raw: string): unknown {
  const value = raw.trim();
  if (value.startsWith('[') && value.endsWith(']')) {
    const inner = value.slice(1, -1).trim();
    if (inner === '') return [];
    return inner.split(',').map((item) => parseScalar(item));
  }
  if (/^".*"$/.test(value) || /^'.*'$/.test(value)) return value.slice(1, -1);
  if (/^-?\d+$/.test(value)) return Number(value);
  if (value === 'true') return true;
  if (value === 'false') return false;
  return value;
}

/** Splits `---` delimited YAML frontmatter from the Markdown body. */
export function parseFrontmatter(raw: string): {
  frontmatter: Record<string, unknown>;
  body: string;
} {
  const match = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/.exec(raw);
  if (!match) throw new Error('Missing YAML frontmatter');

  const frontmatter: Record<string, unknown> = {};
  for (const line of match[1].split(/\r?\n/)) {
    if (!line.trim() || line.trimStart().startsWith('#')) continue;
    const separator = line.indexOf(':');
    if (separator === -1) continue;
    frontmatter[line.slice(0, separator).trim()] = parseScalar(line.slice(separator + 1));
  }

  return { frontmatter, body: match[2] };
}

export function readRoadmap(root = process.cwd()): Roadmap {
  const raw = readFileSync(join(root, 'content/roadmap.json'), 'utf8');
  return roadmapSchema.parse(JSON.parse(raw));
}

export function readLessonFiles(root = process.cwd()): LessonFile[] {
  const dir = join(root, 'content/lessons');
  return readdirSync(dir)
    .filter((name) => name.endsWith('.md') && !name.startsWith('_'))
    .sort()
    .map((name) => {
      const raw = readFileSync(join(dir, name), 'utf8');
      const { frontmatter, body } = parseFrontmatter(raw);
      return { file: `content/lessons/${name}`, slug: stripOrderPrefix(name), frontmatter, body };
    });
}
```

- [ ] **Step 4: Implement `scripts/validate-content.ts`**

```ts
import { lessonSchema } from '../src/content.config';
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

function main(): void {
  const errors = validateContent(readRoadmap(), readLessonFiles());
  if (errors.length > 0) {
    console.error(`\nContent validation failed with ${errors.length} error(s):\n`);
    for (const error of errors) console.error(`  ✗ ${error}`);
    console.error('');
    process.exit(1);
  }
  console.log('✓ Content validation passed.');
}

if (import.meta.url === `file://${process.argv[1]}`) main();
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `npx vitest run tests/scripts/validate-content.test.ts`
Expected: PASS — 13 tests, including the real-content check against the two Stage 0 lessons.

- [ ] **Step 6: Verify the CLI fails loudly on bad content**

Run: `npm run validate`
Expected: `✓ Content validation passed.`

Then temporarily add `"prerequisites": ["ghost"]` to the `what-is-ai` node in `content/roadmap.json` and run again.
Expected: exit code 1 and `✗ Node "what-is-ai" lists unknown prerequisite "ghost".`

Revert the change and confirm it passes again.

- [ ] **Step 7: Commit**

```bash
git add scripts tests/scripts
git commit -m "feat: add content integrity validation with failing build on errors"
```

---

### Task 5: Generate `CURRICULUM.md` and write `README.md`

**Files:**
- Create: `scripts/generate-curriculum.ts`
- Create: `README.md`
- Generated: `CURRICULUM.md`
- Test: `tests/scripts/generate-curriculum.test.ts`

**Interfaces:**
- Consumes: `readRoadmap`, `readLessonFiles`, `LessonFile` from `scripts/lib/content-io.ts`; `topologicalOrder` from `src/lib/roadmap.ts`.
- Produces: `renderCurriculum(roadmap: Roadmap, lessons: LessonFile[]): string` — the full Markdown text of `CURRICULUM.md`.

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/generate-curriculum.test.ts`:

```ts
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run tests/scripts/generate-curriculum.test.ts`
Expected: FAIL — cannot resolve `../../scripts/generate-curriculum`.

- [ ] **Step 3: Implement `scripts/generate-curriculum.ts`**

```ts
import { writeFileSync } from 'node:fs';
import { lessonSchema } from '../src/content.config';
import { topologicalOrder, type Roadmap } from '../src/lib/roadmap';
import { readRoadmap, readLessonFiles, type LessonFile } from './lib/content-io';

export function renderCurriculum(roadmap: Roadmap, lessons: LessonFile[]): string {
  const bySlug = new Map(lessons.map((l) => [l.slug, l]));
  const ordered = topologicalOrder(roadmap.nodes);

  const totalMinutes = lessons.reduce((sum, l) => {
    const parsed = lessonSchema.safeParse(l.frontmatter);
    return sum + (parsed.success ? parsed.data.minutes : 0);
  }, 0);

  const lines: string[] = [
    '<!-- Generated by scripts/generate-curriculum.ts — DO NOT EDIT BY HAND. -->',
    '<!-- Run `npm run gen:curriculum` after changing content/roadmap.json. -->',
    '',
    '# Curriculum',
    '',
    `**${lessons.length} lessons** · roughly ${Math.round(totalMinutes / 60)} hours of reading ·`,
    'free and open source.',
    '',
    'Work top to bottom. Each lesson lists what it assumes you already know, so if',
    'you already have some of this, skip ahead — the prerequisites tell you whether',
    'you are ready.',
    '',
  ];

  for (const stage of roadmap.stages) {
    const stageNodes = ordered.filter((n) => n.stage === stage.id);
    if (stageNodes.length === 0) continue;

    lines.push(`## Stage ${stage.id} · ${stage.name}`, '');
    lines.push('| # | Lesson | Time | Level | What it covers |');
    lines.push('|---|---|---|---|---|');

    for (const node of stageNodes) {
      const lesson = bySlug.get(node.id);
      if (!lesson) continue;
      const parsed = lessonSchema.safeParse(lesson.frontmatter);
      if (!parsed.success) continue;
      const { title, minutes, difficulty, summary, order } = parsed.data;
      lines.push(
        `| ${String(order).padStart(2, '0')} | [${title}](${lesson.file}) | ${minutes} min | ${difficulty} | ${summary} |`,
      );
    }
    lines.push('');
  }

  lines.push(
    '---',
    '',
    'Prefer an interactive version with progress tracking and search?',
    'Visit <https://sumitsingh4411.github.io/ai-roadmap>.',
    '',
  );

  return lines.join('\n');
}

function main(): void {
  writeFileSync('CURRICULUM.md', renderCurriculum(readRoadmap(), readLessonFiles()), 'utf8');
  console.log('✓ Wrote CURRICULUM.md');
}

if (import.meta.url === `file://${process.argv[1]}`) main();
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run tests/scripts/generate-curriculum.test.ts`
Expected: PASS — 6 tests.

- [ ] **Step 5: Generate the file**

Run: `npm run gen:curriculum`
Expected: `✓ Wrote CURRICULUM.md`, and the file contains a Stage 0 table with both lessons.

- [ ] **Step 6: Write `README.md`**

Must contain, in this order: project title and one-line pitch; a "Learn straight from GitHub — no website needed" section linking to `CURRICULUM.md` and explaining that every lesson is a plain Markdown file; a link to the live site; the seven-stage overview as a bullet list with lesson counts; a "How this repo is organised" section describing `content/roadmap.json`, `content/lessons/`, and `src/`; local development commands (`npm install`, `npm run dev`, `npm test`, `npm run build`); a contributing section stating the portable-Markdown rule and that `npm run validate` must pass; and a licence line (MIT, content CC BY 4.0).

- [ ] **Step 7: Commit**

```bash
git add scripts/generate-curriculum.ts tests/scripts/generate-curriculum.test.ts CURRICULUM.md README.md
git commit -m "feat: generate CURRICULUM.md from the roadmap graph"
```

---

### Task 6: Theme system and base layout

**Files:**
- Create: `src/styles/theme.css`
- Create: `src/styles/global.css`
- Create: `src/layouts/BaseLayout.astro`
- Create: `src/components/ThemeToggle.astro`
- Modify: `src/pages/index.astro`

**Interfaces:**
- Consumes: `href()` from `src/lib/url.ts`.
- Produces: `BaseLayout` accepting props `{ title: string; description: string; wide?: boolean }` and a default slot. Tasks 7, 9 and 12 wrap their pages in it. Theme is applied by a `data-theme="dark" | "light"` attribute on `<html>`.

- [ ] **Step 1: Create `src/styles/theme.css`**

```css
:root {
  --violet-400: #a78bfa;
  --violet-500: #8b5cf6;
  --violet-600: #7c3aed;
  --cyan-300: #67e8f9;
  --cyan-400: #22d3ee;

  --font-sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --font-mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;

  --radius: 14px;
  --max-width: 72ch;
  --step: 0.25rem;
}

:root,
:root[data-theme="dark"] {
  --bg: #06060f;
  --bg-raised: #0e0e1c;
  --surface: rgba(255, 255, 255, 0.055);
  --surface-strong: rgba(255, 255, 255, 0.1);
  --border: rgba(255, 255, 255, 0.12);
  --border-bright: rgba(255, 255, 255, 0.28);
  --text: #f4f4f7;
  --text-muted: #a09bbb;
  --accent: var(--violet-400);
  --accent-2: var(--cyan-400);
  --glow-accent: rgba(167, 139, 250, 0.45);
  --glow-accent-2: rgba(34, 211, 238, 0.35);
  --tile-face: linear-gradient(135deg, #2a2350, #171233);
  --tile-side: #120e28;
  --shadow-deep: 0 26px 44px rgba(0, 0, 0, 0.7);
  color-scheme: dark;
}

:root[data-theme="light"] {
  --bg: #f6f4fb;
  --bg-raised: #ffffff;
  --surface: rgba(20, 12, 46, 0.04);
  --surface-strong: rgba(20, 12, 46, 0.07);
  --border: rgba(20, 12, 46, 0.14);
  --border-bright: rgba(20, 12, 46, 0.3);
  --text: #14102b;
  --text-muted: #5b5478;
  --accent: var(--violet-600);
  --accent-2: #0891b2;
  --glow-accent: rgba(124, 58, 237, 0.24);
  --glow-accent-2: rgba(8, 145, 178, 0.2);
  --tile-face: linear-gradient(135deg, #ffffff, #ece8f8);
  --tile-side: #c9c1e4;
  --shadow-deep: 0 22px 38px rgba(30, 20, 70, 0.18);
  color-scheme: light;
}
```

- [ ] **Step 2: Create `src/styles/global.css`**

Contains: a modern reset (`box-sizing: border-box`, zeroed margins, `img { max-width: 100% }`); `body` using `--bg`, `--text`, `--font-sans`, `-webkit-font-smoothing: antialiased`; a `.prose` block styling `h2`–`h4`, `p`, `ul`, `ol`, `blockquote`, `table`, `a` (accent colour, underline on hover), and `pre` (rounded, `--bg-raised`, horizontal scroll, `--font-mono`); `:focus-visible` outlines using `--accent`; a `.skip-link` that is visually hidden until focused; and this block, which every later task relies on:

```css
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.001ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.001ms !important;
    scroll-behavior: auto !important;
  }
}
```

- [ ] **Step 3: Create `src/components/ThemeToggle.astro`**

```astro
---
// Toggles data-theme on <html> and persists the choice.
---
<button id="theme-toggle" type="button" aria-label="Toggle colour theme" aria-pressed="false">
  <span aria-hidden="true" data-icon-dark>◐</span>
  <span aria-hidden="true" data-icon-light>◑</span>
</button>

<script>
  const KEY = 'ai-roadmap:theme';
  const button = document.getElementById('theme-toggle');

  function apply(theme: string) {
    document.documentElement.dataset.theme = theme;
    button?.setAttribute('aria-pressed', String(theme === 'light'));
    try {
      localStorage.setItem(KEY, theme);
    } catch {
      /* private browsing — the choice simply does not persist */
    }
  }

  button?.addEventListener('click', () => {
    apply(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light');
  });
</script>

<style>
  #theme-toggle {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 999px;
    color: var(--text-muted);
    cursor: pointer;
    font-size: 0.95rem;
    padding: 0.4rem 0.7rem;
    transition: color 0.2s, border-color 0.2s;
  }
  #theme-toggle:hover {
    color: var(--text);
    border-color: var(--border-bright);
  }
  :root[data-theme='light'] [data-icon-dark],
  :root:not([data-theme='light']) [data-icon-light] {
    display: none;
  }
</style>
```

- [ ] **Step 4: Create `src/layouts/BaseLayout.astro`**

```astro
---
import '../styles/theme.css';
import '../styles/global.css';
import ThemeToggle from '../components/ThemeToggle.astro';
import { href } from '../lib/url';

interface Props {
  title: string;
  description: string;
  wide?: boolean;
}

const { title, description, wide = false } = Astro.props;
const fullTitle = title === 'AI Roadmap' ? title : `${title} · AI Roadmap`;
---
<!doctype html>
<html lang="en" data-theme="dark">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{fullTitle}</title>
    <meta name="description" content={description} />
    <meta property="og:title" content={fullTitle} />
    <meta property="og:description" content={description} />
    <meta property="og:type" content="website" />
    <link rel="canonical" href={new URL(Astro.url.pathname, Astro.site)} />
    <!-- Applied before first paint so the page never flashes the wrong theme. -->
    <script is:inline>
      try {
        const stored = localStorage.getItem('ai-roadmap:theme');
        if (stored === 'light' || stored === 'dark') {
          document.documentElement.dataset.theme = stored;
        } else if (window.matchMedia('(prefers-color-scheme: light)').matches) {
          document.documentElement.dataset.theme = 'light';
        }
      } catch {}
    </script>
  </head>
  <body>
    <a class="skip-link" href="#main">Skip to content</a>

    <header class="site-header">
      <a class="brand" href={href('/')}>
        <span class="brand-mark" aria-hidden="true"></span>
        AI Roadmap
      </a>
      <nav>
        <a href={href('/')}>Roadmap</a>
        <a href="https://github.com/sumitsingh4411/ai-roadmap" rel="noopener">GitHub</a>
        <ThemeToggle />
      </nav>
    </header>

    <main id="main" class={wide ? 'wide' : ''}>
      <slot />
    </main>

    <footer class="site-footer">
      <p>
        Free and open source. Code MIT, content CC BY 4.0 ·
        <a href="https://github.com/sumitsingh4411/ai-roadmap" rel="noopener">Contribute on GitHub</a>
      </p>
    </footer>
  </body>
</html>

<style>
  .site-header {
    align-items: center;
    border-bottom: 1px solid var(--border);
    display: flex;
    gap: 1rem;
    justify-content: space-between;
    padding: 1rem clamp(1rem, 4vw, 2.5rem);
  }
  .brand {
    align-items: center;
    color: var(--text);
    display: flex;
    font-weight: 700;
    gap: 0.55rem;
    letter-spacing: -0.01em;
    text-decoration: none;
  }
  .brand-mark {
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    border-radius: 6px;
    box-shadow: 0 0 18px var(--glow-accent);
    height: 18px;
    width: 18px;
  }
  .site-header nav {
    align-items: center;
    display: flex;
    gap: 1.1rem;
  }
  .site-header nav a {
    color: var(--text-muted);
    font-size: 0.9rem;
    text-decoration: none;
  }
  .site-header nav a:hover { color: var(--text); }
  main { margin: 0 auto; max-width: var(--max-width); padding: clamp(1.5rem, 5vw, 3rem) 1.25rem; }
  main.wide { max-width: 1200px; }
  .site-footer {
    border-top: 1px solid var(--border);
    color: var(--text-muted);
    font-size: 0.85rem;
    padding: 2rem 1.25rem;
    text-align: center;
  }
</style>
```

- [ ] **Step 5: Rewrite `src/pages/index.astro` to use the layout**

```astro
---
import BaseLayout from '../layouts/BaseLayout.astro';
---
<BaseLayout title="AI Roadmap" description="A free, open-source roadmap for learning AI from zero.">
  <h1>AI Roadmap</h1>
  <p>The interactive roadmap lands here in a later task.</p>
</BaseLayout>
```

- [ ] **Step 6: Verify the build and the theme**

Run: `npx astro build`
Expected: `Complete!` with no errors.

Run: `npx astro dev` and open the local URL.
Expected: dark theme by default; clicking the toggle switches to light and survives a reload; no flash of the wrong theme on refresh.

- [ ] **Step 7: Commit**

```bash
git add src/styles src/layouts src/components/ThemeToggle.astro src/pages/index.astro
git commit -m "feat: add theme system and base layout"
```

---

### Task 7: Lesson pages with depth-stack header, prev/next, and copy buttons

**Files:**
- Create: `src/components/LessonCard.astro`
- Create: `src/components/CodeCopy.astro`
- Create: `src/pages/lessons/[slug].astro`
- Create: `src/pages/404.astro`

**Interfaces:**
- Consumes: `BaseLayout`, `href()`, `topologicalOrder`, `readRoadmap` equivalent via `astro:content`.
- Produces: routes at `/lessons/<slug>` for all 34 slugs. `LessonCard` takes props `{ title: string; stage: RoadmapStage; minutes: number; difficulty: string; tags: string[]; summary: string; githubUrl: string }`.

- [ ] **Step 1: Create `src/components/LessonCard.astro`**

The depth-stack treatment: two dimmed offset layers behind a glass card with a masked gradient rim-light, tilting toward the cursor.

```astro
---
interface Props {
  title: string;
  stageName: string;
  minutes: number;
  difficulty: string;
  tags: string[];
  summary: string;
  githubUrl: string;
}
const { title, stageName, minutes, difficulty, tags, summary, githubUrl } = Astro.props;
---
<div class="deck" data-tilt>
  <div class="layer layer-back" aria-hidden="true"></div>
  <div class="layer layer-mid" aria-hidden="true"></div>
  <article class="layer card">
    <div class="rim" aria-hidden="true"></div>
    <p class="eyebrow">{stageName}</p>
    <h1>{title}</h1>
    <p class="summary">{summary}</p>
    <ul class="meta">
      <li>{minutes} min</li>
      <li>{difficulty}</li>
      {tags.map((tag) => <li class="tag">{tag}</li>)}
    </ul>
    <a class="github" href={githubUrl} rel="noopener">View this lesson on GitHub →</a>
  </article>
</div>

<script>
  // Cursor tilt. Skipped entirely when the user prefers reduced motion.
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  if (!reduced.matches) {
    for (const deck of document.querySelectorAll<HTMLElement>('[data-tilt]')) {
      deck.addEventListener('pointermove', (event) => {
        const box = deck.getBoundingClientRect();
        const x = (event.clientX - box.left) / box.width - 0.5;
        const y = (event.clientY - box.top) / box.height - 0.5;
        deck.style.setProperty('--tilt-x', `${(-y * 6).toFixed(2)}deg`);
        deck.style.setProperty('--tilt-y', `${(x * 8).toFixed(2)}deg`);
      });
      deck.addEventListener('pointerleave', () => {
        deck.style.setProperty('--tilt-x', '0deg');
        deck.style.setProperty('--tilt-y', '0deg');
      });
    }
  }
</script>

<style>
  .deck {
    --tilt-x: 0deg;
    --tilt-y: 0deg;
    margin-bottom: 3rem;
    perspective: 1000px;
    position: relative;
  }
  .layer {
    border: 1px solid var(--border);
    border-radius: 20px;
    transform-style: preserve-3d;
  }
  .layer-back,
  .layer-mid {
    background: var(--surface);
    inset: 0;
    position: absolute;
  }
  .layer-back { opacity: 0.35; transform: translate3d(22px, 18px, -90px); }
  .layer-mid { opacity: 0.6; transform: translate3d(11px, 9px, -45px); }
  .card {
    backdrop-filter: blur(14px);
    background: linear-gradient(150deg, var(--surface-strong), var(--surface));
    border-color: var(--border-bright);
    box-shadow: var(--shadow-deep);
    padding: 1.6rem 1.7rem;
    position: relative;
    transform: rotateX(var(--tilt-x)) rotateY(var(--tilt-y));
    transition: transform 0.25s ease-out;
  }
  /* Rim light: a 1px gradient border drawn with a mask. */
  .rim {
    background: linear-gradient(
      135deg,
      var(--border-bright),
      transparent 40%,
      transparent 60%,
      var(--glow-accent)
    );
    border-radius: 20px;
    inset: -1px;
    mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
    mask-composite: exclude;
    padding: 1px;
    pointer-events: none;
    position: absolute;
  }
  .eyebrow {
    color: var(--accent);
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    margin: 0 0 0.5rem;
    text-transform: uppercase;
  }
  .card h1 { font-size: clamp(1.6rem, 4vw, 2.2rem); letter-spacing: -0.025em; margin: 0; }
  .summary { color: var(--text-muted); margin: 0.6rem 0 1rem; }
  .meta { display: flex; flex-wrap: wrap; gap: 0.4rem; list-style: none; margin: 0 0 1.1rem; padding: 0; }
  .meta li {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 999px;
    color: var(--text-muted);
    font-size: 0.72rem;
    padding: 0.2rem 0.6rem;
  }
  .github { color: var(--accent-2); font-size: 0.85rem; text-decoration: none; }
  .github:hover { text-decoration: underline; }
</style>
```

- [ ] **Step 2: Create `src/components/CodeCopy.astro`**

```astro
---
// Adds a copy button to every Shiki-rendered <pre> in the prose body.
---
<script>
  for (const pre of document.querySelectorAll<HTMLElement>('.prose pre')) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'copy-button';
    button.textContent = 'Copy';
    button.setAttribute('aria-label', 'Copy code to clipboard');

    button.addEventListener('click', async () => {
      const code = pre.querySelector('code')?.textContent ?? '';
      try {
        await navigator.clipboard.writeText(code);
        button.textContent = 'Copied';
      } catch {
        button.textContent = 'Press ⌘C';
      }
      setTimeout(() => (button.textContent = 'Copy'), 1800);
    });

    pre.style.position = 'relative';
    pre.appendChild(button);
  }
</script>

<style is:global>
  .prose pre .copy-button {
    background: var(--surface-strong);
    border: 1px solid var(--border);
    border-radius: 7px;
    color: var(--text-muted);
    cursor: pointer;
    font-family: var(--font-sans);
    font-size: 0.7rem;
    opacity: 0;
    padding: 0.25rem 0.55rem;
    position: absolute;
    right: 0.6rem;
    top: 0.6rem;
    transition: opacity 0.15s, color 0.15s;
  }
  .prose pre:hover .copy-button,
  .prose pre .copy-button:focus-visible { opacity: 1; }
  .prose pre .copy-button:hover { color: var(--text); }
</style>
```

- [ ] **Step 3: Create `src/pages/lessons/[slug].astro`**

```astro
---
import { getCollection, render } from 'astro:content';
import roadmapData from '../../../content/roadmap.json';
import BaseLayout from '../../layouts/BaseLayout.astro';
import LessonCard from '../../components/LessonCard.astro';
import CodeCopy from '../../components/CodeCopy.astro';
import { roadmapSchema, topologicalOrder } from '../../lib/roadmap';
import { href } from '../../lib/url';

export async function getStaticPaths() {
  const lessons = await getCollection('lessons');
  return lessons.map((lesson) => ({
    params: { slug: lesson.id },
    props: { lesson },
  }));
}

const { lesson } = Astro.props;
const { Content, headings } = await render(lesson);

const roadmap = roadmapSchema.parse(roadmapData);
const ordered = topologicalOrder(roadmap.nodes);
const index = ordered.findIndex((n) => n.id === lesson.id);
const previous = index > 0 ? ordered[index - 1] : null;
const next = index >= 0 && index < ordered.length - 1 ? ordered[index + 1] : null;

const node = roadmap.nodes.find((n) => n.id === lesson.id);
const stage = roadmap.stages.find((s) => s.id === node?.stage);
const githubUrl = `https://github.com/sumitsingh4411/ai-roadmap/blob/main/content/lessons/${node?.lesson}.md`;

const titleOf = (id: string) =>
  id.split('-').map((w) => w[0].toUpperCase() + w.slice(1)).join(' ');
---
<BaseLayout title={lesson.data.title} description={lesson.data.summary}>
  <LessonCard
    title={lesson.data.title}
    stageName={`Stage ${stage?.id} · ${stage?.name}`}
    minutes={lesson.data.minutes}
    difficulty={lesson.data.difficulty}
    tags={lesson.data.tags}
    summary={lesson.data.summary}
    githubUrl={githubUrl}
  />

  {headings.length > 2 && (
    <nav class="toc" aria-label="On this page">
      <p class="toc-title">On this page</p>
      <ul>
        {headings.filter((h) => h.depth === 2).map((h) => (
          <li><a href={`#${h.slug}`}>{h.text}</a></li>
        ))}
      </ul>
    </nav>
  )}

  <div class="prose"><Content /></div>

  <button id="mark-complete" type="button" data-slug={lesson.id} aria-pressed="false">
    Mark as complete
  </button>

  <nav class="pager" aria-label="Lesson navigation">
    {previous
      ? <a class="prev" href={href(`/lessons/${previous.id}`)}>← {titleOf(previous.id)}</a>
      : <span></span>}
    {next
      ? <a class="next" href={href(`/lessons/${next.id}`)}>{titleOf(next.id)} →</a>
      : <span></span>}
  </nav>

  <CodeCopy />
</BaseLayout>

<style>
  .toc {
    border-left: 2px solid var(--border);
    margin-bottom: 2rem;
    padding-left: 1rem;
  }
  .toc-title {
    color: var(--text-muted);
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    margin: 0 0 0.4rem;
    text-transform: uppercase;
  }
  .toc ul { list-style: none; margin: 0; padding: 0; }
  .toc a { color: var(--text-muted); font-size: 0.88rem; text-decoration: none; }
  .toc a:hover { color: var(--accent); }
  #mark-complete {
    background: var(--surface);
    border: 1px solid var(--border-bright);
    border-radius: 999px;
    color: var(--text);
    cursor: pointer;
    font-size: 0.9rem;
    margin: 3rem 0 2rem;
    padding: 0.6rem 1.3rem;
    transition: background 0.2s, box-shadow 0.2s;
  }
  #mark-complete[aria-pressed='true'] {
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    box-shadow: 0 0 26px var(--glow-accent);
    color: #0b0a15;
    font-weight: 700;
  }
  .pager {
    border-top: 1px solid var(--border);
    display: flex;
    gap: 1rem;
    justify-content: space-between;
    padding-top: 1.5rem;
  }
  .pager a { color: var(--accent-2); font-size: 0.9rem; text-decoration: none; }
  .pager a:hover { text-decoration: underline; }
</style>
```

The `#mark-complete` button is wired to storage in Task 8; here it only renders.

**Deviation from the spec, recorded deliberately.** The spec's Pages section
lists "sidebar nav" for lesson pages, while its Visual Design section requires
"a single readable column" so the drama stays in the chrome. A persistent
34-item curriculum sidebar cannot satisfy both. This plan resolves it in favour
of the reading experience: navigation is served by the on-page table of
contents, prev/next, the ⌘K palette (Task 10), and the roadmap homepage. If a
full sidebar is wanted later, it is an additive change to this layout and
breaks nothing.

- [ ] **Step 4: Create `src/pages/404.astro`**

```astro
---
import BaseLayout from '../layouts/BaseLayout.astro';
import { href } from '../lib/url';
---
<BaseLayout title="Not found" description="That page does not exist.">
  <h1>404</h1>
  <p>That page does not exist — but the roadmap does.</p>
  <p><a href={href('/')}>← Back to the roadmap</a></p>
</BaseLayout>
```

- [ ] **Step 5: Build and verify the routes**

Run: `npx astro build`
Expected: `Complete!`, with `dist/lessons/what-is-ai/index.html` and `dist/lessons/how-to-learn-ai/index.html` present.

Run: `ls dist/lessons`
Expected: `how-to-learn-ai  what-is-ai`

- [ ] **Step 6: Verify in the browser**

Run: `npx astro dev`, open `/ai-roadmap/lessons/what-is-ai`.
Expected: the depth-stack header tilts toward the cursor; code blocks are syntax-highlighted and show a Copy button on hover that actually copies; the "Next" link goes to `how-to-learn-ai`; the GitHub link resolves to the file in the repo.

- [ ] **Step 7: Commit**

```bash
git add src/components/LessonCard.astro src/components/CodeCopy.astro src/pages/lessons src/pages/404.astro
git commit -m "feat: add lesson pages with depth-stack header and copy buttons"
```

---

### Task 8: Progress tracking

**Files:**
- Create: `src/lib/progress.ts`
- Create: `src/components/ProgressBar.astro`
- Modify: `src/pages/lessons/[slug].astro` (wire up `#mark-complete`)
- Test: `tests/lib/progress.test.ts`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `createProgressStore(storage?: StorageLike): ProgressStore` with methods `completed()`, `isComplete(slug)`, `toggle(slug)`, `markComplete(slug)`, `clear()`. `STORAGE_KEY` is exported. Task 9 reads the same store on the homepage. Storage is injected so tests need no DOM.

- [ ] **Step 1: Write the failing test**

Create `tests/lib/progress.test.ts`:

```ts
import { describe, it, expect, beforeEach } from 'vitest';
import { createProgressStore, STORAGE_KEY, type StorageLike } from '../../src/lib/progress';

function memoryStorage(seed: Record<string, string> = {}): StorageLike {
  const map = new Map(Object.entries(seed));
  return {
    getItem: (k) => map.get(k) ?? null,
    setItem: (k, v) => void map.set(k, v),
    removeItem: (k) => void map.delete(k),
  };
}

describe('createProgressStore', () => {
  let storage: StorageLike;

  beforeEach(() => {
    storage = memoryStorage();
  });

  it('starts empty', () => {
    expect(createProgressStore(storage).completed()).toEqual([]);
  });

  it('marks a lesson complete', () => {
    const store = createProgressStore(storage);
    store.markComplete('numpy');
    expect(store.isComplete('numpy')).toBe(true);
  });

  it('does not duplicate a lesson marked twice', () => {
    const store = createProgressStore(storage);
    store.markComplete('numpy');
    store.markComplete('numpy');
    expect(store.completed()).toEqual(['numpy']);
  });

  it('toggles a lesson off again', () => {
    const store = createProgressStore(storage);
    store.toggle('numpy');
    expect(store.isComplete('numpy')).toBe(true);
    store.toggle('numpy');
    expect(store.isComplete('numpy')).toBe(false);
  });

  it('persists across store instances', () => {
    createProgressStore(storage).markComplete('pandas');
    expect(createProgressStore(storage).completed()).toEqual(['pandas']);
  });

  it('clears everything', () => {
    const store = createProgressStore(storage);
    store.markComplete('a');
    store.clear();
    expect(store.completed()).toEqual([]);
  });

  it('recovers from corrupt stored JSON', () => {
    const store = createProgressStore(memoryStorage({ [STORAGE_KEY]: 'not json{{' }));
    expect(store.completed()).toEqual([]);
    store.markComplete('a');
    expect(store.completed()).toEqual(['a']);
  });

  it('discards stored data that is not an array of strings', () => {
    const store = createProgressStore(memoryStorage({ [STORAGE_KEY]: '{"a":1}' }));
    expect(store.completed()).toEqual([]);
  });

  it('filters non-string members out of a stored array', () => {
    const store = createProgressStore(memoryStorage({ [STORAGE_KEY]: '["a",3,null,"b"]' }));
    expect(store.completed()).toEqual(['a', 'b']);
  });

  it('survives a storage that throws on write', () => {
    const throwing: StorageLike = {
      getItem: () => null,
      setItem: () => {
        throw new Error('QuotaExceeded');
      },
      removeItem: () => {},
    };
    const store = createProgressStore(throwing);
    expect(() => store.markComplete('a')).not.toThrow();
    expect(store.isComplete('a')).toBe(true);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run tests/lib/progress.test.ts`
Expected: FAIL — cannot resolve `../../src/lib/progress`.

- [ ] **Step 3: Implement `src/lib/progress.ts`**

```ts
export const STORAGE_KEY = 'ai-roadmap:progress';

export interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export interface ProgressStore {
  completed(): string[];
  isComplete(slug: string): boolean;
  toggle(slug: string): string[];
  markComplete(slug: string): string[];
  clear(): void;
}

const noopStorage: StorageLike = {
  getItem: () => null,
  setItem: () => {},
  removeItem: () => {},
};

function resolveStorage(storage?: StorageLike): StorageLike {
  if (storage) return storage;
  try {
    return globalThis.localStorage ?? noopStorage;
  } catch {
    return noopStorage; // blocked by browser privacy settings
  }
}

/**
 * Completed-lesson store backed by localStorage.
 *
 * State is held in memory as well as written through, so the UI stays correct
 * even when the browser refuses to persist (private mode, quota exceeded).
 */
export function createProgressStore(storage?: StorageLike): ProgressStore {
  const backing = resolveStorage(storage);

  function load(): string[] {
    try {
      const raw = backing.getItem(STORAGE_KEY);
      if (!raw) return [];
      const parsed: unknown = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed.filter((item): item is string => typeof item === 'string');
    } catch {
      return []; // corrupt payload — start clean rather than crash the page
    }
  }

  let state = load();

  function persist(): string[] {
    try {
      backing.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch {
      /* keep the in-memory value; this session still behaves correctly */
    }
    return [...state];
  }

  return {
    completed: () => [...state],
    isComplete: (slug) => state.includes(slug),
    markComplete(slug) {
      if (!state.includes(slug)) state = [...state, slug];
      return persist();
    },
    toggle(slug) {
      state = state.includes(slug) ? state.filter((s) => s !== slug) : [...state, slug];
      return persist();
    },
    clear() {
      state = [];
      try {
        backing.removeItem(STORAGE_KEY);
      } catch {
        /* ignore */
      }
    },
  };
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run tests/lib/progress.test.ts`
Expected: PASS — 10 tests.

- [ ] **Step 5: Wire `#mark-complete` in `src/pages/lessons/[slug].astro`**

Append this script block to the page, below the existing markup:

```astro
<script>
  import { createProgressStore } from '../../lib/progress';

  const button = document.getElementById('mark-complete');
  const slug = button?.dataset.slug;

  if (button && slug) {
    const store = createProgressStore();

    function paint() {
      const done = store.isComplete(slug!);
      button!.setAttribute('aria-pressed', String(done));
      button!.textContent = done ? '✓ Completed' : 'Mark as complete';
    }

    button.addEventListener('click', () => {
      store.toggle(slug);
      paint();
    });

    paint();
  }
</script>
```

- [ ] **Step 6: Create `src/components/ProgressBar.astro`**

Renders `<div class="progress" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">` with a gradient fill using `--accent` → `--accent-2` and a `<p class="progress-label">0 of 34 lessons complete</p>`. It accepts prop `{ total: number }` and exposes `data-progress-fill` and `data-progress-label` hooks that Task 9's homepage script updates. No script of its own.

- [ ] **Step 7: Verify in the browser**

Run: `npx astro dev`, open a lesson, click "Mark as complete".
Expected: the button turns into a glowing "✓ Completed", survives a page reload, and toggles back off on a second click.

Verify in DevTools: `localStorage.getItem('ai-roadmap:progress')` returns `["what-is-ai"]`.

- [ ] **Step 8: Commit**

```bash
git add src/lib/progress.ts src/components/ProgressBar.astro src/pages/lessons tests/lib/progress.test.ts
git commit -m "feat: add localStorage progress tracking"
```

---

### Task 9: The isometric 3D homepage

**Files:**
- Create: `src/components/RoadmapTile.astro`
- Create: `src/components/IsometricBoard.astro`
- Modify: `src/pages/index.astro`

**Interfaces:**
- Consumes: `roadmapSchema` from `src/lib/roadmap.ts` (build time only); `createProgressStore` from `src/lib/progress.ts`; `href()`; `ProgressBar`.
- Produces: the homepage. Tiles carry `data-node-id`, `data-prereqs` (space-separated), and `data-state`; the client script recomputes `data-state` from stored progress on load.

Note: the browser script deliberately recomputes unlock state from the
`data-prereqs` attributes rather than importing `unlockedIds`/`nodeState`. Those
functions need the full node array, which would mean shipping `roadmap.json` to
the client for data already present in the DOM. The two implementations are kept
honest by Task 3's tests plus the visual check in Step 5.

- [ ] **Step 1: Create `src/components/RoadmapTile.astro`**

```astro
---
import { href } from '../lib/url';

interface Props {
  id: string;
  title: string;
  x: number;
  y: number;
  prerequisites: string[];
}
const { id, title, x, y, prerequisites } = Astro.props;
const CELL = 118;
---
<a
  class="tile"
  href={href(`/lessons/${id}`)}
  data-node-id={id}
  data-prereqs={prerequisites.join(' ')}
  data-state="locked"
  style={`--tx:${x * CELL}px; --ty:${y * CELL}px;`}
>
  <span class="face">
    <span class="label">{title}</span>
    <span class="check" aria-hidden="true">✓</span>
  </span>
</a>

<style>
  .tile {
    height: 96px;
    left: 0;
    position: absolute;
    text-decoration: none;
    top: 0;
    transform: translate3d(var(--tx), var(--ty), 0);
    transform-style: preserve-3d;
    transition: transform 0.35s cubic-bezier(0.2, 0.8, 0.3, 1);
    width: 96px;
  }
  .face {
    align-items: center;
    background: var(--tile-face);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow:
      0 8px 0 var(--tile-side),
      0 16px 0 color-mix(in srgb, var(--tile-side) 80%, black),
      var(--shadow-deep),
      inset 0 1px 0 rgba(255, 255, 255, 0.14);
    display: flex;
    height: 100%;
    justify-content: center;
    position: relative;
    transition: box-shadow 0.35s, border-color 0.35s, filter 0.35s;
    width: 100%;
  }
  /* Counter-rotate so the text reads flat against the tilted board. */
  .label {
    color: var(--text);
    font-size: 11px;
    font-weight: 600;
    line-height: 1.2;
    padding: 0 8px;
    text-align: center;
    transform: rotateZ(45deg) rotateX(-58deg) translateZ(26px);
  }
  .check { display: none; }

  .tile[data-state='locked'] .face { filter: saturate(0.25) brightness(0.6); }
  .tile[data-state='locked'] .label { color: var(--text-muted); }

  .tile[data-state='available'] .face {
    border-color: var(--accent-2);
    box-shadow:
      0 8px 0 var(--tile-side),
      0 16px 0 color-mix(in srgb, var(--tile-side) 80%, black),
      0 26px 40px var(--glow-accent-2),
      inset 0 1px 0 rgba(255, 255, 255, 0.3);
  }

  .tile[data-state='complete'] { transform: translate3d(var(--tx), var(--ty), 22px); }
  .tile[data-state='complete'] .face {
    background: linear-gradient(135deg, var(--violet-500), var(--violet-600));
    border-color: var(--violet-400);
    box-shadow:
      0 8px 0 #3b1a80,
      0 16px 0 #2a1160,
      0 26px 44px var(--glow-accent),
      inset 0 1px 0 rgba(255, 255, 255, 0.35);
  }
  .tile[data-state='complete'] .check {
    color: #fff;
    display: block;
    font-size: 10px;
    position: absolute;
    right: 8px;
    top: 6px;
    transform: rotateZ(45deg) rotateX(-58deg) translateZ(26px);
  }

  .tile:hover { transform: translate3d(var(--tx), var(--ty), 34px); }
  .tile:focus-visible { outline: 2px solid var(--accent); outline-offset: 4px; }

  @media (prefers-reduced-motion: reduce) {
    .tile,
    .tile:hover,
    .tile[data-state='complete'] { transform: translate3d(var(--tx), var(--ty), 0); }
  }
</style>
```

- [ ] **Step 2: Create `src/components/IsometricBoard.astro`**

Responsibilities: the tilted stage, the grid floor, the connector lines, and the client script that recomputes state.

```astro
---
import RoadmapTile from './RoadmapTile.astro';
import { roadmapSchema, type Roadmap } from '../lib/roadmap';
import roadmapData from '../../content/roadmap.json';

interface Props {
  titles: Record<string, string>;
}
const { titles } = Astro.props;
const roadmap: Roadmap = roadmapSchema.parse(roadmapData);
const CELL = 118;

const maxX = Math.max(...roadmap.nodes.map((n) => n.grid.x));
const maxY = Math.max(...roadmap.nodes.map((n) => n.grid.y));
const boardW = (maxX + 1) * CELL;
const boardH = (maxY + 1) * CELL;

// Connector segments, computed server-side so they render without JavaScript.
const links = roadmap.nodes.flatMap((node) =>
  node.prerequisites.map((prereqId) => {
    const from = roadmap.nodes.find((n) => n.id === prereqId);
    if (!from) return null;
    const x1 = from.grid.x * CELL + 48;
    const y1 = from.grid.y * CELL + 48;
    const x2 = node.grid.x * CELL + 48;
    const y2 = node.grid.y * CELL + 48;
    const length = Math.hypot(x2 - x1, y2 - y1);
    const angle = (Math.atan2(y2 - y1, x2 - x1) * 180) / Math.PI;
    return { id: `${prereqId}->${node.id}`, x1, y1, length, angle, to: node.id };
  }),
).filter((l): l is NonNullable<typeof l> => l !== null);
---
<div class="stage" data-board>
  <div class="board" style={`--board-w:${boardW}px; --board-h:${boardH}px;`}>
    <div class="floor" aria-hidden="true"></div>

    {links.map((link) => (
      <span
        class="link"
        data-link-to={link.to}
        style={`--lx:${link.x1}px; --ly:${link.y1}px; --len:${link.length}px; --angle:${link.angle}deg;`}
      ></span>
    ))}

    {roadmap.nodes.map((node) => (
      <RoadmapTile
        id={node.id}
        title={titles[node.id] ?? node.id}
        x={node.grid.x}
        y={node.grid.y}
        prerequisites={node.prerequisites}
      />
    ))}
  </div>
</div>

<script>
  import { createProgressStore } from '../lib/progress';

  const store = createProgressStore();

  function paintBoard() {
    const done = new Set(store.completed());

    for (const tile of document.querySelectorAll<HTMLElement>('[data-node-id]')) {
      const id = tile.dataset.nodeId!;
      const prereqs = (tile.dataset.prereqs ?? '').split(' ').filter(Boolean);
      const unlocked = prereqs.every((p) => done.has(p));
      tile.dataset.state = done.has(id) ? 'complete' : unlocked ? 'available' : 'locked';
      tile.setAttribute('aria-current', done.has(id) ? 'true' : 'false');
    }

    for (const link of document.querySelectorAll<HTMLElement>('[data-link-to]')) {
      const target = link.dataset.linkTo!;
      link.dataset.active = String(done.has(target));
    }

    const total = document.querySelectorAll('[data-node-id]').length;
    const fill = document.querySelector<HTMLElement>('[data-progress-fill]');
    const label = document.querySelector<HTMLElement>('[data-progress-label]');
    const bar = document.querySelector<HTMLElement>('[role="progressbar"]');
    const percent = total === 0 ? 0 : Math.round((done.size / total) * 100);

    if (fill) fill.style.width = `${percent}%`;
    if (label) label.textContent = `${done.size} of ${total} lessons complete`;
    if (bar) bar.setAttribute('aria-valuenow', String(percent));
  }

  paintBoard();
  // Progress can change in another tab; keep the board in sync.
  window.addEventListener('storage', paintBoard);
  window.addEventListener('pageshow', paintBoard);
</script>

<style>
  .stage {
    overflow: auto;
    padding: 3rem 1rem 5rem;
    perspective: 1600px;
  }
  .board {
    height: var(--board-h);
    margin: 0 auto;
    position: relative;
    transform: rotateX(58deg) rotateZ(-45deg);
    transform-style: preserve-3d;
    width: var(--board-w);
  }
  .floor {
    background-image:
      linear-gradient(var(--border) 1px, transparent 1px),
      linear-gradient(90deg, var(--border) 1px, transparent 1px);
    background-size: 59px 59px;
    inset: -80px;
    mask-image: radial-gradient(circle at 50% 50%, #000 40%, transparent 78%);
    position: absolute;
    transform: translateZ(-30px);
  }
  .link {
    background: var(--border-bright);
    border-radius: 3px;
    height: 4px;
    left: var(--lx);
    position: absolute;
    top: var(--ly);
    transform: rotate(var(--angle));
    transform-origin: 0 50%;
    transition: background 0.35s, box-shadow 0.35s;
    width: var(--len);
  }
  .link[data-active='true'] {
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    box-shadow: 0 0 14px var(--glow-accent);
  }

  @media (max-width: 720px) {
    /* The isometric projection is unusable on a narrow screen. */
    .board { transform: none; }
    .link { display: none; }
  }
  @media (prefers-reduced-motion: reduce) {
    .board { transform: none; }
  }
</style>
```

Note: under `transform: none` (mobile and reduced-motion), tile labels must also drop their counter-rotation. Add to `RoadmapTile.astro`:

```css
@media (max-width: 720px), (prefers-reduced-motion: reduce) {
  .label,
  .check { transform: none; }
}
```

- [ ] **Step 3: Rewrite `src/pages/index.astro`**

```astro
---
import { getCollection } from 'astro:content';
import BaseLayout from '../layouts/BaseLayout.astro';
import IsometricBoard from '../components/IsometricBoard.astro';
import ProgressBar from '../components/ProgressBar.astro';

const lessons = await getCollection('lessons');
const titles = Object.fromEntries(lessons.map((l) => [l.id, l.data.title]));
---
<BaseLayout
  title="AI Roadmap"
  description="A free, open-source roadmap for learning AI from zero — 34 lessons from Python to LLMs."
  wide
>
  <section class="hero">
    <h1>Climb the <span class="grad">AI ladder</span></h1>
    <p class="lede">
      34 lessons from your first line of Python to shipping an LLM application.
      Free, open source, and readable straight from GitHub.
    </p>
    <ProgressBar total={lessons.length} />
  </section>

  <IsometricBoard titles={titles} />
</BaseLayout>

<style>
  .hero { margin: 0 auto; max-width: 62ch; padding: 2rem 0 0; text-align: center; }
  .hero h1 {
    font-size: clamp(2.2rem, 7vw, 3.6rem);
    letter-spacing: -0.03em;
    line-height: 1.05;
    margin: 0;
  }
  .grad {
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
  }
  .lede { color: var(--text-muted); margin: 1rem auto 2rem; max-width: 46ch; }
</style>
```

- [ ] **Step 4: Add scroll-driven camera drift**

The spec calls for the board to drift gently as the page scrolls. Add to the
`IsometricBoard.astro` script:

```ts
// Camera drift: the board's tilt eases as the reader scrolls down.
const board = document.querySelector<HTMLElement>('.board');
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

if (board && !reducedMotion.matches && window.innerWidth > 720) {
  let ticking = false;
  const applyDrift = () => {
    const progress = Math.min(window.scrollY / 900, 1);
    // 58deg -> 46deg, -45deg -> -38deg: the board flattens slightly as you read.
    board.style.transform =
      `rotateX(${58 - progress * 12}deg) rotateZ(${-45 + progress * 7}deg)`;
    ticking = false;
  };
  window.addEventListener('scroll', () => {
    if (!ticking) {
      ticking = true;
      requestAnimationFrame(applyDrift);
    }
  }, { passive: true });
  applyDrift();
}
```

The inline `transform` must not be applied when the media query matches or the
viewport is narrow, otherwise it would override the flat-layout CSS rules.

- [ ] **Step 5: Build and verify**

Run: `npx astro build`
Expected: `Complete!` with no errors.

- [ ] **Step 6: Verify the board in the browser**

Run: `npx astro dev`, open the homepage.
Expected: a tilted 3D board of 34 tiles on a grid floor; only `what-is-ai` is cyan (available), the rest dim; completing a lesson turns its tile violet, raises it, lights its outgoing connector, and advances the progress bar. Resizing below 720px flattens the board to a readable 2D list.

Verify reduced motion: enable "Reduce motion" in OS settings, reload.
Expected: the board renders flat with no tilt, hover lift, transitions, or scroll drift, and every tile label is legible.

- [ ] **Step 7: Verify it works without JavaScript**

In DevTools, disable JavaScript and reload the homepage.
Expected: the board still renders with all 34 tiles as working links; every tile shows the `locked` styling, which is acceptable since no progress can be read.

- [ ] **Step 8: Commit**

```bash
git add src/components/RoadmapTile.astro src/components/IsometricBoard.astro src/pages/index.astro
git commit -m "feat: add isometric 3D roadmap homepage"
```

---

### Task 10: Search index and ⌘K dialog

**Files:**
- Create: `scripts/build-search-index.ts`
- Create: `src/lib/search.ts`
- Create: `src/components/SearchDialog.astro`
- Modify: `src/layouts/BaseLayout.astro` (mount the dialog and its trigger)
- Test: `tests/scripts/build-search-index.test.ts`, `tests/lib/search.test.ts`

**Interfaces:**
- Consumes: `readRoadmap`, `readLessonFiles` from `scripts/lib/content-io.ts`; `lessonSchema`.
- Produces:
  - `SearchDoc = { slug: string; title: string; stage: number; summary: string; headings: string[]; text: string }`.
  - `stripMarkdown(md: string): string` and `buildSearchIndex(lessons: LessonFile[]): SearchDoc[]` from `scripts/build-search-index.ts`.
  - `searchDocs(docs: SearchDoc[], query: string, limit?: number): SearchDoc[]` from `src/lib/search.ts`.

- [ ] **Step 1: Write the failing index-builder test**

Create `tests/scripts/build-search-index.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import { stripMarkdown, buildSearchIndex } from '../../scripts/build-search-index';
import { readLessonFiles } from '../../scripts/lib/content-io';

describe('stripMarkdown', () => {
  it('removes fenced code blocks', () => {
    expect(stripMarkdown('before\n```py\nx = 1\n```\nafter')).toBe('before after');
  });

  it('removes heading markers but keeps the text', () => {
    expect(stripMarkdown('## Why this matters')).toBe('Why this matters');
  });

  it('keeps link text and drops the URL', () => {
    expect(stripMarkdown('see [the docs](https://example.com) now')).toBe('see the docs now');
  });

  it('removes emphasis markers', () => {
    expect(stripMarkdown('**bold** and _italic_')).toBe('bold and italic');
  });

  it('collapses whitespace', () => {
    expect(stripMarkdown('a\n\n\n   b')).toBe('a b');
  });
});

describe('buildSearchIndex', () => {
  const docs = buildSearchIndex(readLessonFiles());

  it('produces one document per lesson', () => {
    expect(docs.length).toBe(readLessonFiles().length);
  });

  it('captures the h2 headings of each lesson', () => {
    const doc = docs.find((d) => d.slug === 'what-is-ai')!;
    expect(doc.headings).toContain('Why this matters');
  });

  it('stores body text with no code fences', () => {
    for (const doc of docs) expect(doc.text).not.toContain('```');
  });

  it('carries the title and summary through', () => {
    const doc = docs.find((d) => d.slug === 'what-is-ai')!;
    expect(doc.title.length).toBeGreaterThan(0);
    expect(doc.summary.length).toBeGreaterThan(0);
  });
});
```

- [ ] **Step 2: Run it and verify it fails**

Run: `npx vitest run tests/scripts/build-search-index.test.ts`
Expected: FAIL — cannot resolve `../../scripts/build-search-index`.

- [ ] **Step 3: Implement `scripts/build-search-index.ts`**

```ts
import { writeFileSync, mkdirSync } from 'node:fs';
import { lessonSchema } from '../src/content.config';
import { readLessonFiles, type LessonFile } from './lib/content-io';

export interface SearchDoc {
  slug: string;
  title: string;
  stage: number;
  summary: string;
  headings: string[];
  text: string;
}

/** Reduces Markdown to plain prose suitable for substring matching. */
export function stripMarkdown(markdown: string): string {
  return markdown
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`([^`]*)`/g, '$1')
    .replace(/!\[[^\]]*\]\([^)]*\)/g, ' ')
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/^\s{0,3}#{1,6}\s+/gm, '')
    .replace(/^\s{0,3}>\s?/gm, '')
    .replace(/[*_]{1,3}([^*_]+)[*_]{1,3}/g, '$1')
    .replace(/^\s*[-*+]\s+/gm, '')
    .replace(/\|/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function extractHeadings(markdown: string): string[] {
  return [...markdown.matchAll(/^##\s+(.+)$/gm)].map((m) => m[1].trim());
}

export function buildSearchIndex(lessons: LessonFile[]): SearchDoc[] {
  return lessons.flatMap((lesson) => {
    const parsed = lessonSchema.safeParse(lesson.frontmatter);
    if (!parsed.success) return [];
    return [
      {
        slug: lesson.slug,
        title: parsed.data.title,
        stage: parsed.data.stage,
        summary: parsed.data.summary,
        headings: extractHeadings(lesson.body),
        text: stripMarkdown(lesson.body).slice(0, 4000),
      },
    ];
  });
}

function main(): void {
  const docs = buildSearchIndex(readLessonFiles());
  mkdirSync('public', { recursive: true });
  writeFileSync('public/search-index.json', JSON.stringify(docs), 'utf8');
  console.log(`✓ Wrote public/search-index.json (${docs.length} documents)`);
}

if (import.meta.url === `file://${process.argv[1]}`) main();
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run tests/scripts/build-search-index.test.ts`
Expected: PASS — 9 tests.

- [ ] **Step 5: Write the failing query test**

Create `tests/lib/search.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import { searchDocs } from '../../src/lib/search';
import type { SearchDoc } from '../../scripts/build-search-index';

const docs: SearchDoc[] = [
  { slug: 'numpy', title: 'NumPy', stage: 1, summary: 'Arrays and shapes.', headings: ['Why this matters'], text: 'vectorised arrays broadcasting' },
  { slug: 'pandas', title: 'Pandas', stage: 1, summary: 'DataFrames.', headings: ['In code'], text: 'tables columns groupby numpy interop' },
  { slug: 'rag', title: 'RAG', stage: 5, summary: 'Retrieval augmented generation.', headings: [], text: 'retrieval chunks embeddings' },
];

describe('searchDocs', () => {
  it('returns nothing for an empty query', () => {
    expect(searchDocs(docs, '')).toEqual([]);
  });

  it('matches on title', () => {
    expect(searchDocs(docs, 'numpy')[0].slug).toBe('numpy');
  });

  it('is case-insensitive', () => {
    expect(searchDocs(docs, 'NUMPY')[0].slug).toBe('numpy');
  });

  it('ranks a title match above a body match', () => {
    const results = searchDocs(docs, 'numpy');
    expect(results.map((d) => d.slug)).toEqual(['numpy', 'pandas']);
  });

  it('matches on summary', () => {
    expect(searchDocs(docs, 'dataframes')[0].slug).toBe('pandas');
  });

  it('matches on body text', () => {
    expect(searchDocs(docs, 'broadcasting')[0].slug).toBe('numpy');
  });

  it('matches on headings', () => {
    expect(searchDocs(docs, 'why this matters')[0].slug).toBe('numpy');
  });

  it('requires every term to match', () => {
    expect(searchDocs(docs, 'numpy zzz')).toEqual([]);
  });

  it('respects the result limit', () => {
    expect(searchDocs(docs, 'a', 1).length).toBeLessThanOrEqual(1);
  });
});
```

- [ ] **Step 6: Run it and verify it fails**

Run: `npx vitest run tests/lib/search.test.ts`
Expected: FAIL — cannot resolve `../../src/lib/search`.

- [ ] **Step 7: Implement `src/lib/search.ts`**

```ts
import type { SearchDoc } from '../../scripts/build-search-index';

const FIELD_WEIGHTS = { title: 10, summary: 5, headings: 3, text: 1 } as const;

function scoreField(haystack: string, term: string, weight: number): number {
  const index = haystack.indexOf(term);
  if (index === -1) return 0;
  // A match at a word boundary is a stronger signal than one mid-word.
  const boundary = index === 0 || /\W/.test(haystack[index - 1]);
  return weight * (boundary ? 2 : 1);
}

/**
 * Ranks documents by weighted substring matches.
 * Every term must match somewhere, so extra words narrow the result set.
 */
export function searchDocs(docs: SearchDoc[], query: string, limit = 12): SearchDoc[] {
  const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (terms.length === 0) return [];

  const ranked: Array<{ doc: SearchDoc; score: number }> = [];

  for (const doc of docs) {
    const fields = {
      title: doc.title.toLowerCase(),
      summary: doc.summary.toLowerCase(),
      headings: doc.headings.join(' ').toLowerCase(),
      text: doc.text.toLowerCase(),
    };

    let total = 0;
    let allTermsMatched = true;

    for (const term of terms) {
      let termScore = 0;
      for (const [field, weight] of Object.entries(FIELD_WEIGHTS)) {
        termScore += scoreField(fields[field as keyof typeof fields], term, weight);
      }
      if (termScore === 0) {
        allTermsMatched = false;
        break;
      }
      total += termScore;
    }

    if (allTermsMatched) ranked.push({ doc, score: total });
  }

  return ranked
    .sort((a, b) => b.score - a.score || a.doc.slug.localeCompare(b.doc.slug))
    .slice(0, limit)
    .map((r) => r.doc);
}
```

- [ ] **Step 8: Run the test to verify it passes**

Run: `npx vitest run tests/lib/search.test.ts`
Expected: PASS — 9 tests.

- [ ] **Step 9: Create `src/components/SearchDialog.astro`**

Requirements, implemented with a native `<dialog>`:

- A header trigger button labelled `Search` showing the `⌘K` hint.
- Opens on `⌘K`, `Ctrl+K`, or `/` — but **not** while focus is in an `input`, `textarea`, or `[contenteditable]`.
- Closes on `Escape` and on backdrop click.
- Fetches `search-index.json` **once**, lazily on first open, via `fetch(href('/search-index.json'))`; caches the parsed array in a module-scope variable.
- Renders results as a `role="listbox"` of links showing title, stage name, and summary.
- `ArrowUp`/`ArrowDown` move the active option (`aria-activedescendant`); `Enter` navigates to it.
- Shows "No lessons match *query*" for zero results, and "Search unavailable" if the fetch rejects — the rest of the site must keep working.
- Styling: centred panel using `--bg-raised`, `--border-bright`, `var(--shadow-deep)`, `border-radius: var(--radius)`, and a blurred `::backdrop`.

- [ ] **Step 10: Mount the dialog in `BaseLayout.astro`**

Import `SearchDialog` and place `<SearchDialog />` inside `<nav>`, before `<ThemeToggle />`.

- [ ] **Step 11: Generate the index and verify end to end**

Run: `npm run gen:search`
Expected: `✓ Wrote public/search-index.json (2 documents)`

Run: `npx astro dev`, press `⌘K`, type `neural`.
Expected: matching lessons appear ranked; arrow keys move the selection; Enter opens the lesson; Escape closes. Typing `/` inside the search input types a slash rather than reopening the dialog.

- [ ] **Step 12: Add `public/search-index.json` to `.gitignore`**

It is generated by `prebuild`; committing it would create merge conflicts on every content change.

```
public/search-index.json
```

- [ ] **Step 13: Commit**

```bash
git add scripts/build-search-index.ts src/lib/search.ts src/components/SearchDialog.astro src/layouts/BaseLayout.astro tests .gitignore
git commit -m "feat: add build-time search index and command palette"
```

---

### Tasks 11–15: Write the remaining 32 lessons

These five tasks are content authoring. **Every lesson follows the exact five-part structure of `content/lessons/00-what-is-ai.md` from Task 2** — read that file first and match its voice, depth, and formatting.

**Rules that apply to all five tasks:**

- Frontmatter `stage` and `prerequisites` **must exactly match** `content/roadmap.json` for that node — `npm run validate` enforces this and the build fails otherwise.
- `order` is the global sequence number, equal to the file's numeric prefix.
- Portable Markdown only. No HTML, no MDX, no components.
- Every code block is **runnable as written** and shows its expected output in a following plain fenced block.
- Every lesson ends with `**Next:** [Title](NN-slug.md)` using a **relative path**, so the link works on github.com.
- "Go deeper" lists 3–5 **free** resources with real, working URLs.
- After each task: run `npm run validate`, `npm run gen:curriculum`, `npm test`, and `npx astro build` — all must pass — then commit.

**Definition of done for a lesson:** a motivated beginner who has completed the prerequisites can read it end to end, run the code, and complete the exercise without opening another source.

---

### Task 11: Stage 1 — Python & Data (5 lessons)

**Files:**
- Create: `content/lessons/02-python-basics.md` (order 2, 60 min, beginner, prereq `how-to-learn-ai`)
- Create: `content/lessons/03-numpy.md` (order 3, 45 min, beginner, prereq `python-basics`)
- Create: `content/lessons/04-pandas.md` (order 4, 50 min, beginner, prereq `numpy`)
- Create: `content/lessons/05-data-visualization.md` (order 5, 40 min, beginner, prereq `pandas`)
- Create: `content/lessons/06-real-datasets.md` (order 6, 45 min, beginner, prereq `pandas`)

**Interfaces:**
- Consumes: node ids and prerequisites from `content/roadmap.json`; the lesson template from `00-what-is-ai.md`.
- Produces: five lesson files whose slugs are `python-basics`, `numpy`, `pandas`, `data-visualization`, `real-datasets`.

- [ ] **Step 1: Write `02-python-basics.md`** — variables and types, lists and dicts, `for`/`if`, functions, imports, virtual environments, and reading a traceback. Exercise: a script that reads a list of numbers and reports mean, min, and max without using libraries.

- [ ] **Step 2: Write `03-numpy.md`** — why arrays beat lists, `ndarray` creation, `shape` and `dtype`, indexing and slicing, broadcasting, vectorised maths and `axis`. Exercise: normalise a 2D array column-wise without a loop.

- [ ] **Step 3: Write `04-pandas.md`** — `Series` vs `DataFrame`, `read_csv`, `head`/`info`/`describe`, selection with `loc`/`iloc`, filtering, `groupby`, and handling missing values. Exercise: load a CSV and answer three questions about it with `groupby`.

- [ ] **Step 4: Write `05-data-visualization.md`** — when to use each chart type, `matplotlib` basics, plotting straight from pandas, labelling axes properly, and what a histogram tells you about a feature. Exercise: plot a distribution and a relationship from the previous lesson's dataset and write one sentence about each.

- [ ] **Step 5: Write `06-real-datasets.md`** — where to find datasets, loading messy CSVs, types and parsing dates, duplicates, outliers, and building a reusable cleaning checklist. Exercise: clean a deliberately messy dataset and document every decision.

- [ ] **Step 6: Validate, regenerate, test, and build**

```bash
npm run validate && npm run gen:curriculum && npm test && npx astro build
```

Expected: validation passes, `CURRICULUM.md` shows 7 lessons, all tests pass, build completes.

- [ ] **Step 7: Commit**

```bash
git add content/lessons CURRICULUM.md
git commit -m "content: add Stage 1 Python and data lessons"
```

---

### Task 12: Stage 2 — Math (3 lessons) and Stage 3 — Classical ML (8 lessons)

**Files:**
- Create: `content/lessons/07-linear-algebra.md` (order 7, 50 min, beginner, prereq `how-to-learn-ai`)
- Create: `content/lessons/08-calculus.md` (order 8, 50 min, intermediate, prereq `linear-algebra`)
- Create: `content/lessons/09-probability-stats.md` (order 9, 50 min, beginner, prereq `how-to-learn-ai`)
- Create: `content/lessons/10-ml-fundamentals.md` (order 10, 45 min, beginner, prereqs `numpy`, `probability-stats`)
- Create: `content/lessons/11-regression.md` (order 11, 50 min, beginner, prereqs `ml-fundamentals`, `linear-algebra`)
- Create: `content/lessons/12-classification.md` (order 12, 50 min, beginner, prereq `regression`)
- Create: `content/lessons/13-model-evaluation.md` (order 13, 45 min, intermediate, prereq `classification`)
- Create: `content/lessons/14-feature-engineering.md` (order 14, 45 min, intermediate, prereqs `pandas`, `model-evaluation`)
- Create: `content/lessons/15-clustering-pca.md` (order 15, 45 min, intermediate, prereqs `ml-fundamentals`, `linear-algebra`)
- Create: `content/lessons/16-trees-ensembles.md` (order 16, 50 min, intermediate, prereq `model-evaluation`)
- Create: `content/lessons/17-first-ml-project.md` (order 17, 90 min, intermediate, prereqs `feature-engineering`, `trees-ensembles`, `data-visualization`, `real-datasets`)

**Interfaces:**
- Consumes: Stage 1 lessons from Task 11.
- Produces: slugs `linear-algebra`, `calculus`, `probability-stats`, `ml-fundamentals`, `regression`, `classification`, `model-evaluation`, `feature-engineering`, `clustering-pca`, `trees-ensembles`, `first-ml-project`.

- [ ] **Step 1: Write the three maths lessons** — vectors, matrices, dot products and matrix multiply as "combining features with weights" (`07`); derivatives as slope, gradients, the chain rule, and gradient descent by hand in NumPy (`08`); distributions, mean/variance, conditional probability, Bayes, sampling, and why a p-value is not what people think (`09`). Every maths concept must be introduced through the ML problem it solves, never as abstract theory.

- [ ] **Step 2: Write `10-ml-fundamentals.md`** — supervised vs unsupervised, features and labels, the train/validation/test split, overfitting and underfitting, and the bias–variance tradeoff. Exercise: split a dataset and show the train/test score gap widening as a model's complexity grows.

- [ ] **Step 3: Write `11-regression.md`** — linear regression from first principles in NumPy, then the same in scikit-learn; the cost function; MSE, MAE and R²; regularisation with Ridge and Lasso. Exercise: predict house prices and interpret the coefficients.

- [ ] **Step 4: Write `12-classification.md`** — logistic regression, the sigmoid, decision boundaries, k-NN, and multi-class strategies. Exercise: classify the Iris dataset and plot the decision boundary.

- [ ] **Step 5: Write `13-model-evaluation.md`** — the confusion matrix, precision, recall, F1, ROC-AUC, cross-validation, and why accuracy misleads on imbalanced data. Exercise: build a deliberately imbalanced problem where a 97%-accurate model is worthless.

- [ ] **Step 6: Write `14-feature-engineering.md`** — scaling and normalisation, encoding categoricals, dates, binning, interaction terms, and leakage. Exercise: raise a baseline model's score using features alone, no model change.

- [ ] **Step 7: Write `15-clustering-pca.md`** — k-means, choosing k with the elbow method, hierarchical clustering, PCA as compression, and explained variance. Exercise: cluster customers and describe each segment.

- [ ] **Step 8: Write `16-trees-ensembles.md`** — decision trees and how splits are chosen, random forests, gradient boosting, XGBoost/LightGBM, and feature importance. Exercise: beat the previous lesson's best score with a boosted model and explain its top five features.

- [ ] **Step 9: Write `17-first-ml-project.md`** — the full pipeline end to end: problem framing, EDA, cleaning, features, baseline, iteration, evaluation, and writing up results. This lesson is a guided project, so it may exceed the usual length. Exercise: ship the project to GitHub with a README.

- [ ] **Step 10: Validate, regenerate, test, and build**

```bash
npm run validate && npm run gen:curriculum && npm test && npx astro build
```

Expected: all pass; `CURRICULUM.md` shows 18 lessons.

- [ ] **Step 11: Commit**

```bash
git add content/lessons CURRICULUM.md
git commit -m "content: add maths and classical ML lessons"
```

---

### Task 13: Stage 4 — Deep Learning (6 lessons)

**Files:**
- Create: `content/lessons/18-neural-networks.md` (order 18, 55 min, intermediate, prereqs `first-ml-project`, `calculus`)
- Create: `content/lessons/19-backprop-training.md` (order 19, 60 min, intermediate, prereq `neural-networks`)
- Create: `content/lessons/20-pytorch.md` (order 20, 60 min, intermediate, prereq `backprop-training`)
- Create: `content/lessons/21-cnns-vision.md` (order 21, 55 min, intermediate, prereq `pytorch`)
- Create: `content/lessons/22-sequence-models.md` (order 22, 50 min, advanced, prereq `pytorch`)
- Create: `content/lessons/23-transformers.md` (order 23, 70 min, advanced, prereq `sequence-models`)

**Interfaces:**
- Consumes: `pytorch` is the dependency root for `cnns-vision`, `sequence-models`, and (in Task 14) `fine-tuning`.
- Produces: slugs `neural-networks`, `backprop-training`, `pytorch`, `cnns-vision`, `sequence-models`, `transformers`.

- [ ] **Step 1: Write `18-neural-networks.md`** — what one neuron computes, why non-linear activations are necessary, layers and depth, and the universal approximation intuition. Exercise: implement a single neuron in NumPy and fit it to a straight line.

- [ ] **Step 2: Write `19-backprop-training.md`** — the loss function, gradients through a network, the chain rule applied layer by layer, learning rate, epochs and batches. Exercise: implement a two-layer network with manual backprop in NumPy that learns XOR.

- [ ] **Step 3: Write `20-pytorch.md`** — tensors, autograd, `nn.Module`, optimisers, the canonical training loop, and moving to GPU. Exercise: rewrite the previous lesson's XOR network in PyTorch and compare the line count.

- [ ] **Step 4: Write `21-cnns-vision.md`** — convolution as a learned filter, kernels, stride and padding, pooling, typical architectures, and transfer learning. Exercise: classify CIFAR-10 with a small CNN, then again with a pretrained backbone, and compare.

- [ ] **Step 5: Write `22-sequence-models.md`** — why order matters, RNNs, the vanishing-gradient problem, LSTM and GRU, and the limits that motivated attention. Exercise: character-level text generation on a small corpus.

- [ ] **Step 6: Write `23-transformers.md`** — attention as learned lookup, query/key/value, self-attention, multi-head attention, positional encoding, and the encoder/decoder split. Include a worked numerical example of a single attention head on a three-token sequence. Exercise: implement scaled dot-product attention in ~20 lines of PyTorch and verify the output shape.

- [ ] **Step 7: Validate, regenerate, test, and build**

```bash
npm run validate && npm run gen:curriculum && npm test && npx astro build
```

Expected: all pass; `CURRICULUM.md` shows 24 lessons.

- [ ] **Step 8: Commit**

```bash
git add content/lessons CURRICULUM.md
git commit -m "content: add deep learning lessons"
```

---

### Task 14: Stage 5 — Generative AI & LLMs (7 lessons)

**Files:**
- Create: `content/lessons/24-how-llms-work.md` (order 24, 55 min, intermediate, prereq `transformers`)
- Create: `content/lessons/25-prompt-engineering.md` (order 25, 45 min, beginner, prereq `how-llms-work`)
- Create: `content/lessons/26-embeddings.md` (order 26, 45 min, intermediate, prereq `how-llms-work`)
- Create: `content/lessons/27-rag.md` (order 27, 60 min, intermediate, prereqs `embeddings`, `prompt-engineering`)
- Create: `content/lessons/28-fine-tuning.md` (order 28, 60 min, advanced, prereqs `how-llms-work`, `pytorch`)
- Create: `content/lessons/29-ai-agents.md` (order 29, 60 min, advanced, prereq `rag`)
- Create: `content/lessons/30-evals-guardrails.md` (order 30, 50 min, advanced, prereq `ai-agents`)

**Interfaces:**
- Consumes: `transformers` and `pytorch` from Task 13.
- Produces: slugs `how-llms-work`, `prompt-engineering`, `embeddings`, `rag`, `fine-tuning`, `ai-agents`, `evals-guardrails`.

- [ ] **Step 1: Write `24-how-llms-work.md`** — tokenisation, next-token prediction, pretraining vs post-training, context windows, temperature and sampling, and an honest account of what causes hallucination. Exercise: tokenise a sentence and inspect how the model's vocabulary splits it.

- [ ] **Step 2: Write `25-prompt-engineering.md`** — clear instructions, few-shot examples, chain-of-thought, structured output, system prompts, and iterating on failures. Exercise: take a prompt that fails and fix it in three documented iterations.

- [ ] **Step 3: Write `26-embeddings.md`** — text as vectors, cosine similarity, embedding models, vector databases, and chunking strategy. Exercise: build a semantic search over ~50 documents using cosine similarity in NumPy alone.

- [ ] **Step 4: Write `27-rag.md`** — why retrieval beats stuffing the context window, the ingest → chunk → embed → retrieve → generate pipeline, chunk sizing, and common failure modes. Exercise: build a working RAG system over your own notes.

- [ ] **Step 5: Write `28-fine-tuning.md`** — when fine-tuning is the right answer and when RAG or prompting is, full fine-tuning vs LoRA/PEFT, dataset preparation, and evaluating the result. Exercise: fine-tune a small open model with LoRA on a toy dataset.

- [ ] **Step 6: Write `29-ai-agents.md`** — tool use, the reason–act loop, planning, memory, multi-step failures, and cost control. Exercise: build an agent with two tools that answers questions requiring both.

- [ ] **Step 7: Write `30-evals-guardrails.md`** — why manual spot-checking does not scale, building an eval set, LLM-as-judge and its biases, regression testing prompts, and input/output guardrails. Exercise: write an eval suite for the Task 27 RAG system and catch one real regression.

- [ ] **Step 8: Validate, regenerate, test, and build**

```bash
npm run validate && npm run gen:curriculum && npm test && npx astro build
```

Expected: all pass; `CURRICULUM.md` shows 31 lessons.

- [ ] **Step 9: Commit**

```bash
git add content/lessons CURRICULUM.md
git commit -m "content: add generative AI and LLM lessons"
```

---

### Task 15: Stage 6 — Ship It (3 lessons)

**Files:**
- Create: `content/lessons/31-mlops-basics.md` (order 31, 45 min, intermediate, prereq `first-ml-project`)
- Create: `content/lessons/32-deploying-models.md` (order 32, 55 min, intermediate, prereq `mlops-basics`)
- Create: `content/lessons/33-portfolio-career.md` (order 33, 40 min, beginner, prereqs `deploying-models`, `evals-guardrails`)

**Interfaces:**
- Consumes: `first-ml-project` (Task 12) and `evals-guardrails` (Task 14).
- Produces: slugs `mlops-basics`, `deploying-models`, `portfolio-career`. This completes all 34 lessons.

- [ ] **Step 1: Write `31-mlops-basics.md`** — experiment tracking, model and data versioning, reproducibility, and monitoring for drift. Exercise: add experiment tracking to the Task 12 project.

- [ ] **Step 2: Write `32-deploying-models.md`** — wrapping a model in a FastAPI endpoint, request/response schemas, Dockerising it, latency and batching, and free hosting options. Exercise: deploy a model and call it over HTTP from another machine.

- [ ] **Step 3: Write `33-portfolio-career.md`** — what makes a project worth showing, writing a project README that gets read, the main AI job families and what each actually requires day to day, how to keep learning after this roadmap, and where the community is. Exercise: publish a portfolio README for the three projects built in this roadmap.

- [ ] **Step 4: Verify all 34 lessons are present and consistent**

```bash
ls content/lessons/*.md | wc -l
```

Expected: `34`

```bash
npm run validate && npm run gen:curriculum && npm test && npx astro build
```

Expected: all pass; `CURRICULUM.md` reports `**34 lessons**`; the build emits 34 lesson pages.

```bash
ls dist/lessons | wc -l
```

Expected: `34`

- [ ] **Step 5: Verify the homepage with a full graph**

Run: `npx astro dev`, open the homepage.
Expected: all 34 tiles laid out with no overlaps, connectors drawn between dependent tiles, and only `what-is-ai` available at zero progress.

- [ ] **Step 6: Commit**

```bash
git add content/lessons CURRICULUM.md
git commit -m "content: add deployment and career lessons, completing all 34"
```

---

### Task 16: Accessibility and cross-browser pass

**Files:**
- Modify: `src/styles/global.css`, `src/components/RoadmapTile.astro`, `src/components/IsometricBoard.astro`, `src/components/SearchDialog.astro`

**Interfaces:**
- Consumes: everything built so far.
- Produces: no new API. This task only hardens what exists.

- [ ] **Step 1: Verify keyboard navigation**

Tab through the homepage.
Expected: the skip link appears first; every tile is reachable in reading order with a visible focus ring that is not clipped by the 3D transform; Enter opens a lesson.

Fix any tile whose focus ring is hidden by adding `outline-offset` and ensuring no ancestor sets `overflow: hidden` on the focused element's stacking context.

- [ ] **Step 2: Verify state is not conveyed by colour alone**

Expected: complete tiles show the `✓` mark; locked tiles carry `aria-disabled="false"` but announce their state through visually hidden text. Add to `RoadmapTile.astro` inside `.face`:

```astro
<span class="sr-only" data-state-text></span>
```

and set its text in the board script alongside `data-state`:

```ts
const stateText = tile.querySelector<HTMLElement>('[data-state-text]');
if (stateText) {
  stateText.textContent =
    tile.dataset.state === 'complete' ? ' (completed)'
    : tile.dataset.state === 'available' ? ' (ready to start)'
    : ' (locked — finish the earlier lessons first)';
}
```

Add the `.sr-only` utility to `global.css`:

```css
.sr-only {
  clip: rect(0 0 0 0);
  clip-path: inset(50%);
  height: 1px;
  overflow: hidden;
  position: absolute;
  white-space: nowrap;
  width: 1px;
}
```

- [ ] **Step 3: Check contrast in both themes**

Run Lighthouse (Chrome DevTools → Lighthouse → Accessibility) on the homepage and on a lesson page, in dark and light.
Expected: Accessibility score ≥ 95, with zero contrast failures. Raise `--text-muted` lightness in whichever theme fails.

- [ ] **Step 4: Verify reduced motion end to end**

Enable "Reduce motion" in OS settings, reload every page type.
Expected: no tilt on lesson cards, no board rotation, no transitions, and all content legible.

- [ ] **Step 5: Verify mobile layout**

In DevTools, test at 375px and 768px.
Expected: the board is flat and scrollable with no horizontal page overflow; the header wraps cleanly; the search dialog fits the viewport; tap targets are at least 44×44px.

- [ ] **Step 6: Verify in Safari and Firefox**

Expected: `backdrop-filter`, `mask-composite`, and `color-mix()` all render. Where Safari needs it, add the `-webkit-` prefixed `mask-composite: xor` alongside the standard `exclude`.

- [ ] **Step 7: Run the full suite and build**

```bash
npm test && npx astro build
```

Expected: all tests pass; build completes with zero warnings.

- [ ] **Step 8: Commit**

```bash
git add src
git commit -m "fix: accessibility, reduced motion, and cross-browser hardening"
```

---

### Task 17: CI and GitHub Pages deployment

**Files:**
- Create: `.github/workflows/deploy.yml`
- Modify: `README.md` (add build status and live link)

**Interfaces:**
- Consumes: the `validate`, `test`, and `build` npm scripts.
- Produces: the live site at `https://sumitsingh4411.github.io/ai-roadmap`.

- [ ] **Step 1: Create `.github/workflows/deploy.yml`**

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: 22
          cache: npm

      - run: npm ci

      - name: Validate content
        run: npm run validate

      - name: Run tests
        run: npm test

      - name: Build site
        run: npm run build

      - uses: actions/upload-pages-artifact@v3
        with:
          path: dist

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

`npm run build` triggers `prebuild`, so validation, `CURRICULUM.md`, and the search index are all regenerated before the build. The explicit `npm run validate` step exists so a content error fails fast with a clear step name.

- [ ] **Step 2: Verify the workflow locally first**

```bash
npm ci && npm run validate && npm test && npm run build
```

Expected: every command exits 0. This is exactly what CI runs.

- [ ] **Step 3: Confirm `CURRICULUM.md` is not stale**

```bash
npm run gen:curriculum && git diff --exit-code CURRICULUM.md
```

Expected: exit code 0 — no diff. If there is a diff, commit the regenerated file.

- [ ] **Step 4: Create the GitHub repository and push**

```bash
git remote add origin https://github.com/sumitsingh4411/ai-roadmap.git
git branch -M main
git push -u origin main
```

- [ ] **Step 5: Enable GitHub Pages**

In the repository: **Settings → Pages → Build and deployment → Source → GitHub Actions**.

- [ ] **Step 6: Verify the deployment**

Watch the Actions tab.
Expected: both jobs succeed and `https://sumitsingh4411.github.io/ai-roadmap` loads the isometric homepage.

Click through to a lesson.
Expected: the URL is `https://sumitsingh4411.github.io/ai-roadmap/lessons/what-is-ai`, CSS and the search index both load (confirming `base` is applied everywhere), and no console errors appear.

- [ ] **Step 7: Verify the GitHub reading path**

Open `https://github.com/sumitsingh4411/ai-roadmap/blob/main/CURRICULUM.md`.
Expected: the stage tables render, and every lesson link opens a correctly formatted lesson with working relative `**Next:**` navigation.

- [ ] **Step 8: Add the status badge to `README.md`**

```markdown
[![Deploy](https://github.com/sumitsingh4411/ai-roadmap/actions/workflows/deploy.yml/badge.svg)](https://github.com/sumitsingh4411/ai-roadmap/actions/workflows/deploy.yml)
```

- [ ] **Step 9: Commit and push**

```bash
git add .github README.md
git commit -m "ci: deploy to GitHub Pages on push to main"
git push
```
