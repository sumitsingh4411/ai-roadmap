# AI Learning Roadmap — Design

**Date:** 2026-07-24
**Status:** Approved
**Repo:** `github.com/sumitsingh4411/ai-roadmap`
**Live URL:** `https://sumitsingh4411.github.io/ai-roadmap`

## Purpose

A free, open-source beginner-to-practitioner roadmap for learning AI. The
curriculum is written as plain Markdown files in the repository. Those files are
the product: a reader can learn entirely from github.com without ever opening the
website. The website is a second, richer reader over the same files — a 3D
interactive roadmap with progress tracking and search.

**Non-goals for v1:** user accounts, a backend, comments, video hosting,
paid content, i18n.

## Success Criteria

1. All 34 lessons render correctly and readably on github.com.
2. `CURRICULUM.md` lets a GitHub-only reader follow the full path in order.
3. The site builds to static files and deploys to GitHub Pages via CI.
4. The homepage roadmap is visibly three-dimensional and reflects completion state.
5. Progress, search, and theme all work with JavaScript running locally only —
   no network calls after page load.
6. A broken content reference (missing lesson, unknown prerequisite, cycle)
   fails the build rather than shipping.

## Architecture

### Repository layout

```
content/
  roadmap.json              # node graph: ids, stages, grid positions, prerequisites
  lessons/                  # 34 Markdown lessons, the canonical content
    00-what-is-ai.md
    01-how-to-learn-ai.md
    ...
CURRICULUM.md               # ordered index of every lesson, linked
README.md                   # project intro + "learn straight from GitHub"
src/
  content.config.ts         # Astro collection: glob loader over ../content/lessons
  layouts/
  components/
  pages/
    index.astro             # isometric 3D roadmap
    lessons/[slug].astro    # lesson reader
  lib/
    progress.ts             # localStorage progress store
    search.ts               # search index builder + client query
    roadmap.ts              # graph loading, unlock logic, validation helpers
  styles/
scripts/
  validate-content.ts       # content integrity checks (run in CI and prebuild)
.github/workflows/deploy.yml
astro.config.mjs
```

Content sits at the repository root, not inside `src/`, so the GitHub file
browser presents it first and paths stay short and human. Astro reaches it with a
`glob()` loader whose `base` is `./content/lessons`. Content does not depend on
the site; the site depends on the content. Deleting `src/` would leave a
perfectly usable GitHub-based course.

### Lesson file format

Each lesson is standard Markdown with YAML frontmatter:

```yaml
---
title: "Neural Network Fundamentals"
stage: 4
order: 19
minutes: 45
difficulty: intermediate      # beginner | intermediate | advanced
prerequisites: ["linear-algebra", "classification"]
tags: ["deep-learning", "pytorch"]
summary: "What a neuron actually computes, and why stacking them works."
---
```

GitHub renders frontmatter as a table, so it reads as a useful metadata header
rather than noise. The body uses only portable Markdown — headings, fenced code
blocks with language hints, tables, blockquotes, links. No MDX, no custom
components, no HTML. Portability to github.com is the constraint that decides
every formatting question.

Every lesson follows the same five-part shape:

1. **Why this matters** — plain language, no jargon, one short paragraph.
2. **The concept** — explanation with analogies, building from what came before.
3. **In code** — runnable Python, self-contained, with expected output shown.
4. **Build this** — one concrete exercise, plus a stretch goal.
5. **Go deeper** — 3–5 curated free resources (docs, papers, videos) and an
   explicit "next lesson" pointer.

### `content/roadmap.json`

The graph the homepage renders and the unlock logic reads:

```json
{
  "stages": [
    { "id": 0, "name": "Orientation", "color": "violet" }
  ],
  "nodes": [
    {
      "id": "python-basics",
      "lesson": "02-python-basics",
      "stage": 1,
      "grid": { "x": 0, "y": 1 },
      "prerequisites": ["how-to-learn-ai"]
    }
  ]
}
```

`grid` holds integer coordinates on the isometric board; the renderer converts
them to screen space. Authoring positions as a grid rather than pixels keeps the
layout editable by hand and stable across viewport sizes.

## Curriculum — 34 lessons across 7 stages

**Stage 0 · Orientation (2)**
What AI, ML, DL and GenAI actually are · How to learn AI: study plan and mindset

**Stage 1 · Python & Data (5)**
Python basics · NumPy · Pandas · Data visualization · Working with real datasets

**Stage 2 · Math you actually need (3)**
Linear algebra · Calculus and gradients · Probability and statistics

**Stage 3 · Classical ML (8)**
ML fundamentals · Regression · Classification · Model evaluation · Feature
engineering · Clustering and PCA · Trees and ensembles · First end-to-end project

**Stage 4 · Deep Learning (6)**
Neural network fundamentals · Backpropagation and the training loop · PyTorch
essentials · CNNs and computer vision · Sequence models · Transformers and attention

**Stage 5 · Generative AI & LLMs (7)**
How LLMs work · Prompt engineering · Embeddings and vector search · RAG ·
Fine-tuning with LoRA · AI agents and tool use · Evaluation and guardrails

**Stage 6 · Ship it (3)**
MLOps basics · Deploying a model with FastAPI and Docker · Portfolio projects and
career paths

Ordering is a directed acyclic graph, not a straight line: Math (stage 2) and
Python (stage 1) are independent of each other, and both gate Classical ML.

## Visual Design

Two treatments, chosen deliberately for different jobs.

### Homepage — Isometric Skill World

The roadmap is a 3D board viewed from above, built with CSS 3D transforms
(`rotateX(58deg) rotateZ(-45deg)` on a `preserve-3d` container). Each node is an
extruded tile: layered `box-shadow` gives thickness, an inset top highlight gives
a light source. Labels are counter-rotated so text stays flat and legible.

State is expressed physically:

- **Locked** (prerequisites unmet) — dim, desaturated, sitting low.
- **Available** — cyan rim, soft outward glow.
- **Complete** — violet gradient fill, raised on the Z axis, stronger glow.

Palette: near-black canvas `#06060f`, violet `#7c3aed`→`#a78bfa`, cyan
`#22d3ee`→`#67e8f9`, radial light blooms behind the board, and a masked grid
floor for depth reference. Scroll drives a gentle camera drift.

### Lesson pages — Depth Stack

The lesson header is a glass card floating in Z-space: `backdrop-filter` blur, a
gradient rim-light border drawn with a masked pseudo-element, long soft shadow,
and a subtle tilt toward the cursor. Behind it sit two dimmer offset layers
implying a deck. The body below is a single readable column — the drama stays in
the chrome, never in the text the reader is trying to study.

### Motion and accessibility

All 3D transforms and parallax respect `prefers-reduced-motion: reduce`, which
collapses the board to a flat 2D layout with no drift or tilt. Contrast targets
WCAG AA for body text. Node state is never conveyed by color alone — icons and
text labels carry it too. The board is keyboard navigable, and every node is a
real link, so the roadmap works with JavaScript disabled.

Light theme uses the same geometry with a paper-toned canvas and darkened rim
lighting; both themes are defined as CSS custom properties on `:root`.

## Features

### Progress tracking

`localStorage` key `ai-roadmap:progress` holds an array of completed lesson
slugs. Lesson pages have a "Mark complete" toggle; the homepage reads the same
store to light up tiles, recompute which nodes are unlocked, and drive a
percentage bar. No accounts, no sync, no analytics. `src/lib/progress.ts` is the
only module that touches storage, and it degrades silently when storage is
unavailable (private browsing).

### Search

`scripts/build-search-index.ts` runs as an npm `prebuild` step, alongside
content validation. It walks `content/lessons` and emits
`public/search-index.json` — slug, title, stage, summary, headings, and stripped
body text. The client loads it on first open of the search dialog and runs a
small fuzzy matcher. Opened with `⌘K` or `/`, navigable by arrow keys.

### Code blocks

Shiki (built into Astro) highlights at build time, so no highlighting library
ships to the browser. A small client script adds a copy button to each `<pre>`.

### GitHub link

Every lesson page links to its own source file on GitHub, reinforcing that the
Markdown is the real artifact and inviting contributions.

## Data Flow

**Build:** `content/roadmap.json` and lesson frontmatter are parsed and validated
against Zod schemas → `astro build` generates one static HTML page per lesson
plus the homepage → the search index is written to `public/` → output lands in
`dist/`.

**Deploy:** push to `main` triggers `.github/workflows/deploy.yml`, which
installs, validates, builds, and publishes `dist/` to GitHub Pages. `astro.config.mjs`
sets `site: 'https://sumitsingh4411.github.io'` and `base: '/ai-roadmap'`; all
internal links are built from `import.meta.env.BASE_URL` so the base path is
never hardcoded.

**Runtime:** static HTML and CSS render immediately. JavaScript only adds
progress state, the search dialog, theme toggling, and cursor tilt. Nothing
fetches from the network after load.

## Error Handling and Validation

`scripts/validate-content.ts` runs before every build and in CI. It fails loudly on:

- a roadmap node referencing a lesson file that does not exist
- a lesson file not referenced by any roadmap node (orphan)
- a prerequisite id that matches no node
- a cycle in the prerequisite graph
- two nodes occupying the same grid coordinate
- frontmatter failing the Zod schema (missing title, unknown difficulty, etc.)

Runtime failures are handled by degrading, not crashing: corrupt `localStorage`
is discarded and reset; a failed search-index fetch shows "search unavailable"
while the site stays fully usable; a missing lesson URL renders a 404 page that
links back to the roadmap.

## Testing

Scaled to a static content site:

- **Content integrity** — the validation script above, as an automated test over
  the real `content/` directory. This is the highest-value test: content is the
  product and it is edited most often.
- **Unit tests** (Vitest) — `progress.ts` (add, remove, unlock computation,
  corrupt-storage recovery), `roadmap.ts` (topological ordering, cycle
  detection), and the search-index builder (stripping, field extraction).
- **Build verification** — `astro build` must complete with zero warnings in CI
  before deploy runs.

No end-to-end browser tests in v1; the interactive surface is small and the cost
would outweigh the benefit.

## Implementation Order

1. Scaffold Astro, configure the content collection over `content/lessons`, set
   base path, verify a trivial build.
2. `roadmap.json` with all 34 nodes plus the validation script and its tests.
3. Lesson layout, routing, `CURRICULUM.md`, `README.md` — with 2–3 real lessons.
4. Write the remaining 31 lessons.
5. Isometric homepage.
6. Progress tracking, then search, then theme toggle and copy buttons.
7. CI workflow and first deploy.

Content is written before the homepage polish so the site always has something
real to render, and so the largest chunk of work is not blocked behind visual
iteration.
