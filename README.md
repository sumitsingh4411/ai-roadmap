# AI Roadmap

A free, structured path from "what is AI?" to shipping your own model — 34 lessons, in dependency order, that you can read as plain files or as an interactive site.

## Learn straight from GitHub — no website needed

Every lesson is a plain Markdown file in `content/lessons/`. There is no build step between you and the content: open [`CURRICULUM.md`](CURRICULUM.md) and start reading. It lists all 34 lessons in the order you should do them, with time estimates and a one-line summary of each, and every link on that page points straight at the lesson file. GitHub renders the tables, the code blocks, and the images natively — nothing about this roadmap requires the site to be useful.

`CURRICULUM.md` is generated from `content/roadmap.json` by `npm run gen:curriculum`, so it's always in sync with the actual prerequisite graph. Don't edit it by hand.

## The interactive version

If you'd rather have progress tracking, a visual dependency graph, and search, the same content is published at **<https://sumitsingh4411.github.io/ai-roadmap>**.

## The seven stages

- **Stage 0 · Orientation** (2 lessons) — what AI/ML/DL/GenAI actually mean, and how to study this without burning out.
- **Stage 1 · Python & Data** (5 lessons) — Python basics, NumPy, pandas, data visualization, working with real datasets.
- **Stage 2 · Math You Actually Need** (3 lessons) — linear algebra, calculus, and probability & statistics, taught for the intuition, not the exam.
- **Stage 3 · Classical ML** (8 lessons) — ML fundamentals, regression, classification, evaluation, feature engineering, clustering & PCA, trees & ensembles, and your first end-to-end project.
- **Stage 4 · Deep Learning** (6 lessons) — neural networks, backprop and training, PyTorch, CNNs for vision, sequence models, and transformers.
- **Stage 5 · Generative AI & LLMs** (7 lessons) — how LLMs work, prompt engineering, embeddings, RAG, fine-tuning, AI agents, and evals & guardrails.
- **Stage 6 · Ship It** (3 lessons) — MLOps basics, deploying models, and building a portfolio for your career.

## How this repo is organised

- **`content/roadmap.json`** — the source of truth for the curriculum's structure: every stage, every node, and the prerequisite edges between them. This is what generates the dependency order, the graph on the site, and `CURRICULUM.md`.
- **`content/lessons/`** — the lessons themselves, one Markdown file per node, named `NN-slug.md`. Frontmatter carries the title, stage, order, time estimate, difficulty, prerequisites, tags, and summary; the body is the lesson.
- **`src/`** — the Astro site: the schema and helpers that read lesson content (`src/lib/`), the content collection wiring (`src/content.config.ts`), and the pages that render the roadmap.

Everything that isn't hand-written content — `CURRICULUM.md`, the search index, the site itself — is generated or built from those two `content/` sources.

## Local development

```bash
npm install
npm run dev      # start the site locally
npm test         # run the test suite
npm run build    # validate content, regenerate CURRICULUM.md, build the site
```

## Contributing

Lessons are plain Markdown: no HTML, no framework-specific syntax, nothing that stops the file from rendering correctly on github.com. If you're adding or editing a lesson, run `npm run validate` before opening a pull request — it checks frontmatter against the schema, confirms every prerequisite exists, and catches cycles in the roadmap graph. It must pass.

## Licence

Code is MIT licensed. Lesson content is licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
