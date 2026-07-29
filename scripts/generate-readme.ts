import { writeFileSync } from 'node:fs';
import { readRoadmap, readLessonFiles } from './lib/content-io';
import { topologicalOrder } from '../src/lib/roadmap';
import { lessonSchema } from '../src/lib/lesson';

const SITE = 'https://sumitsingh4411.github.io/ai-roadmap';
const roadmap = readRoadmap();
const files = readLessonFiles();
const bySlug = new Map(files.map((f) => [f.slug, f]));
const ordered = topologicalOrder(roadmap.nodes);
const data = (slug: string) => lessonSchema.parse(bySlug.get(slug)!.frontmatter);

const totalMin = files.reduce((n, f) => n + lessonSchema.parse(f.frontmatter).minutes, 0);
const hours = Math.round(totalMin / 60);
const stageCount = roadmap.stages.filter((s) => ordered.some((n) => n.stage === s.id)).length;
const lessonCount = files.length;

const EMOJI: Record<number, string> = { 0: '🧭', 1: '🐍', 2: '📐', 3: '🌳', 4: '🧠', 5: '🤖', 6: '🚀' };
const FOCUS: Record<number, string> = {
  0: 'Get your bearings before writing code',
  1: 'The everyday tools — Python, NumPy, pandas, plots',
  2: 'Just enough linear algebra, calculus & statistics',
  3: 'How machines learn from tables — regression to ensembles',
  4: 'Neural networks, from one neuron to the transformer',
  5: 'How LLMs work, and how to build with them',
  6: 'Take a model off your laptop and ship it',
};
const badge = (l: string, m: string, c: string) =>
  `![${l}](https://img.shields.io/badge/${encodeURIComponent(l)}-${encodeURIComponent(m)}-${c}?style=flat-square)`;

let out = '';

// ---------- Hero ----------
out += `<div align="center">\n\n`;
out += `<a href="${SITE}"><img src="${SITE}/og.png" alt="AI Roadmap — learn AI from zero to shipping an LLM app" width="840" /></a>\n\n`;
out += `# 🧠 AI Roadmap\n\n`;
out += `### Learn AI from zero — ${lessonCount} free, hands-on lessons from your first line of Python to shipping an LLM app.\n\n`;
out += `[![Deploy](https://github.com/sumitsingh4411/ai-roadmap/actions/workflows/deploy.yml/badge.svg)](https://github.com/sumitsingh4411/ai-roadmap/actions/workflows/deploy.yml) `;
out += `${badge('lessons', String(lessonCount), '8b5cf6')} `;
out += `${badge('reading', '~' + hours + 'h', '22d3ee')} `;
out += `${badge('stages', String(stageCount), '38bdf8')} `;
out += `${badge('license', 'MIT · CC BY 4.0', '2dd4bf')} `;
out += `${badge('PRs', 'welcome', 'fb7185')}\n\n`;
out += `**[🌐 Open the site](${SITE})** &nbsp;·&nbsp; **[▶️ Start lesson 1](${SITE}/lessons/${ordered[0].id})** &nbsp;·&nbsp; `;
out += `**[🗺️ Roadmap](${SITE}/roadmap)** &nbsp;·&nbsp; **[🛠️ Projects](${SITE}/projects)** &nbsp;·&nbsp; **[🎯 What's next](${SITE}/advanced)**\n\n`;
out += `</div>\n\n`;
out += `---\n\n`;

// ---------- Why ----------
out += `## ✨ Why this roadmap\n\n`;
out += `Most people learning AI drown in scattered tutorials with no order and no idea what to learn next. This is **one dependency-ordered path** — every lesson tells you exactly what it assumes you already know, so you always know where you are and what comes next. It runs from *"what even is AI?"* all the way to fine-tuning models, building agents, and shipping them.\n\n`;
out += `- 🆓 **Free and open forever** — MIT code, CC BY 4.0 content. No sign-up, paywall, or ads.\n`;
out += `- 📖 **Learn right here on GitHub** — every lesson below is a plain Markdown file you can read without leaving. The [website](${SITE}) is the same content with progress tracking, search, and a visual roadmap.\n`;
out += `- ✅ **Runnable, verified code** — every sample was actually executed; the output you see is the output it produces.\n`;
out += `- 🧩 **Build as you go** — ${'30+'} project ideas, each mapped to the lesson it builds on.\n\n`;

// ---------- Quick start ----------
out += `## 🚀 Quick start\n\n`;
out += `| You want to… | Do this |\n|---|---|\n`;
out += `| **Just learn** | Open the **[web app](${SITE})** — sidebar, search (\`⌘K\`), progress tracking, dark/light. |\n`;
out += `| **Read on GitHub** | Start at **[${data(ordered[0].id).title}](content/lessons/${roadmap.nodes.find((n) => n.id === ordered[0].id)!.lesson}.md)** and follow the curriculum below, top to bottom. |\n`;
out += `| **Run it locally** | \`npm install\` → \`npm run dev\`. Build with \`npm run build\`, test with \`npm test\`. |\n\n`;
out += `> **New to all this?** Do the lessons in order. Already know some? Each lesson lists its prerequisites — skip ahead whenever you already have them.\n\n`;

// ---------- Curriculum ----------
out += `## 🗺️ The curriculum\n\n`;
out += `**${lessonCount} lessons · ${stageCount} stages · ~${hours} hours.** Click any lesson to read it right here on GitHub.\n\n`;
out += `| Stage | Focus | Lessons |\n|---|---|:--:|\n`;
for (const stage of roadmap.stages) {
  const ls = ordered.filter((n) => n.stage === stage.id);
  if (!ls.length) continue;
  out += `| ${EMOJI[stage.id]} **${stage.id} · ${stage.name}** | ${FOCUS[stage.id]} | ${ls.length} |\n`;
}
out += `\n`;

for (const stage of roadmap.stages) {
  const ls = ordered.filter((n) => n.stage === stage.id);
  if (!ls.length) continue;
  out += `<details>\n<summary><b>${EMOJI[stage.id]} Stage ${stage.id} · ${stage.name}</b> &nbsp;—&nbsp; ${ls.length} lessons</summary>\n\n`;
  out += `| # | Lesson | Time | Level | What you'll learn |\n|--:|---|--:|---|---|\n`;
  for (const n of ls) {
    const d = data(n.id);
    out += `| \`${String(d.order).padStart(2, '0')}\` | **[${d.title}](content/lessons/${n.lesson}.md)** | ${d.minutes}m | ${d.difficulty} | ${d.summary} |\n`;
  }
  out += `\n</details>\n\n`;
}
out += `> Prefer a table of everything at a glance? See **[CURRICULUM.md](CURRICULUM.md)**.\n\n`;

// ---------- Projects ----------
out += `## 🛠️ Build projects\n\n`;
out += `You learn AI by making things. The **[Projects page](${SITE}/projects)** has 30+ ideas across four tiers, each mapped to a lesson:\n\n`;
out += `- 🟦 **Beginner** — Titanic predictor, a stats CLI, an MNIST digit recognizer, a tic-tac-toe AI.\n`;
out += `- 🟪 **Intermediate** — semantic search, transfer-learning image classifier, an end-to-end churn model.\n`;
out += `- 🟥 **Advanced / GenAI** — a RAG chatbot over your docs, LoRA fine-tuning, an AI agent, reproduce nanoGPT.\n`;
out += `- 🟨 **Capstone** — a full-stack AI product, a Kaggle competition, reproduce a paper, an open-source contribution.\n\n`;

// ---------- What's next ----------
out += `## 🎯 After the roadmap\n\n`;
out += `Finished? The **[What's next page](${SITE}/advanced)** is a curated guide to going further — specializing (computer vision, RL, diffusion, LLMs from scratch), practicing on Kaggle, free courses & books, communities, MLOps, and turning it all into a career. Every resource is free.\n\n`;

// ---------- Tips ----------
out += `## 💡 How to get the most out of it\n\n`;
out += `- **Run every code sample.** Don't just read it — type it, break it, change the numbers. That's where the learning is.\n`;
out += `- **Do the "Build this" exercise** at the end of each lesson before moving on.\n`;
out += `- **Ship 3 projects** as you go (a data project, a deep-learning project, an LLM app). A public repo beats any certificate.\n`;
out += `- **A few focused hours a week beats cramming** — the whole path is ~${hours} hours of reading.\n\n`;

// ---------- Build / contribute / license ----------
out += `## 🧑‍💻 Built with\n\n`;
out += `[Astro](https://astro.build) static site · TypeScript · Markdown content · Shiki highlighting · deployed to GitHub Pages by CI, which validates content and runs 100+ tests on every push. The lessons are the product; the site is a nice reader over them.\n\n`;
out += `## 🤝 Contributing\n\n`;
out += `Spotted a bug or a clearer explanation? Issues and PRs are welcome.\n\n`;
out += `- Lessons live in [\`content/lessons/\`](content/lessons) as portable Markdown — no site-specific syntax, so they read cleanly on GitHub too.\n`;
out += `- Run \`npm run validate -- --strict\` and \`npm test\` before opening a PR; CI runs both.\n`;
out += `- Lesson order and prerequisites are defined in [\`content/roadmap.json\`](content/roadmap.json).\n\n`;
out += `## 📄 License\n\n`;
out += `Code is [MIT](LICENSE). Lesson content is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — use it, remix it, just credit the source.\n\n`;
out += `---\n\n`;
out += `<div align="center"><sub>Built to be the roadmap I wish I'd had. ⭐ it if it helps — and <a href="${SITE}">start learning →</a></sub></div>\n`;

writeFileSync('README.md', out, 'utf8');
console.log(`✓ Wrote README.md — ${out.length} chars, ${lessonCount} lessons across ${stageCount} stages`);
