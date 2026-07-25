<div align="center">

<a href="https://sumitsingh4411.github.io/ai-roadmap"><img src="https://sumitsingh4411.github.io/ai-roadmap/og.png" alt="AI Roadmap — learn AI from zero to shipping an LLM app" width="820" /></a>

# 🧠 AI Roadmap

</div>

**A free, open-source path from your first line of Python to shipping an LLM app.** 34 lessons in dependency order — read them on the polished web app, or straight from the Markdown right here.

[![Deploy](https://github.com/sumitsingh4411/ai-roadmap/actions/workflows/deploy.yml/badge.svg)](https://github.com/sumitsingh4411/ai-roadmap/actions/workflows/deploy.yml) ![lessons](https://img.shields.io/badge/lessons-34-8b5cf6?style=flat-square) ![reading](https://img.shields.io/badge/reading-~29h-22d3ee?style=flat-square) ![license](https://img.shields.io/badge/license-MIT%20%C2%B7%20CC%20BY%204.0-2dd4bf?style=flat-square) ![PRs](https://img.shields.io/badge/PRs-welcome-fb7185?style=flat-square)

### 👉 [**Open the interactive site**](https://sumitsingh4411.github.io/ai-roadmap) &nbsp;·&nbsp; [Start the first lesson](https://sumitsingh4411.github.io/ai-roadmap/lessons/what-is-ai) &nbsp;·&nbsp; [Full curriculum](CURRICULUM.md)

---

## Why this exists

Most people learning AI drown in scattered tutorials with no order and no idea what to learn next. This is one **dependency-ordered** path: every lesson tells you exactly what it assumes you already know, so you always know where you are and what comes next — from "what even *is* AI?" all the way to fine-tuning models and serving them behind an API.

- 🆓 **Free and open forever** — MIT-licensed code, CC BY 4.0 content.
- 📖 **Read it anywhere** — a fast, searchable web app, or plain Markdown on GitHub. The lessons never need the website to be useful.
- ✅ **Runnable, verified code** — every sample was actually executed; the output shown is the output it produces.
- 📈 **Tracks your progress** — the site remembers what you've finished, in your browser, no account.

## How to use it

| You want to… | Do this |
|---|---|
| Just learn, nicely | Open the **[web app](https://sumitsingh4411.github.io/ai-roadmap)** — sidebar, search (`⌘K`), progress tracking, dark/light. |
| Read on GitHub | Start at **[CURRICULUM.md](CURRICULUM.md)** and follow the order. Every lesson is a file in [`content/lessons/`](content/lessons). |
| Run it locally | `npm install` → `npm run dev`. Build with `npm run build`, test with `npm test`. |

## The curriculum

**7 stages · 34 lessons · ~29 hours.** Work top to bottom, or jump in wherever you already have the prerequisites.

| Stage | Focus | Lessons |
|---|---|:--:|
| **0 · Orientation** | Get your bearings before writing code | 2 |
| **1 · Python & Data** | The everyday tools — Python, NumPy, pandas, plots | 5 |
| **2 · Math You Actually Need** | Just enough linear algebra, calculus & statistics | 3 |
| **3 · Classical ML** | How machines learn from tables — regression to ensembles | 8 |
| **4 · Deep Learning** | Neural networks, from one neuron to the transformer | 6 |
| **5 · Generative AI & LLMs** | How LLMs work, and how to build with them | 7 |
| **6 · Ship It** | Take a model off your laptop and ship it | 3 |

<details>
<summary><b>Stage 0 — Orientation</b> · 2 lessons</summary>

- **[What AI, ML, Deep Learning and GenAI Actually Are](https://sumitsingh4411.github.io/ai-roadmap/lessons/what-is-ai)** &nbsp;·&nbsp; 20 min · beginner &nbsp;·&nbsp; [source](content/lessons/00-what-is-ai.md)  
  <sub>The four words everyone mixes up, sorted out once, with a mental model you can keep.</sub>
- **[How to Learn AI Without Burning Out](https://sumitsingh4411.github.io/ai-roadmap/lessons/how-to-learn-ai)** &nbsp;·&nbsp; 15 min · beginner &nbsp;·&nbsp; [source](content/lessons/01-how-to-learn-ai.md)  
  <sub>A realistic schedule, the order to learn things in, and the three traps that stop most beginners.</sub>

</details>

<details>
<summary><b>Stage 1 — Python & Data</b> · 5 lessons</summary>

- **[Python Basics](https://sumitsingh4411.github.io/ai-roadmap/lessons/python-basics)** &nbsp;·&nbsp; 60 min · beginner &nbsp;·&nbsp; [source](content/lessons/02-python-basics.md)  
  <sub>Variables, lists, dicts, loops, functions, imports, and how to read the error messages you'll see constantly.</sub>
- **[NumPy](https://sumitsingh4411.github.io/ai-roadmap/lessons/numpy)** &nbsp;·&nbsp; 45 min · beginner &nbsp;·&nbsp; [source](content/lessons/03-numpy.md)  
  <sub>Why arrays beat lists for numeric work, plus shape, dtype, indexing, broadcasting, and vectorised math.</sub>
- **[Pandas](https://sumitsingh4411.github.io/ai-roadmap/lessons/pandas)** &nbsp;·&nbsp; 50 min · beginner &nbsp;·&nbsp; [source](content/lessons/04-pandas.md)  
  <sub>Series and DataFrames, loading CSVs, selecting and filtering rows, grouping, and handling missing values.</sub>
- **[Data Visualization](https://sumitsingh4411.github.io/ai-roadmap/lessons/data-visualization)** &nbsp;·&nbsp; 40 min · beginner &nbsp;·&nbsp; [source](content/lessons/05-data-visualization.md)  
  <sub>Choosing the right chart type, matplotlib basics, plotting straight from pandas, and reading what a histogram tells you.</sub>
- **[Real Datasets](https://sumitsingh4411.github.io/ai-roadmap/lessons/real-datasets)** &nbsp;·&nbsp; 45 min · beginner &nbsp;·&nbsp; [source](content/lessons/06-real-datasets.md)  
  <sub>Where to find datasets, loading messy CSVs, fixing types and dates, removing duplicates, spotting outliers, and a reusable cleaning checklist.</sub>

</details>

<details>
<summary><b>Stage 2 — Math You Actually Need</b> · 3 lessons</summary>

- **[Linear Algebra](https://sumitsingh4411.github.io/ai-roadmap/lessons/linear-algebra)** &nbsp;·&nbsp; 50 min · beginner &nbsp;·&nbsp; [source](content/lessons/07-linear-algebra.md)  
  <sub>Vectors and matrices as the data structures behind every model, dot products as weighted sums, and matrix multiplication as batch prediction.</sub>
- **[Calculus](https://sumitsingh4411.github.io/ai-roadmap/lessons/calculus)** &nbsp;·&nbsp; 50 min · intermediate &nbsp;·&nbsp; [source](content/lessons/08-calculus.md)  
  <sub>Derivatives as slope, the chain rule, and gradient descent implemented by hand in NumPy — how a model actually learns.</sub>
- **[Probability & Statistics](https://sumitsingh4411.github.io/ai-roadmap/lessons/probability-stats)** &nbsp;·&nbsp; 50 min · beginner &nbsp;·&nbsp; [source](content/lessons/09-probability-stats.md)  
  <sub>Distributions, mean and variance, conditional probability, Bayes' theorem, sampling, and what a p-value actually means.</sub>

</details>

<details>
<summary><b>Stage 3 — Classical ML</b> · 8 lessons</summary>

- **[ML Fundamentals](https://sumitsingh4411.github.io/ai-roadmap/lessons/ml-fundamentals)** &nbsp;·&nbsp; 45 min · beginner &nbsp;·&nbsp; [source](content/lessons/10-ml-fundamentals.md)  
  <sub>Supervised vs unsupervised learning, the train/validation/test split, and overfitting vs underfitting through the bias-variance tradeoff.</sub>
- **[Regression](https://sumitsingh4411.github.io/ai-roadmap/lessons/regression)** &nbsp;·&nbsp; 50 min · beginner &nbsp;·&nbsp; [source](content/lessons/11-regression.md)  
  <sub>Linear regression from the normal equation to scikit-learn, the MSE/MAE/R² metrics, and Ridge/Lasso regularisation to fight overfitting.</sub>
- **[Classification](https://sumitsingh4411.github.io/ai-roadmap/lessons/classification)** &nbsp;·&nbsp; 50 min · beginner &nbsp;·&nbsp; [source](content/lessons/12-classification.md)  
  <sub>Logistic regression and the sigmoid, decision boundaries, k-nearest neighbours, and strategies for more than two classes.</sub>
- **[Model Evaluation](https://sumitsingh4411.github.io/ai-roadmap/lessons/model-evaluation)** &nbsp;·&nbsp; 45 min · intermediate &nbsp;·&nbsp; [source](content/lessons/13-model-evaluation.md)  
  <sub>The confusion matrix, precision, recall, F1, ROC-AUC, cross-validation, and why accuracy alone can make a worthless model look great.</sub>
- **[Feature Engineering](https://sumitsingh4411.github.io/ai-roadmap/lessons/feature-engineering)** &nbsp;·&nbsp; 45 min · intermediate &nbsp;·&nbsp; [source](content/lessons/14-feature-engineering.md)  
  <sub>Scaling, encoding categoricals, dates, binning, interaction terms, and data leakage — raising a model's score with features, not a new algorithm.</sub>
- **[Clustering & PCA](https://sumitsingh4411.github.io/ai-roadmap/lessons/clustering-pca)** &nbsp;·&nbsp; 45 min · intermediate &nbsp;·&nbsp; [source](content/lessons/15-clustering-pca.md)  
  <sub>k-means clustering, choosing k with the elbow method, hierarchical clustering, and PCA as compression with explained variance.</sub>
- **[Trees & Ensembles](https://sumitsingh4411.github.io/ai-roadmap/lessons/trees-ensembles)** &nbsp;·&nbsp; 50 min · intermediate &nbsp;·&nbsp; [source](content/lessons/16-trees-ensembles.md)  
  <sub>Decision trees and how splits are chosen, random forests as bagging, gradient boosting, and feature importance.</sub>
- **[First ML Project](https://sumitsingh4411.github.io/ai-roadmap/lessons/first-ml-project)** &nbsp;·&nbsp; 90 min · intermediate &nbsp;·&nbsp; [source](content/lessons/17-first-ml-project.md)  
  <sub>A full guided ML pipeline end to end: problem framing, EDA, cleaning, features, baseline, iteration, evaluation, and writing up results.</sub>

</details>

<details>
<summary><b>Stage 4 — Deep Learning</b> · 6 lessons</summary>

- **[Neural Networks](https://sumitsingh4411.github.io/ai-roadmap/lessons/neural-networks)** &nbsp;·&nbsp; 55 min · intermediate &nbsp;·&nbsp; [source](content/lessons/18-neural-networks.md)  
  <sub>What one neuron computes, why nonlinear activations are non-negotiable, how depth builds representations, and the universal approximation intuition.</sub>
- **[Backprop & Training](https://sumitsingh4411.github.io/ai-roadmap/lessons/backprop-training)** &nbsp;·&nbsp; 60 min · intermediate &nbsp;·&nbsp; [source](content/lessons/19-backprop-training.md)  
  <sub>How the chain rule turns one output error into a gradient for every weight in a network, and how learning rate, epochs and batches shape training.</sub>
- **[PyTorch](https://sumitsingh4411.github.io/ai-roadmap/lessons/pytorch)** &nbsp;·&nbsp; 60 min · intermediate &nbsp;·&nbsp; [source](content/lessons/20-pytorch.md)  
  <sub>Tensors, autograd, nn.Module, optimisers, and the canonical training loop — the framework that automates the backprop you just wrote by hand.</sub>
- **[CNNs & Vision](https://sumitsingh4411.github.io/ai-roadmap/lessons/cnns-vision)** &nbsp;·&nbsp; 55 min · intermediate &nbsp;·&nbsp; [source](content/lessons/21-cnns-vision.md)  
  <sub>Convolution as a learned filter, stride, padding and pooling, how a CNN's shapes flow layer to layer, and transfer learning with a pretrained backbone.</sub>
- **[Sequence Models](https://sumitsingh4411.github.io/ai-roadmap/lessons/sequence-models)** &nbsp;·&nbsp; 50 min · advanced &nbsp;·&nbsp; [source](content/lessons/22-sequence-models.md)  
  <sub>Why order matters, how RNNs process sequences step by step, the vanishing gradient problem, LSTM/GRU, and why attention replaced them.</sub>
- **[Transformers](https://sumitsingh4411.github.io/ai-roadmap/lessons/transformers)** &nbsp;·&nbsp; 70 min · advanced &nbsp;·&nbsp; [source](content/lessons/23-transformers.md)  
  <sub>Attention as a learned lookup over query, key and value, self-attention and multi-head attention, positional encoding, and the encoder/decoder split.</sub>

</details>

<details>
<summary><b>Stage 5 — Generative AI & LLMs</b> · 7 lessons</summary>

- **[How LLMs Work](https://sumitsingh4411.github.io/ai-roadmap/lessons/how-llms-work)** &nbsp;·&nbsp; 55 min · intermediate &nbsp;·&nbsp; [source](content/lessons/24-how-llms-work.md)  
  <sub>Tokenization, next-token prediction, pretraining vs post-training, context windows, temperature and sampling, and why models hallucinate.</sub>
- **[Prompt Engineering](https://sumitsingh4411.github.io/ai-roadmap/lessons/prompt-engineering)** &nbsp;·&nbsp; 45 min · beginner &nbsp;·&nbsp; [source](content/lessons/25-prompt-engineering.md)  
  <sub>Clear instructions, few-shot examples, chain-of-thought, structured output, system prompts, and fixing a failing prompt in documented iterations.</sub>
- **[Embeddings](https://sumitsingh4411.github.io/ai-roadmap/lessons/embeddings)** &nbsp;·&nbsp; 45 min · intermediate &nbsp;·&nbsp; [source](content/lessons/26-embeddings.md)  
  <sub>Text as vectors, cosine similarity, embedding models, vector databases, and chunking strategy, with a real semantic search built in NumPy.</sub>
- **[Retrieval-Augmented Generation (RAG)](https://sumitsingh4411.github.io/ai-roadmap/lessons/rag)** &nbsp;·&nbsp; 60 min · intermediate &nbsp;·&nbsp; [source](content/lessons/27-rag.md)  
  <sub>Why retrieval beats stuffing the context window, the ingest-chunk-embed-retrieve-generate pipeline, chunk sizing, and common RAG failure modes.</sub>
- **[Fine-tuning](https://sumitsingh4411.github.io/ai-roadmap/lessons/fine-tuning)** &nbsp;·&nbsp; 60 min · advanced &nbsp;·&nbsp; [source](content/lessons/28-fine-tuning.md)  
  <sub>When fine-tuning beats RAG or prompting, full fine-tuning vs LoRA/PEFT, dataset preparation, and evaluating the result, with a real LoRA parameter-count demo.</sub>
- **[AI Agents](https://sumitsingh4411.github.io/ai-roadmap/lessons/ai-agents)** &nbsp;·&nbsp; 60 min · advanced &nbsp;·&nbsp; [source](content/lessons/29-ai-agents.md)  
  <sub>Tool use, the reason-act loop, planning, memory, multi-step failure modes, and cost control, with a real non-LLM demo of the loop mechanics.</sub>
- **[Evals & Guardrails](https://sumitsingh4411.github.io/ai-roadmap/lessons/evals-guardrails)** &nbsp;·&nbsp; 50 min · advanced &nbsp;·&nbsp; [source](content/lessons/30-evals-guardrails.md)  
  <sub>Why manual spot-checking doesn't scale, building an eval set, LLM-as-judge and its biases, regression testing, and input/output guardrails.</sub>

</details>

<details>
<summary><b>Stage 6 — Ship It</b> · 3 lessons</summary>

- **[MLOps Basics](https://sumitsingh4411.github.io/ai-roadmap/lessons/mlops-basics)** &nbsp;·&nbsp; 45 min · intermediate &nbsp;·&nbsp; [source](content/lessons/31-mlops-basics.md)  
  <sub>Experiment tracking, model and data versioning, reproducibility, and drift monitoring — what keeps a shipped model trustworthy after it leaves your notebook.</sub>
- **[Deploying a Model](https://sumitsingh4411.github.io/ai-roadmap/lessons/deploying-models)** &nbsp;·&nbsp; 55 min · intermediate &nbsp;·&nbsp; [source](content/lessons/32-deploying-models.md)  
  <sub>Wrapping a trained model behind a validated FastAPI endpoint, measuring latency and batching, Dockerizing it, and free hosting options.</sub>
- **[Portfolio and Career](https://sumitsingh4411.github.io/ai-roadmap/lessons/portfolio-career)** &nbsp;·&nbsp; 40 min · beginner &nbsp;·&nbsp; [source](content/lessons/33-portfolio-career.md)  
  <sub>What makes a project worth showing, writing a README that gets read, the real AI job families, and how to keep learning after this roadmap.</sub>

</details>

## How to get the most out of it

- **Follow the order, but skip what you know.** Each lesson lists its prerequisites — if you already have them, jump ahead.
- **Run every code sample.** Don't just read it. Type it, break it, change the numbers. That's where the learning is.
- **Do the "Build this" exercise** at the end of each lesson before moving on. A little friction now saves a lot of confusion later.
- **Ship three projects** as you go (a data project, a deep-learning project, an LLM app). A public repo with a good README beats any certificate.
- **A few focused hours a week beats cramming.** The whole path is ~29 hours of reading — steady wins.

## Where to go next

Finished the 34 lessons? The **[What's next →](https://sumitsingh4411.github.io/ai-roadmap/advanced)** page is a curated guide to going further — specializing (computer vision, NLP, reinforcement learning, diffusion models, LLMs from scratch), practicing on Kaggle, reading the field, MLOps, communities, and turning it all into a career. Every resource there is free.

## How your progress works

Mark a lesson complete on the site and it saves to your browser's local storage — no sign-up, no tracking, nothing leaves your machine. The homepage spine and the sidebar light up as you go, and a lesson only counts as "unlocked" once its prerequisites are done.

## Built with

[Astro](https://astro.build) static site · TypeScript · Markdown content · Shiki syntax highlighting · deployed to GitHub Pages by CI, which validates content and runs 100+ tests on every push. The lessons are the product; the site is a nice reader over them.

## Contributing

Spotted a bug or a clearer explanation? Issues and PRs are welcome.

- Lessons live in [`content/lessons/`](content/lessons) as portable Markdown — no site-specific syntax, so they read cleanly on GitHub too.
- Run `npm run validate -- --strict` and `npm test` before opening a PR; CI runs both.
- Lesson order and prerequisites are defined in [`content/roadmap.json`](content/roadmap.json).

## License

Code is [MIT](LICENSE). Lesson content is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — use it, remix it, just credit the source.

---

<div align="center"><sub>Built to be the roadmap I wish I'd had. ⭐ it if it helps — and <a href="https://sumitsingh4411.github.io/ai-roadmap">start learning →</a></sub></div>
