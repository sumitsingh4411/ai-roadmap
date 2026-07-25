---
title: "Portfolio and Career"
stage: 6
order: 33
minutes: 40
difficulty: beginner
prerequisites: ["deploying-models", "evals-guardrails"]
tags: ["career", "portfolio"]
summary: "What makes a project worth showing, writing a README that gets read, the real AI job families, and how to keep learning after this roadmap."
---

# Portfolio and Career

## Why this matters

You didn't just read 33 lessons — you built things. A full classical ML
pipeline with an honest baseline and a documented tradeoff (lesson 17). A
RAG system over your own notes, with a real bug you found and either
fixed or consciously left (lesson 27). A model wrapped in a validated API
and, if you did the deploy stretch goal, actually reachable from outside
your own machine (lesson 32). Nobody sees any of that unless you show it
well — a brilliant project with no write-up and a repo nobody can run is,
to everyone but you, indistinguishable from a project that doesn't exist.
This lesson is the last piece of craft in the roadmap: not new technical
material, but how to make the work you already did actually count, and
an honest look at what happens after you close this tab.

## The concept

**What makes a project worth showing.** Four things, in order of how
often they're missing. First, it has to solve something real enough that
you had to make actual decisions — a copied tutorial with the serial
numbers filed off is usually obvious to anyone who's seen the same
tutorial, and it teaches nothing about how you think. Every project
you built following this roadmap's exercises clears this bar already,
because each one asked you to make a judgment call the lesson didn't
make for you (which metric matters, whether to fix an outlier, what
`k` to retrieve). Second, it has an honest number and an honest limit —
"72% accuracy, and here's specifically the case where it fails" reads as
far more credible than an unqualified "it works," the same lesson
13's precision/recall work and lesson 17's write-up template already
drilled in. Third, it's runnable by someone who isn't you — clear setup
steps, pinned dependencies (lesson 31's reproducibility habit, paying off
again here), and ideally a live demo or deployed endpoint (lesson 32) so
a reviewer can try it without touching your code at all. Fourth, depth
beats breadth: three projects with a real write-up, a real limitation
section, and evidence you iterated will always beat ten repos that are
each one commit of boilerplate.

**Writing a README that gets read.** The realistic constraint: whoever's
looking — a hiring manager, a recruiter doing a first pass, an engineer
skimming your GitHub before a call — gives your README something like
60 seconds before deciding whether to look closer. That means the most
important sentence in the whole document is the first one, and it needs
to say, in plain language, what the project does and why it exists —
not the tech stack, not "a machine learning project I built," but the
actual problem, the way lesson 17's problem-framing step insisted on.
Everything after that supports skimming: a screenshot or short demo clip
near the top if there's anything visual to show; exact setup and run
commands that work if copy-pasted; your real result with its real
caveat; and a short "what I'd do next" section, which does double duty —
it's useful information, and it signals you know your project's
limitations rather than believing your own hype.

**The main AI job families, honestly.** Titles are inconsistent between
companies — the same responsibilities show up under different names
depending on where you look — so treat these as rough shapes, not fixed
categories, and always read the actual listed responsibilities over the
title. **Data Scientist**: closer to analysis than engineering day to
day — SQL, pandas, dashboards, and a lot of communicating findings to
people who don't want to see the model, only the conclusion; Stage 1 and
3 of this roadmap are the core toolkit. **Machine Learning Engineer**:
takes a model from a notebook to something running in production — closer
to software engineering than research most days, and lessons 31-32 are
the direct toolkit. **MLOps / ML Platform Engineer**: builds the
infrastructure *other* ML people use — pipelines, tracking systems,
monitoring — a specialization inside lesson 31's territory, with more
DevOps overlap than modeling. **Applied Scientist / Research Engineer**:
closer to research — reading papers, running experiments to move a
metric — draws most heavily on Stage 2 and 4, and postings for this title
disproportionately (not universally) expect a graduate degree.
**AI Engineer / LLM Application Engineer**: the newest of these titles,
building products on top of existing foundation models — prompting, RAG,
agents, evals — Stage 5's direct toolkit, and notably a role that usually
doesn't train models from scratch at all. **Data Engineer**: adjacent to
all of the above and barely touched by this roadmap — building and
maintaining the pipelines that get data into a usable state before any
of the other roles can do anything with it; every title above quietly
depends on this one working. The honest reality check underneath all six:
most real day-to-day work, regardless of title, is debugging pipelines,
cleaning data, and writing ordinary code — closer to lesson 17's cleaning
step than to any headline-grabbing demo, on every team, at every company.

**How to keep learning after this roadmap.** Pick one thing from Stage 3,
4, or 5 that genuinely interested you and go deeper on it specifically,
rather than trying to stay broad forever — this roadmap was intentionally
broad so you could find out what that is. Read the paper or documentation
behind a tool you already use before reaching for the next new one; you
now have the background (Stage 2's math, Stage 4's deep learning) to
actually follow a real ML paper's methods section, which most people
never build up to. Kaggle competitions are still one of the best ways to
practice classical ML against a real leaderboard and other people's
published solutions, once you've placed. And the fastest way to find the
gaps in your own understanding is still to build something a little too
big for your current skill and get stuck on purpose.

**Where the community is.** [Kaggle](https://www.kaggle.com/) is both a
competition platform and a forum where people openly discuss approaches
after competitions close — genuinely useful even if you never compete.
The [Hugging Face forums](https://discuss.huggingface.co/) are active and
specific to the modern deep learning and LLM tooling from Stage 4 and 5.
Local and virtual meetups (PyData chapters, local ML meetups) exist in
most sizable cities and are a low-pressure way to talk to people doing
this work professionally. None of these require you to already be an
expert to show up — showing up with a real question from a real project
you built is exactly what they're for.

## In code

There isn't new runnable code in this lesson — the only thing worth
showing in a fence is a README structure you can actually copy:

```markdown
# Customer Churn Prediction

One sentence: what this predicts, for whom, and why it matters.

## Demo
[Link to the deployed endpoint, or a screenshot/GIF of it running.]

## Result
The one number that matters, plus its honest caveat.
e.g. "72% test accuracy; recall on churners was only 9% at the default
threshold, which is unacceptable for the actual business use — see
'What I'd do next'."

## How to run it
git clone ...
pip install -r requirements.txt
python train.py      # reproduces the result above
uvicorn app:app       # runs the deployed service locally

## What I'd do next
The one or two things you'd actually improve, and why you didn't
already — this is a feature, not an admission of failure.
```

This is intentionally close to lesson 17's write-up template — a good
README and a good project write-up are close to the same document, aimed
at a slightly less technical reader in slightly less time.

## Build this

Write and publish a portfolio README for the three projects you built in
this roadmap: the end-to-end ML pipeline (lesson 17), the RAG system over
your own notes (lesson 27), and the deployed model service (lesson 32).
For each one, use the template above: one clear sentence on what it does,
your real result with its real caveat, exact steps to run it, and one
honest "what I'd do next." Push each project to its own public GitHub
repository if it isn't already there, and link all three from a top-level
profile README or a short personal page.

**Stretch:** for one of the three, actually act on your own "what I'd do
next" — pick the smallest listed improvement and implement it. Update
that project's README to describe what changed and by how much. That's
the entire loop this roadmap has been teaching, one more time, on your
own project instead of a guided one: try something, measure it honestly,
write down what you found.

## Go deeper

- [Make a README](https://www.makeareadme.com/) — a focused, practical guide to what a README needs and why, independent of any specific language or framework.
- [freeCodeCamp: Machine Learning with Python](https://www.freecodecamp.org/learn/machine-learning-with-python/) — a free, project-based certification if you want structured practice beyond this roadmap's exercises.
- [swyx: The Rise of the AI Engineer](https://www.latent.space/p/ai-engineer) — the essay that named and popularized the "AI Engineer" role described above, useful context for how recent and fast-moving that title actually is.
- [Chip Huyen: MLOps Guide](https://huyenchip.com/mlops/) — includes a genuinely useful, free career and interview-prep section alongside its technical material.
- [Kaggle Competitions](https://www.kaggle.com/competitions) — where to go next for structured, scored practice against a real leaderboard.

---

**You've finished the roadmap.** All 34 lessons, start to finish — from
"what do these four words even mean" to a deployed, monitored model with
a portfolio to show for it. That's the whole thing this roadmap set out
to do, and you did it.

Nothing about this field stays still for long, so the most useful next
move isn't a 35th lesson — it's picking one of your three projects and
making it a little better, or picking one Stage that pulled at you and
going deeper. If you want to see how everything you just learned fits
together, or figure out where to send someone else who's just starting,
[the full curriculum](../CURRICULUM.md) is the map.
