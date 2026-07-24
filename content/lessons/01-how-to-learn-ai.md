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

# How to Learn AI Without Burning Out

## Why this matters

Most people who quit learning AI do not quit because it is too hard. They quit
because they opened a linear algebra textbook in week one, or booked ninety
minutes a day and lasted eleven days, or followed forty tutorials without ever
building anything of their own. Those are all avoidable. This lesson is the
plan that avoids them.

## The concept

**The maths-first trap.** It is tempting to think you need calculus and linear
algebra *before* you can touch machine learning, the way you'd need arithmetic
before algebra. You don't. Write the code first, watch it work, and only then
go back and ask why. In the previous lesson you fit a line to five points
without proving anything about gradients or loss functions — and you still
understood what "learning" meant. Formal maths sharpens intuition you already
have; it rarely creates intuition from nothing. Chase the proof only when it is
blocking something you are actively trying to build.

**How much time is realistic.** Not as much as you think, and that's fine.
Three to five focused hours a week, spread across several short sessions,
beats one heroic Saturday that leaves you too tired to touch a keyboard again
until the guilt forces you back. Consistency compounds; intensity burns out.

| Day | Time | Activity |
|---|---|---|
| Mon | 45 min | Read one lesson, take notes in your own words |
| Wed | 45 min | Redo Monday's code from memory, no copy-paste |
| Sat | 90 min | Work through that lesson's "Build this" project |
| Sun | 30 min | Review: reread your notes, no new material |

That is roughly 3.5 hours a week — enough to finish this roadmap's 33 lessons
in well under a year without any single session feeling like a chore.

**Why projects beat tutorials.** A tutorial gives you the feeling of learning
because the code runs. But code that runs because you followed twelve steps in
order teaches you to follow steps, not to build. A project — even a small one,
even the "Build this" exercise at the end of each lesson — forces you to get
stuck, look things up, and decide what to do next. Getting stuck is where the
learning actually happens.

**The three quitting points.** Nearly everyone who abandons AI study quits at
one of three moments. Around week 2–3, the *math wall*: a lesson references
gradients or matrix multiplication and it feels like you're missing a
prerequisite you'll never catch up on. Around week 5–6, *tutorial fatigue*:
you've followed enough examples that nothing feels new, but you haven't built
anything unprompted, so you plateau. Around week 9–10, the *comparison trap*:
you see someone else's polished project online and conclude you're behind,
when you're actually just further from the beginning than they are from
wherever they started. Knowing these are coming in advance robs them of most
of their power.

**Using this roadmap's prerequisite graph.** Every lesson lists its
`prerequisites` in the frontmatter — the slugs of lessons it assumes you've
done. Follow that order rather than jumping to whatever sounds most exciting.
If you can do a lesson's "Build this" step from memory, you're allowed to skip
its later repeats elsewhere in the graph; if you can't, that's a signal to
slow down, not a reason to push through.

## In code

Keep a study log. It costs two minutes a session and it is the single best
predictor of whether you'll still be doing this in month three.

```bash
mkdir -p study-log
touch study-log/week-01.md
```

```markdown
## Week 01
- Lesson: What AI, ML, Deep Learning and GenAI Actually Are
- Time spent: 1h 10m
- What I built: ran the linear regression example, changed the prices
- What confused me: why `.fit()` is called "learning"
- Next: How to Learn AI Without Burning Out
```

Writing "what confused me" every session matters more than it sounds. Confusion
you name is a question you'll eventually answer. Confusion you don't name is a
reason you quit and can't explain why.

## Build this

Create `study-log/plan.md` and write your own 12-week plan. For each week,
list which lesson you'll do (use this roadmap's stage and order to sequence
them) and which two or three days from the schedule above you'll actually use.
Be specific about days and times, not just "some time this week" — vague plans
are the ones that quietly disappear.

**Stretch:** next to weeks 2–3, 5–6, and 9–10 on your plan, write one sentence
for each describing what you'll do when that week's trap shows up. Deciding
now, while it's hypothetical, is far easier than deciding in the moment.

## Go deeper

- [fast.ai: Practical Deep Learning for Coders](https://course.fast.ai/) — the original code-first, theory-later course this lesson's philosophy borrows from.
- [Barbara Oakley: Learning How to Learn](https://www.coursera.org/learn/learning-how-to-learn) — the underlying science of spaced practice and avoiding burnout.
- [Andrej Karpathy: A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) — written about training models, but its "build up complexity incrementally" advice applies just as well to training yourself.

**Next:** [Python Basics](02-python-basics.md)
