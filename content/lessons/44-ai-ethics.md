---
title: "AI Ethics & Responsible AI"
stage: 5
order: 44
minutes: 45
difficulty: intermediate
prerequisites: ["evals-guardrails"]
tags: ["ethics", "fairness", "responsible-ai"]
summary: "Building AI that doesn't cause harm — bias and fairness, privacy, transparency, misuse, and the responsibility that comes with shipping models."
---

# AI Ethics & Responsible AI

## Why this matters

[Evals & guardrails](30-evals-guardrails.md) gave you tools to measure
whether a system does what you intended and to catch known bad inputs and
outputs at runtime. Ethics is a different question underneath that one:
even a system that passes every eval and every guardrail can still treat
people unfairly, leak private information, or get used for harm — because
none of those failures show up as "the wrong answer" in the way an eval
checks for. A model that's 95% accurate overall can still be systematically
wrong for one group of people, every time, and a clean eval score won't
tell you. This isn't a compliance checkbox tacked on after shipping — it's
an engineering concern with the same standing as correctness or latency,
because the systems in this roadmap increasingly make or influence real
decisions about real people: who gets a loan, whose resume gets seen,
whose speech gets moderated.

## The concept

**Bias and fairness.** A model trained on data learns whatever patterns
are in that data — including the historical biases baked into it. If past
hiring decisions favored one group, a model trained to predict "who gets
hired" from historical outcomes will learn to reproduce that pattern, not
correct it, because from the model's perspective it's simply the pattern
that best predicts the labels it was shown. This shows up as **disparate
impact**: a model that performs well *on average* while treating different
groups differently — different accuracy, different error rates, or
different rates of a favorable outcome across groups defined by a
sensitive attribute (race, gender, age, and so on). The uncomfortable part
is that a model doesn't need to be given the sensitive attribute directly
to learn it — a **proxy** feature (zip code, school, even word choice) can
correlate strongly enough with the sensitive attribute that dropping the
attribute itself barely moves the outcome. The runnable example below
measures exactly this, including the proxy problem, rather than treating
"bias" as an abstract concern.

**Privacy.** Models trained on real data can memorize fragments of it —
a name, an address, a passage verbatim — and reproduce them when prompted
in the right way, which turns training data into a potential leak of
whatever personal information it contained. This is why what data a model
was trained on, and whether people consented to that use, is itself an
ethical question, not just a legal one; and why lesson 30's output
guardrail — scanning generated text for PII before a user sees it — exists
as a second line of defense on top of careful training-data curation, not
a replacement for it.

**Transparency and consent.** People interacting with an AI system
generally have a reasonable expectation to know that it *is* one — a
support chat that's actually a model, a piece of writing that was
generated, a decision that was automated rather than made by a person.
Transparency also covers *how* a system was built: what data trained it,
what its known limitations are, and where its outputs can be wrong with
confidence (the same overconfidence problem from lesson 24's hallucination
discussion, now viewed as an ethical issue rather than a quality one — a
fluent wrong answer that nobody flagged as automated can mislead someone
who had no reason to double-check it).

**Misuse and dual-use.** Most capable AI systems are **dual-use**: the
same summarization model that condenses meeting notes can condense
someone's private messages without consent; the same code-generation model
that helps a developer can help write malware. Responsible deployment
means thinking through realistic misuse *before* shipping — not because
every misuse can be prevented, but because the ones you anticipated are
the only ones you get a chance to design against, through guardrails
(lesson 30), rate limits, use-case restrictions, or simply declining to
ship a capability where the realistic harm outweighs the benefit.

**Environmental cost.** Training large models consumes real, non-trivial
amounts of energy and water for cooling, and inference at scale — millions
of requests a day, forever — adds up on top of that one-time training
cost. It's a genuine cost of the field, worth weighing against a system's
actual benefit rather than assuming bigger models are free to build and
run.

**Accountability.** When a model produces a harmful or wrong output that
affects someone, "the model did it" is not an answer — a model is software
that a team built, trained, evaluated, and shipped, and that team remains
responsible for what it does in the world, the same as for any other
software they ship. Practically, this means: keeping a human in the loop
for high-stakes decisions rather than letting a model decide unsupervised,
maintaining an audit trail of what a system did and why (the same
transcript discipline from lesson 29's agent memory, now serving
accountability instead of debugging), and having a real process for
someone affected by a wrong output to contest it and get a human review.

## In code

**A concrete fairness check.** "Bias" stops being abstract the moment you
measure it. Build a small synthetic dataset with a sensitive attribute
(group A/B) where the *label* is deliberately generated to depend on group
membership on top of a legitimate feature — the way real historical data
often does — train a plain classifier on it, and look at accuracy and the
positive-prediction rate **per group**, not just overall.

```python
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=RuntimeWarning)  # benign BLAS FP-flag noise on some platforms

rng = np.random.default_rng(42)
n = 2000

# A synthetic loan-approval-style dataset. `group` is a sensitive attribute
# (A/B). `score` is a legitimate feature (like a credit score), independent
# of group. The LABEL is deliberately generated from the true group, on top
# of score -- exactly the kind of historical bias a model trained on real
# decisions can learn. `zip_code` is a PROXY: it is NOT the sensitive
# attribute, but it is correlated with it (85% of the time it agrees with
# `group`), the way a real zip code or school correlates with race or income
# without ever naming it.
group = rng.choice(["A", "B"], size=n)
score = rng.normal(loc=0, scale=1, size=n)
zip_agrees_with_group = rng.random(n) < 0.85
zip_code = np.where(zip_agrees_with_group, group, np.where(group == "A", "B", "A"))

group_bonus = np.where(group == "A", 0.0, -1.2)  # group B starts at a disadvantage
approval_logit = score + group_bonus
approval_prob = 1 / (1 + np.exp(-approval_logit))
label = rng.binomial(1, approval_prob)

group_bin = (group == "B").astype(int)      # the sensitive attribute itself
zip_bin = (zip_code == "B").astype(int)      # a proxy for it, not identical

X_with_group = np.column_stack([score, group_bin])
X_with_proxy = np.column_stack([score, zip_bin])  # sensitive attr dropped, proxy kept

(X_wg_train, X_wg_test,
 X_wp_train, X_wp_test,
 y_train, y_test,
 g_train, g_test) = train_test_split(
    X_with_group, X_with_proxy, label, group, test_size=0.3, random_state=42
)

model_with_group = LogisticRegression(solver="liblinear").fit(X_wg_train, y_train)
pred_with_group = model_with_group.predict(X_wg_test)

model_with_proxy = LogisticRegression(solver="liblinear").fit(X_wp_train, y_train)  # "mitigation"
pred_with_proxy = model_with_proxy.predict(X_wp_test)


def report(name, y_true, y_pred, groups):
    print(f"--- {name} ---")
    overall_acc = (y_pred == y_true).mean()
    print(f"  Overall accuracy: {overall_acc:.2%}")
    for g in ["A", "B"]:
        mask = groups == g
        acc = (y_pred[mask] == y_true[mask]).mean()
        positive_rate = y_pred[mask].mean()
        print(f"  Group {g}: accuracy={acc:.2%}  positive-prediction rate={positive_rate:.2%}  (n={mask.sum()})")
    gap = abs(y_pred[groups == "A"].mean() - y_pred[groups == "B"].mean())
    print(f"  Demographic-parity gap (|rate_A - rate_B|): {gap:.2%}\n")
    return gap


gap_with_group = report("Model 1: score + raw group attribute", y_test, pred_with_group, g_test)
gap_with_proxy = report("Model 2: score + zip_code proxy (group DROPPED)", y_test, pred_with_proxy, g_test)

print(
    f"Dropping the sensitive attribute took the gap from {gap_with_group:.2%} "
    f"to {gap_with_proxy:.2%} -- smaller, but still there, because zip_code "
    f"still predicts group 85% of the time. The proxy carries the bias "
    f"forward even with 'group' nowhere in the feature list."
)
```

```
--- Model 1: score + raw group attribute ---
  Overall accuracy: 70.50%
  Group A: accuracy=68.15%  positive-prediction rate=46.92%  (n=292)
  Group B: accuracy=72.73%  positive-prediction rate=15.26%  (n=308)
  Demographic-parity gap (|rate_A - rate_B|): 31.66%

--- Model 2: score + zip_code proxy (group DROPPED) ---
  Overall accuracy: 70.67%
  Group A: accuracy=68.84%  positive-prediction rate=36.64%  (n=292)
  Group B: accuracy=72.40%  positive-prediction rate=21.43%  (n=308)
  Demographic-parity gap (|rate_A - rate_B|): 15.22%

Dropping the sensitive attribute took the gap from 31.66% to 15.22% -- smaller, but still there, because zip_code still predicts group 85% of the time. The proxy carries the bias forward even with 'group' nowhere in the feature list.
```

Both models land at essentially the same ~70% overall accuracy — a metric
that alone hides the problem completely. Broken down by group, group A is
approved at roughly 3x the rate of group B in Model 1, despite `score`
being generated identically for both groups; that gap is disparate impact,
made measurable instead of anecdotal. Model 2 shows the mitigation *and*
its limit in one run: removing `group` as a feature roughly halves the
gap, but doesn't close it, because `zip_code` still carries 85% of the
information `group` did. This is the proxy problem from "The concept" —
not a theoretical caveat, but a number you just watched survive a
real fix attempt.

## Build this

Run the code above and confirm your own numbers roughly match (exact
values will differ slightly by machine/library version, but the pattern —
a large gap that shrinks but doesn't vanish — should hold). Then:

1. Measure the gap yourself: write down Model 1's and Model 2's
   demographic-parity gap from your run, and confirm Model 2's is smaller
   but still clearly non-zero.
2. Try a second mitigation: change `zip_agrees_with_group`'s threshold
   from `0.85` to `0.55` (a much weaker proxy, closer to random) and
   re-run. Confirm the gap after dropping `group` shrinks much further
   this time — because a weak proxy carries less information about group
   membership than a strong one. This is the practical lesson: mitigation
   effectiveness depends entirely on *how strong the remaining proxies
   are*, which you often can't know without measuring it, the way you just
   did.

**Stretch:** compute the **demographic-parity difference** as its own
named function — `dp_difference(y_pred, groups) = P(pred=1 | group=A) -
P(pred=1 | group=B)` (signed, not absolute, so you can see *which* group
is favored) — and print it for both models. Look up one other fairness
metric (equal opportunity or equalized odds are common next steps) and
note in a comment why it can disagree with demographic parity — they
measure different things and a model can satisfy one while failing the
other.

## Go deeper

- [Google: Machine Learning Fairness crash course](https://developers.google.com/machine-learning/crash-course/fairness) — a free, practical introduction to fairness metrics and mitigation trade-offs.
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) — a widely-referenced, free framework for identifying and managing AI risk end to end.
- [Fairlearn](https://fairlearn.org/) — an open-source Python toolkit for measuring and mitigating fairness issues, including the demographic-parity and equalized-odds metrics referenced above.
- [Anthropic: Core Views on AI Safety](https://www.anthropic.com/news/core-views-on-ai-safety) — how one AI lab reasons publicly about risk, misuse, and responsible deployment.
- [Model Cards for Model Reporting (Mitchell et al.)](https://arxiv.org/abs/1810.03993) — the paper that introduced model cards, a concrete transparency practice for documenting a model's data, limitations, and intended use.

**Next:** [MLOps Basics](31-mlops-basics.md)
