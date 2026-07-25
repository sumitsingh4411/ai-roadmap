---
title: "Probability & Statistics"
stage: 2
order: 9
minutes: 50
difficulty: beginner
prerequisites: ["how-to-learn-ai"]
tags: ["math", "statistics", "probability"]
summary: "Distributions, mean and variance, conditional probability, Bayes' theorem, sampling, and what a p-value actually means."
---

# Probability & Statistics

## Why this matters

Machine learning is a discipline built on uncertainty: a model rarely says
"this email is spam," it says "this email is 92% likely to be spam." Every
evaluation metric you'll meet from here on — and every claim a data
scientist makes about whether a result is real or just noise — rests on
the ideas in this lesson. Get comfortable with them now, because
"probability the model outputs" and "probability the result is real" are
two different things beginners mix up constantly, and this lesson exists to
keep them separate.

## The concept

**A random variable is a quantity whose value comes from chance, and a
distribution describes which values are likely.** Roll a fair die and the
outcome is a random variable that can be 1 through 6, each equally likely —
a **discrete** distribution, since only whole outcomes are possible. A
person's height is a random variable too, but it can take *any* value in a
range — a **continuous** distribution. The most important continuous
distribution in this roadmap is the **normal distribution** (the "bell
curve"): symmetric, with most values clustered near the middle and fewer
the further out you go. You already saw its shape as a histogram in the
data-visualization lesson; now you'll see the numbers that define it.

**Mean, variance, and standard deviation describe where a distribution
sits and how spread out it is.** The **mean** is the long-run average value
you'd see if you sampled forever. **Variance** is the average of the
squared distance from the mean — squaring makes every distance positive
and penalises large deviations more than small ones. **Standard
deviation** is the square root of variance, which brings the units back to
match the original data (variance of a set of dollar amounts is in
"dollars squared," which nobody can picture; standard deviation is back in
dollars). For a normal distribution specifically, the **68–95–99.7 rule**
holds: about 68% of values fall within 1 standard deviation of the mean,
95% within 2, and 99.7% within 3 — a fast way to judge whether a value is
ordinary or unusual.

**Conditional probability is "given that I already know X, what's the
chance of Y?"** Written `P(Y | X)`, read "probability of Y given X." Knowing
a patient tested positive for a disease changes the probability they
actually have it, compared to the probability for a random stranger —
that's a conditional probability, and confusing it with the plain (or the
reversed) probability is one of the most common statistical errors people
make.

**Bayes' theorem tells you how to flip a conditional probability around.**
Often you know `P(positive test | has disease)` — that's how the test was
validated — but what you actually want is `P(has disease | positive test)`.
Bayes' theorem connects them: `P(A | B) = P(B | A) * P(A) / P(B)`.

The famous, counterintuitive result: even a test that's 90% accurate can
leave you *more likely than not to NOT have the disease* after a positive
result, if the disease itself is rare enough. "In code" below works this
out with real numbers — it's one of the clearest demonstrations in all of
statistics for why *base rates* (how common something is to begin with)
matter as much as test accuracy.

**Sampling: you almost never see the whole population, only a sample of
it.** The **population** is every possible data point that could exist (every
customer who could ever buy from you); a **sample** is the finite subset
you actually collected. The **law of large numbers** says that as a random
sample grows, its mean gets closer and closer to the true population mean.
This is *why* train/test splitting works at all: a random sample of your
data is assumed to represent the whole population reasonably well, and the
bigger that sample, the more that assumption holds.

**What a p-value actually is (and isn't).** Suppose you compare two groups
and see a difference. A **p-value** answers one narrow, specific question:
*if there were truly no difference between the groups* (the "null
hypothesis"), how often would random sampling alone produce a gap this big
or bigger? A small p-value (conventionally, under 0.05) means "this would
be a fairly rare coincidence if nothing real were going on" — it does
**not** mean "there's a 95% chance the effect is real," and it does not
mean "the null hypothesis is false." That misreading is so common it has a
name, and "In code" below runs an experiment specifically to make the
correct meaning concrete.

## In code

A discrete distribution — rolling a die — and computing mean, variance,
and standard deviation both by formula and with NumPy's built-ins:

```python
import numpy as np

rng = np.random.default_rng(42)

rolls = rng.integers(1, 7, size=10000)   # 1 through 6, inclusive
values, counts = np.unique(rolls, return_counts=True)
for v, c in zip(values, counts):
    print(f"face {v}: {c} times ({c/len(rolls):.3f} of rolls)")

mean = rolls.mean()
print("sample mean:", round(mean, 3), " theoretical mean:", (1+2+3+4+5+6)/6)

variance = np.mean((rolls - mean) ** 2)   # average squared distance from the mean
print("variance (manual):", round(variance, 3), " np.var:", round(rolls.var(), 3))

std = np.sqrt(variance)
print("std dev:", round(std, 3), " np.std:", round(rolls.std(), 3))
```

```
face 1: 1681 times (0.168 of rolls)
face 2: 1718 times (0.172 of rolls)
face 3: 1662 times (0.166 of rolls)
face 4: 1671 times (0.167 of rolls)
face 5: 1643 times (0.164 of rolls)
face 6: 1625 times (0.163 of rolls)
sample mean: 3.475  theoretical mean: 3.5
variance (manual): 2.905  np.var: 2.905
std dev: 1.704  np.std: 1.704
```

A continuous distribution — the normal distribution's 68–95–99.7 rule,
checked by simulation:

```python
import numpy as np

rng = np.random.default_rng(42)

scores = rng.normal(loc=70, scale=10, size=100000)   # mean 70, std 10

mean = scores.mean()
std = scores.std()
print("mean:", round(mean, 2), " std:", round(std, 2))

within_1 = np.mean(np.abs(scores - mean) <= std)
within_2 = np.mean(np.abs(scores - mean) <= 2 * std)
within_3 = np.mean(np.abs(scores - mean) <= 3 * std)
print(f"within 1 std: {within_1:.3f} (rule says ~0.68)")
print(f"within 2 std: {within_2:.3f} (rule says ~0.95)")
print(f"within 3 std: {within_3:.3f} (rule says ~0.997)")
```

```
mean: 69.96  std: 10.04
within 1 std: 0.684 (rule says ~0.68)
within 2 std: 0.954 (rule says ~0.95)
within 3 std: 0.997 (rule says ~0.997)
```

Bayes' theorem — a disease test with a low base rate, worked out both by
simulating a million people and by the formula directly:

```python
import numpy as np

rng = np.random.default_rng(42)

# 1% of the population has the disease. The test correctly flags 90% of
# people who have it. It incorrectly flags 5% of people who DON'T have it.
n = 1_000_000
has_disease = rng.random(n) < 0.01

test_positive = np.empty(n, dtype=bool)
test_positive[has_disease] = rng.random(has_disease.sum()) < 0.90        # true positives
test_positive[~has_disease] = rng.random((~has_disease).sum()) < 0.05    # false positives

simulated = has_disease[test_positive].mean()
print("simulated P(disease | positive):", round(simulated, 4))

# Bayes' theorem: P(A|B) = P(B|A) * P(A) / P(B)
p_disease = 0.01
p_positive_given_disease = 0.90
p_positive_given_healthy = 0.05
p_positive = p_positive_given_disease * p_disease + p_positive_given_healthy * (1 - p_disease)
p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive
print("Bayes' theorem   P(disease | positive):", round(p_disease_given_positive, 4))
```

```
simulated P(disease | positive): 0.1554
Bayes' theorem   P(disease | positive): 0.1538
```

A positive result on a "90% accurate" test still leaves roughly an 85%
chance the person *doesn't* have the disease — because the disease is rare
(1%) and false positives from the other 99% of the (healthy) population
pile up. This is precisely the reasoning you'll reuse in the model
evaluation lesson to explain why accuracy alone is misleading on rare
events.

Sampling and the law of large numbers:

```python
import numpy as np

rng = np.random.default_rng(42)

true_mean = 50   # the real population mean - normally you'd never know this

for sample_size in [5, 50, 500, 5000, 50000]:
    sample = rng.normal(loc=true_mean, scale=15, size=sample_size)
    print(f"n={sample_size:6d}  sample mean={sample.mean():.3f}  "
          f"error={abs(sample.mean() - true_mean):.3f}")
```

```
n=     5  sample mean=47.014  error=2.986
n=    50  sample mean=51.269  error=1.269
n=   500  sample mean=49.731  error=0.269
n=  5000  sample mean=49.742  error=0.258
n= 50000  sample mean=50.012  error=0.012
```

Bigger samples land closer to the true mean, on average — exactly the law
of large numbers, and exactly why a 60/40 train/test split on a dataset of
20 rows is far less trustworthy than the same split on 20,000.

What a p-value really answers, demonstrated with two groups drawn from the
*same* distribution — so we know for a fact there's no real difference:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

group_a = rng.normal(loc=100, scale=15, size=30)
group_b = rng.normal(loc=100, scale=15, size=30)

t_stat, p_value = stats.ttest_ind(group_a, group_b)
print("group A mean:", round(group_a.mean(), 2))
print("group B mean:", round(group_b.mean(), 2))
print("p-value:", round(p_value, 4))

# Proof by simulation: 1000 experiments where there is NO real difference
# by construction - how often does random noise alone produce p < 0.05?
false_alarms = 0
trials = 1000
for _ in range(trials):
    a = rng.normal(loc=100, scale=15, size=30)
    b = rng.normal(loc=100, scale=15, size=30)
    _, p = stats.ttest_ind(a, b)
    if p < 0.05:
        false_alarms += 1

print(f"false alarms at p<0.05: {false_alarms}/{trials} ({false_alarms/trials:.1%})")
```

```
group A mean: 100.25
group B mean: 101.71
p-value: 0.6367
false alarms at p<0.05: 43/1000 (4.3%)
```

Even though groups A and B are drawn from the *exact same* distribution —
there is truly no difference between them — about 1 in 20 of these
experiments still produces a "statistically significant" (p < 0.05) result
purely from random sampling noise. A p-value tells you how surprising your
data would be *if nothing were going on*; it never tells you the
probability that nothing is going on.

## Build this

Simulate flipping a coin 20 times (`rng.random(20) < 0.5`), count the
fraction of heads, and repeat that experiment (20 flips, count heads) 1,000
times. Plot a histogram of the 1,000 fraction-of-heads results (reuse the
histogram code from the data-visualization lesson). Describe in one or two
sentences what the histogram's shape tells you about how much a small
sample (20 flips) can vary from the true 50% probability just by chance.

**Stretch:** repeat the whole exercise with 500 flips per experiment
instead of 20, and compare the spread of the new histogram to the old one.
Connect what you see to the law of large numbers section above.

## Go deeper

- [StatQuest: Statistics Fundamentals](https://www.youtube.com/playlist?list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9) — a free playlist covering mean, variance, distributions, and hypothesis testing with the same plain-language approach as this lesson.
- [Seeing Theory: A visual introduction to probability and statistics](https://seeing-theory.brown.edu/) — interactive, visual explanations of everything in this lesson, built by Brown University.
- [3Blue1Brown: Bayes' theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM) — the visual-intuition treatment of Bayes' theorem, from the same series as the linear algebra and calculus videos.
- [Khan Academy: Statistics and probability](https://www.khanacademy.org/math/statistics-probability) — a full free course if you want more worked examples and practice problems.

**Next:** [ML Fundamentals](10-ml-fundamentals.md)
