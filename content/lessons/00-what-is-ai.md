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
