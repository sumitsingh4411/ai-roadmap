---
title: "Prompt Engineering"
stage: 5
order: 25
minutes: 45
difficulty: beginner
prerequisites: ["how-llms-work"]
tags: ["prompting", "genai"]
summary: "Clear instructions, few-shot examples, chain-of-thought, structured output, system prompts, and fixing a failing prompt in documented iterations."
---

# Prompt Engineering

## Why this matters

Last lesson's central fact was that an LLM predicts the most plausible
continuation of whatever text it's given. Prompt engineering is what
follows directly from that fact: if the model continues patterns, your job
is to put the pattern you want continued into the prompt, as unambiguously
as possible. There's no trick being played on the model here — every
technique in this lesson is a direct application of "make the intended
continuation the obviously most plausible one," and every prompting
failure you'll ever debug traces back to some part of that not being true.

## The concept

**Clear, direct instructions.** State the task, the exact format you want,
who the response is for, and any hard constraints — explicitly, not
implied. "Summarize this" and "Summarize this in exactly 3 bullet points
for a non-technical manager, no jargon" are different prompts that produce
different distributions of plausible completions; the model can't read the
intent behind a vague instruction, it can only continue the text it was
actually given. When a prompt underperforms, the single most common root
cause is an instruction that felt clear to the person writing it but left
real room for a different, still-plausible interpretation.

**Few-shot examples: show, don't just tell.** Instead of (or in addition
to) describing the pattern in prose, give the model 2–3 concrete input →
output examples of exactly what you want, ending with the real input you
want completed. This is called **few-shot prompting** (zero examples is
"zero-shot," one is "one-shot"), and it tends to be far more reliable than
prose instructions alone, for a reason that follows directly from how the
model works: the model isn't interpreting an abstract rule, it's
continuing a pattern it can see directly in its own context — a much
easier and more constrained completion than inferring the rule behind a
description. The real cost is tokens: every example you add is resent, in
full, on every single call. The "In code" section below measures that cost
for real.

**Chain-of-thought (CoT): let the model use its own output as scratch
space.** Asking the model to reason step by step before giving a final
answer measurably improves accuracy on multi-step problems (arithmetic,
logic, multi-part questions) — not because "thinking harder" is a vague
performance boost, but because of a specific mechanical reason: the
reasoning tokens the model generates become *part of the context* that the
final-answer tokens are conditioned on. The model is using its own prior
output as working memory, the same way you'd use scratch paper — a wrong
answer produced in one shot has to be right on the first try with no
intermediate steps to lean on; a wrong answer produced after visible
reasoning has a chance to be corrected by its own earlier reasoning before
the final token comes out. Some current models do a version of this
internally as "extended thinking," but explicit chain-of-thought prompting
("think through this step by step before answering") still matters: for
models without built-in extended reasoning, and any time you want a
*specific* reasoning structure rather than whatever the model would do on
its own.

**Structured output.** When you need a response in a specific machine-
readable format — JSON matching a schema, a specific set of fields, a
particular delimiter format — say so explicitly and show the exact shape
you want, ideally with an example. Asking nicely in prose ("respond in
JSON") usually works but can still drift on edge cases (extra prose before
the JSON, a slightly different key name, a markdown code fence you didn't
ask for). Production systems increasingly prefer a model provider's native
structured-output support — schema-constrained decoding that mechanically
guarantees the output validates against a JSON schema, rather than relying
on the model choosing to comply — precisely because "the model usually
follows the format" is a probability, not a guarantee, and validating a
guess is worse than not needing to guess at all.

**System prompts: a separate, standing instruction channel.** Chat-style
APIs let you set a **system prompt** — text that establishes role,
persona, tone, and standing constraints for the entire conversation,
distinct from the back-and-forth user/assistant turns. ("You are a support
agent for Acme Corp. Only answer questions about Acme products. Never
discuss competitor pricing. Keep responses under 100 words.") The system
prompt is where stable, session-wide behavior belongs, so it doesn't have
to be repeated — and doesn't compete for attention with — every individual
user message.

**Iterating on failures: treat prompting as empirical, not literary.** A
prompt that "reads well" to you is not evidence it works. The reliable
process: write a first-draft prompt, run it against several real or
representative inputs (not just the one example you had in mind), look at
*exactly* how and where it fails (Wrong format? Missing a constraint?
Hallucinating a field that wasn't in the input? Ignoring an instruction
entirely?), form a specific hypothesis about *why* — usually "the
instruction was ambiguous," "there was no example to anchor the pattern,"
or "the format request wasn't explicit enough" — make **one** targeted
change, and re-test on the same inputs before moving on. Change one thing
at a time; changing three things and re-testing tells you the combination
worked, not which change mattered. This is the same discipline lesson 30
formalizes into a real eval suite — here, you're doing it by hand, on a
handful of examples, as the entry point to that habit.

## In code

There is no API key in this lesson's environment, so nothing here executes
a real prompt against a real model — but one genuinely measurable,
runnable piece of prompt engineering *is* available offline: the token
cost of few-shot prompting, using `tiktoken` from last lesson.

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

zero_shot = """Classify the sentiment of this review as positive, negative, or neutral.

Review: "The battery life is incredible but the camera is a letdown."
Sentiment:"""

few_shot = """Classify the sentiment of each review as positive, negative, or neutral.

Review: "Fast shipping and great build quality."
Sentiment: positive

Review: "Broke after two days, total waste of money."
Sentiment: negative

Review: "It's fine. Does what it says, nothing more."
Sentiment: neutral

Review: "The battery life is incredible but the camera is a letdown."
Sentiment:"""

for name, prompt in [("zero-shot", zero_shot), ("few-shot (3 examples)", few_shot)]:
    n = len(enc.encode(prompt))
    print(f"{name}: {n} tokens")

print()
print("Few-shot costs", len(enc.encode(few_shot)) - len(enc.encode(zero_shot)),
      "more input tokens than zero-shot, on every single call.")
```

```
zero-shot: 34 tokens
few-shot (3 examples): 88 tokens

Few-shot costs 54 more input tokens than zero-shot, on every single call.
```

Nearly 2.6x the input tokens for three examples — real numbers behind the
trade-off from "The concept": few-shot is usually more reliable, but it is
not free, and the cost is paid on every request, not once.

Everything below this line calls a hosted model, which needs your own API
key — it is shown to illustrate idiomatic code, **not run**, and no
response text is shown as if it were captured output. This uses the
[Anthropic Python SDK](https://pypi.org/project/anthropic/) (`pip install
anthropic`); a current model id is `claude-sonnet-5`. A structured-output,
system-prompt example, combining several techniques from this lesson:

```python
# ILLUSTRATIVE — requires your own ANTHROPIC_API_KEY. Not executed here.
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=1024,
    system=(
        "You are a data-extraction assistant. You only output valid JSON "
        "matching the schema you are given -- no prose, no markdown fences, "
        "no extra keys. If a field cannot be determined from the input, "
        "set it to null rather than guessing."
    ),
    messages=[
        {
            "role": "user",
            "content": (
                "Schema: {\"name\": string, \"email\": string|null, \"plan\": string}\n\n"
                "Extract from: \"Hi, I'm Priya Shah, interested in the Enterprise plan. "
                "You can reach me at priya@example.com.\""
            ),
        }
    ],
)
# response.content[0].text would hold the model's reply -- not shown here,
# since we have no key to actually run this and produce a real one.
```

The system prompt fixes role and output-format constraints once, for the
whole conversation; the user turn supplies the schema and the input to
extract from — an explicit format request, per "The concept," rather than
a bare "extract the info as JSON."

## Build this

Pick a real prompting task — extracting structured fields from messy text,
classifying short pieces of text into categories, or summarizing with a
hard length limit all work well. Using your own API key, run this exact
iteration loop and **document each step in writing** as you go:

1. **Iteration 1 (zero-shot).** Write your first-draft prompt, no
   examples. Run it against 5–10 real or representative inputs. Record
   exactly how it fails — not "it didn't work," but the specific pattern
   (wrong format? missing a required field? ignored a stated constraint?).
2. **Iteration 2.** Make one targeted change aimed directly at the failure
   you just documented — usually a clearer instruction or an explicit
   format specification. Re-run the same inputs. Record what changed:
   which failures went away, which didn't, and whether anything new broke.
3. **Iteration 3.** Add 2–3 few-shot examples and/or a system prompt
   establishing role and constraints. Re-run the same inputs one more
   time. Record the result.

For each iteration, write down: the change you made, the specific failure
it targeted, and whether it worked — a real, honest log of three runs, not
a tidied-up story after the fact. Finish by running the token-cost snippet
from "In code" on your iteration-1 and iteration-3 prompts, and note the
token cost you paid for whatever reliability you gained.

## Go deeper

- [Anthropic: Prompt engineering overview](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview) — when prompt engineering is the right tool, and Claude-specific techniques.
- [Anthropic's interactive Prompt Engineering Tutorial](https://github.com/anthropics/prompt-eng-interactive-tutorial) — a free, example-filled, hands-on GitHub course covering everything in this lesson in more depth.
- [OpenAI Cookbook](https://cookbook.openai.com/) — practical, runnable recipes for prompting techniques and common failure patterns.
- [Chip Huyen: Building LLM applications for production](https://huyenchip.com/2023/04/11/llm-engineering.html) — why prompt engineering alone eventually needs the systematic practices (like evals) covered later in this stage.

**Next:** [Embeddings](26-embeddings.md)
