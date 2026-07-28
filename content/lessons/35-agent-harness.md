---
title: "Build an Agent Harness"
stage: 5
order: 35
minutes: 60
difficulty: advanced
prerequisites: ["ai-agents"]
tags: ["agents", "llm", "engineering"]
summary: "The scaffolding that turns a raw language model into an agent: the tool loop, parsing, dispatch, and history — built from scratch."
---

# Build an Agent Harness

## Why this matters

An LLM on its own can only do one thing: read some text and predict the next
tokens. It can't run code, search the web, or read a file. Yet tools like coding
assistants and autonomous agents clearly *do* those things. The trick isn't the
model — it's the **harness**: the ordinary program wrapped around the model that
reads what it wants to do, actually does it, feeds the result back, and repeats.
In [AI agents](29-ai-agents.md) you *used* an agent. Here you'll build the engine
that runs one, so you understand exactly what's happening — because once you've
seen the loop, agents stop being magic.

## The concept

A harness is four plain pieces around the model:

1. **Tools** — normal functions the agent may call (a calculator, a file reader,
   a web search). Each takes text in and returns text out.
2. **The model** — on each turn it looks at the conversation so far and decides
   one thing: *call a tool*, or *give a final answer*. A real model expresses
   that decision as generated text (often JSON); the harness parses it.
3. **The loop** — call the model, do what it asked, append the result to the
   conversation, and go again — until the model says it's done.
4. **The conversation (context)** — the running list of messages: the task, each
   tool call, and each result. This growing history is the agent's only memory.

That's it. An "agent" is a `for` loop that lets a text-predictor take actions and
see what happened. To make the loop reproducible here, our model is a **scripted
stub** — a function that returns a fixed sequence of decisions. That keeps the
focus on the harness, not the model; swapping in a real LLM is the last step.

## In code

A complete, runnable harness. The tools are real; the "LLM" is scripted so the
output is identical every time.

```python
# 1. Tools the agent can call — ordinary Python functions.
def calculator(expression):
    return str(eval(expression, {"__builtins__": {}}))

def word_count(text):
    return str(len(text.split()))

TOOLS = {"calculator": calculator, "word_count": word_count}

# 2. The "model". A real LLM reads `messages` and GENERATES its next step as
#    text; we script the sequence so the loop is 100% reproducible. It branches
#    on how many tools have run so far.
def scripted_llm(messages):
    steps_done = sum(1 for m in messages if m["role"] == "tool")
    plan = [
        {"action": "calculator", "input": "42 * 17"},
        {"action": "final", "input": "42 x 17 = 714."},
    ]
    return plan[steps_done] if steps_done < len(plan) else {"action": "final", "input": "done"}

# 3. The harness: the loop that ties tools + model + history together.
def run_agent(task, llm, max_steps=6):
    messages = [{"role": "user", "content": task}]
    for step in range(1, max_steps + 1):
        decision = llm(messages)                       # model picks the next move
        if decision["action"] == "final":              # it's done
            print(f"step {step}: FINAL -> {decision['input']}")
            return decision["input"]
        tool = TOOLS[decision["action"]]               # look up the tool
        result = tool(decision["input"])               # actually run it
        print(f"step {step}: call {decision['action']}({decision['input']!r}) -> {result}")
        messages.append({"role": "assistant", "content": f"{decision['action']}({decision['input']!r})"})
        messages.append({"role": "tool", "content": f"OBSERVATION: {result}"})  # feed result back
    return "stopped: hit the step budget"

run_agent("What is 42 * 17?", scripted_llm)
```

```
step 1: call calculator('42 * 17') -> 714
step 2: FINAL -> 42 x 17 = 714.
```

Read the loop again: the model didn't compute `714` — the `calculator` tool did.
The model only *decided to call it*, then read the `OBSERVATION` and wrote the
final answer. That division of labour — the model reasons, the tools act, the
harness carries results between them — is the whole idea.

**Swapping in a real model.** The only part that changes is `llm`. Instead of a
scripted plan, you send the messages to a real model and ask it to reply with a
tool call. This is **illustrative** (needs an API key), and modern models have
native "tool calling" that returns the decision as structured data:

```python
# ILLUSTRATIVE — needs your own API key.
import anthropic

client = anthropic.Anthropic()
def real_llm(messages):
    resp = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        tools=[{
            "name": "calculator",
            "description": "Evaluate a maths expression.",
            "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}},
        }],
        messages=messages,
    )
    # Parse resp: if the model returned a tool_use block, that's the action;
    # otherwise its text is the final answer. Same shape your harness expects.
    ...
```

Everything around it — the tools, the loop, the history, the stopping — stays
exactly the same. That's the point of a harness: the model is a swappable part.

## Build this

1. Run the harness above and confirm the output. Then add a third tool —
   `reverse(text)` — and a scripted step that uses it, and watch it flow through
   the same loop untouched.
2. Add a `system` message at the top of `messages` describing the agent's job and
   listing its tools. (A real model needs to be *told* what tools exist.)
3. Print the full `messages` list at the end. That growing transcript is the
   agent's entire memory — every decision and observation lives there.

**Stretch:** wire `llm` to your **local** model from [lesson 34](34-run-local-llm.md).
Prompt it to reply with JSON like `{"action": ..., "input": ...}`, parse it, and
you have a real agent running entirely on your own machine.

## Go deeper

- [Anthropic — Building effective agents](https://www.anthropic.com/research/building-effective-agents) — patterns from the team that builds them.
- [Anthropic — tool use docs](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — how native tool calling actually works.
- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629) — the paper behind the reason→act→observe loop.
- [Karpathy — on agents and harnesses](https://karpathy.ai/) — talks and writing on how these systems really work.

**Next:** [Loop Engineering](36-loop-engineering.md)
