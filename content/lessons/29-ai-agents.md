---
title: "AI Agents"
stage: 5
order: 29
minutes: 60
difficulty: advanced
prerequisites: ["rag"]
tags: ["agents", "tool-use"]
summary: "Tool use, the reason-act loop, planning, memory, multi-step failure modes, and cost control, with a real non-LLM demo of the loop mechanics."
---

# AI Agents

## Why this matters

RAG (lesson 27) gave the model access to information through a single
retrieve-then-generate step per question. An **agent** goes one step
further: instead of one retrieval, the model can decide, across multiple
steps, which action to take next, observe what happens, and decide again —
reading a file, calling an API, running a calculation, searching, asking a
follow-up — chaining these until it has enough to answer. This is the
least "prompt-shaped" lesson in the roadmap: agents fail in ways prompting
alone doesn't, and every step costs real money and real time, so this
lesson treats both honestly rather than glossing over them.

## The concept

**Tool use: the mechanism underneath every agent.** You describe to the
model a set of **tools** — each with a name, a description, and a schema
for its arguments. On any turn, instead of only returning text, the model
can return a structured request to call one of those tools with specific
arguments. Critically, **the model has no side effects of its own** — it
only ever produces text or a structured tool-call request. *Your* code
executes the actual action (querying a database, hitting an API, reading a
file, running a calculator) and returns the result back to the model as an
**observation**, which becomes part of the context the model sees on its
next step. Every action an "agent" ever takes is code you wrote and
control, triggered by a request the model made; the model never reaches
outside the conversation on its own.

**The reason-act loop.** The standard shape, often called **ReAct**
(reason + act), after the paper that named it: **Thought** — the model
reasons, given the question and everything observed so far, about what to
do next. **Action** — the model requests a tool call. **Observation** —
your code runs the tool and returns the result. Repeat, feeding each new
observation back into context, until the model decides it has enough
information and returns a final text answer instead of another tool
request. This loop — and specifically the part where the output of one
tool call becomes the input to deciding the next one — is exactly what
"In code" demonstrates mechanically below.

**Planning.** For a handful of tools and a simple question, a workable
plan can emerge naturally, one step of the loop at a time. For more
complex, multi-step tasks, agents benefit from an explicit up-front plan —
"first look up X, then compute Y using X, then check Z" — either produced
by the model itself before it starts acting, or scaffolded by the
developer. Planning cuts down on wasted or wrong tool calls, but it's a
double-edged sword: a plan made before gathering any real information can
lock the agent into a bad sequence of steps once the world doesn't match
the plan's assumptions, which means a working agent also needs a way to
notice that and replan, not just execute the original plan regardless.

**Memory.** Within a single agent run, "memory" is nothing more than: the
full transcript of every thought, action, and observation so far, resent
as context on every step. This is exactly why the context window (lesson
24) matters directly here — a long-running agent can accumulate enough
transcript to exceed it, forcing some form of summarization or compaction
of older steps. Across separate runs, an agent has **no memory by
default** — the underlying API call is stateless, same as every LLM call
in this roadmap — unless you deliberately build persistence: writing notes
to a file or database that a future run reads back in.

**Multi-step failures.** Be honest about this: agents compound errors.
If step 3 of an 8-step run misreads a tool's result, every step built on
top of step 3 inherits that mistake — and the model's tone stays exactly
as fluent and confident as if step 3 had been correct, because nothing
about *how* the model produces text changes when its premises are wrong
(the same mechanism as hallucination in lesson 24, now compounding across
steps instead of appearing in one answer). Known, common failure patterns:
**looping** — repeating a failed action expecting a different result,
because the model has no built-in "notice I'm stuck" signal unless you add
a step limit or a duplicate-action check; **tool misuse** — calling a tool
with malformed or fabricated arguments; **over-planning** — endless
reasoning without ever actually calling a tool; **premature stopping** —
answering before gathering information the task actually needed; and
**losing the original goal** over a long transcript, drifting toward
answering a nearby-but-different question.

**Cost control.** Every step of the loop is a full LLM call, and a
multi-step agent can silently run 5, 20, or 100+ calls to satisfy one user
request — cost and latency scale directly with the number of steps, not
with how "hard" the task felt. Practical controls: a hard limit on the
number of iterations, an explicit token or cost budget per run, keeping
the tool set restricted to only what the task actually needs (fewer tools
means a shorter tool-schema overhead on every single call, and a smaller
decision space means fewer wrong calls), and reserving your most capable
— and most expensive — model for the steps that actually need it rather
than every step uniformly.

## In code

A real, runnable demonstration of the reason-act loop's *mechanics*, using
two tools and a question that genuinely requires both, in sequence — the
second tool's input depends on the first tool's output:

```python
FACTS = {
    "population of france": "68",
    "capital of japan": "Tokyo",
    "boiling point of water in celsius": "100",
}

def lookup(query):
    return FACTS.get(query.lower().strip(), "not found")

def calculator(expression):
    return eval(expression, {"__builtins__": {}})

TOOLS = {"lookup": lookup, "calculator": calculator}

def run_agent(question, plan):
    """`plan` is a scripted (thought, tool, tool_input) sequence standing in
    for what a real LLM would decide at each turn. This lets us show the
    reason -> act -> observe LOOP MECHANICS for real, without a hosted
    model driving it -- see the illustrative Anthropic tool-use code below
    for how a real model fills this role instead of a hardcoded plan."""
    print(f"Question: {question}\n")
    observation = None
    for step, (thought, tool_name, make_input) in enumerate(plan, start=1):
        tool_input = make_input(observation)
        print(f"Step {step}")
        print(f"  Thought:     {thought}")
        print(f"  Action:      {tool_name}({tool_input!r})")
        observation = TOOLS[tool_name](tool_input)
        print(f"  Observation: {observation}\n")
    print(f"Final answer: {observation}")

plan = [
    ("I need the population of France before I can compute anything.",
     "lookup", lambda obs: "population of france"),
    ("Now double that number.",
     "calculator", lambda obs: f"{obs} * 2"),
]
run_agent("What is double the population of France (in millions)?", plan)
```

```
Question: What is double the population of France (in millions)?

Step 1
  Thought:     I need the population of France before I can compute anything.
  Action:      lookup('population of france')
  Observation: 68

Step 2
  Thought:     Now double that number.
  Action:      calculator('68 * 2')
  Observation: 136

Final answer: 136
```

This is real code that really ran — but the "Thought" and which tool to
call next were scripted by us, not decided by a model. It shows the loop's
*shape* honestly: two tools, genuinely chained (step 2's input, `68`, only
exists because step 1 ran first), an observation feeding back into the
next decision. What it can't show, without an API key, is a model actually
*choosing* that sequence on its own. Real tool-calling code, illustrating
the idiomatic shape for the same two tools, needs your own key — shown
here, **not run**, no model output shown as if it were captured. Using
the [Anthropic Python SDK](https://pypi.org/project/anthropic/):

```python
# ILLUSTRATIVE -- requires your own ANTHROPIC_API_KEY. Not executed here.
import anthropic

client = anthropic.Anthropic()

tools = [
    {
        "name": "lookup",
        "description": "Look up a known fact by its exact key, e.g. 'population of france'.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate a simple arithmetic expression, e.g. '68 * 2'.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
]

messages = [{"role": "user", "content": "What is double the population of France (in millions)?"}]

# The reason-act loop: keep calling the model and feeding tool results back
# until it stops requesting tools and returns a final text answer.
for _ in range(5):  # hard step limit -- see "Cost control" above
    response = client.messages.create(
        model="claude-sonnet-5", max_tokens=1024, tools=tools, messages=messages,
    )
    tool_uses = [b for b in response.content if b.type == "tool_use"]
    if not tool_uses:
        break  # model returned a final text answer, not another tool call
    messages.append({"role": "assistant", "content": response.content})
    tool_results = []
    for call in tool_uses:
        result = TOOLS[call.name](**call.input)   # your code runs the real tool
        tool_results.append({"type": "tool_result", "tool_use_id": call.id, "content": str(result)})
    messages.append({"role": "user", "content": tool_results})
# response.content would now hold the final answer -- not shown, since we
# have no key here to actually run this loop and produce a real one.
```

The loop's control flow — request, check for `tool_use` blocks, execute,
send `tool_result` back, repeat until the model stops asking for tools —
is the real ReAct loop from "The concept," implemented against a real API,
using the exact two tools the mock version above ran by hand.

## Build this

Build an agent with two tools that answers a question requiring both. You
can reuse `lookup` and `calculator` from "In code," or invent your own
(a unit converter and a small local-file search work well). Write the
real tool functions, adapt the illustrative loop above with your own
Anthropic (or other provider's) API key, and ask a question that genuinely
needs both tools in sequence, the way the population-doubling example
does. Run it for real and log the actual thought/action/observation trace
the model produces — then compare it, step by step, to what the scripted
mock version in "In code" did by hand. Note anything that surprised you:
a tool call you didn't expect, an extra step, a different order.

**Stretch:** add a hard step limit (say, 5) to your loop, and deliberately
ask a question that needs more reasoning than your two tools can actually
satisfy. Confirm your code does something sane when the limit is hit —
returns a clear "couldn't finish" state — rather than looping forever
silently or crashing.

## Go deeper

- [Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — a widely-cited, practical guide to when to use an agent at all, and simple patterns that outperform complex agent frameworks.
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models"](https://arxiv.org/abs/2210.03629) — the paper that named and formalized the reason-act loop this lesson builds.
- [Anthropic: Tool use overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview) — the full reference for defining tools and handling tool calls with a real API key.
- [Anthropic: Tutorial — Build a tool-using agent](https://platform.claude.com/docs/en/agents-and-tools/tool-use/build-a-tool-using-agent) — a guided walkthrough from one tool call to a full agentic loop, in the same shape as this lesson's illustrative code.

**Next:** [Evals & Guardrails](30-evals-guardrails.md)
