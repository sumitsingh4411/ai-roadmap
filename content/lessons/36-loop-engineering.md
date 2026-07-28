---
title: "Loop Engineering"
stage: 5
order: 36
minutes: 55
difficulty: advanced
prerequisites: ["agent-harness"]
tags: ["agents", "engineering", "reliability"]
summary: "The naive agent loop breaks in a dozen ways. The engineering that makes it reliable: budgets, loop detection, context control, and error recovery."
---

# Loop Engineering

## Why this matters

The [harness](35-agent-harness.md) you built works — until a real model drives
it. Then it fails in ways a demo never shows: the agent repeats the same broken
action forever, a tool throws and the whole run dies, the conversation grows past
the context window, or it quietly burns through your budget doing nothing useful.
The gap between "an agent that works in a video" and "an agent you'd trust to run
unattended" is almost entirely **loop engineering** — the guards, budgets, and
context management wrapped around the reason→act→observe cycle. This is the
unglamorous work that makes agents actually reliable.

## The concept

Six problems, six guards. Each is a few lines, and each prevents a real failure:

- **Step budget.** Never loop forever — cap the number of turns. If the agent
  hasn't finished in *N* steps, stop and report, don't spin.
- **Loop / no-progress detection.** If the model asks for the *exact same action*
  it already tried, it's stuck. Detect the repeat and break out — otherwise it
  will try `1/0` a hundred times.
- **Error recovery.** A tool *will* throw. Catch it, turn the exception into an
  `OBSERVATION: ERROR ...` the model can read, and let it try something else —
  one failing tool shouldn't kill the run.
- **Context control.** The message history grows every step and the context
  window is finite. Keep the task plus the most recent turns; drop or summarise
  the stale middle so you never overflow (and don't pay to re-send everything).
- **Cost / runaway guard.** Independently cap total tool calls (or tokens, or
  dollars). The step budget bounds turns; this bounds work.
- **A clear done signal.** The model must have an unambiguous way to say "I'm
  finished" — and the loop must honour it immediately.

None of this is AI. It's the same defensive engineering you'd put around any
loop that calls unpredictable code — which is exactly what an agent is.

## In code

The harness from last lesson, hardened. Same shape, now with a step budget, a
no-progress guard, error recovery, a cost cap, and history trimming:

```python
# (reusing the tools from the previous lesson so this runs on its own)
def calculator(expression):
    return str(eval(expression, {"__builtins__": {}}))

TOOLS = {"calculator": calculator}

def run_agent_robust(task, llm, max_steps=8, max_tool_calls=20):
    messages = [{"role": "user", "content": task}]
    seen = set()
    tool_calls = 0
    for step in range(1, max_steps + 1):                 # STEP BUDGET
        decision = llm(messages)
        if decision["action"] == "final":                # DONE SIGNAL
            print(f"step {step}: FINAL -> {decision['input']}")
            return decision["input"]

        key = (decision["action"], decision["input"])
        if key in seen:                                  # NO-PROGRESS GUARD
            print(f"step {step}: loop detected ({key}) -> stopping")
            return "stopped: repeated the same action with no progress"
        seen.add(key)

        if tool_calls >= max_tool_calls:                 # COST GUARD
            return "stopped: tool-call budget exhausted"

        try:
            result = TOOLS[decision["action"]](decision["input"])
        except Exception as e:                           # ERROR RECOVERY
            result = f"ERROR: {e}"
        tool_calls += 1
        print(f"step {step}: call {decision['action']}({decision['input']!r}) -> {result}")
        messages.append({"role": "tool", "content": f"OBSERVATION: {result}"})

        if len(messages) > 12:                           # CONTEXT CONTROL
            messages = [messages[0]] + messages[-8:]     # keep task + recent turns

    return "stopped: hit the step budget"

# A model that keeps trying the same broken thing — the guards must catch it.
def stuck_llm(messages):
    return {"action": "calculator", "input": "1/0"}

run_agent_robust("Compute something", stuck_llm)
```

```
step 1: call calculator('1/0') -> ERROR: division by zero
step 2: loop detected (('calculator', '1/0')) -> stopping
```

That output is the whole lesson. A naive loop would have called `1/0` until it
hit the step budget, throwing on the first try. The hardened loop **caught the
error** (turning it into an observation the model could learn from) and, when the
model stubbornly repeated the identical failing call, **detected the loop and
stopped** — in two steps, with a clear reason, instead of spinning. This is what
"reliable" looks like up close: not cleverness, just guards.

## Build this

1. Run the hardened harness and confirm it stops in two steps. Then give
   `stuck_llm` a *second* distinct action and watch the no-progress guard let the
   new one through but still catch a later repeat.
2. Add a running **cost estimate**: assume each step costs a fixed number of
   "tokens", accumulate it, print the total at the end, and stop early if it
   exceeds a budget you set.
3. Replace the crude history trim with a **summary**: when `messages` gets long,
   collapse the oldest turns into a single `"summary so far: ..."` message
   instead of dropping them.

**Stretch:** add a **sub-agent** — a tool whose implementation is *itself* a call
to `run_agent_robust` with its own budget. Now your agent can delegate a piece of
work to a fresh loop with a clean context, then use the result. That one idea —
loops that spawn loops — is how the most capable agent systems are built.

## Go deeper

- [Anthropic — Building effective agents](https://www.anthropic.com/research/building-effective-agents) — when to add complexity, and when not to.
- [Context engineering for agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — managing the conversation the loop feeds the model.
- [Reflexion](https://arxiv.org/abs/2303.11366) — letting agents learn from their own failed attempts within the loop.
- [12-Factor Agents](https://github.com/humanlayer/12-factor-agents) — practical principles for production-grade agent loops.

**Next:** [MLOps Basics](31-mlops-basics.md)
