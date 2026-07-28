---
title: "Structured Outputs & Function Calling"
stage: 5
order: 42
minutes: 45
difficulty: intermediate
prerequisites: ["prompt-engineering"]
tags: ["llm", "json", "function-calling"]
summary: "Getting reliable JSON out of an LLM — schemas, validation, function/tool calling, and what to do when the model returns something wrong."
---

# Structured Outputs & Function Calling

## Why this matters

Everything lesson 25 covered — clear instructions, few-shot examples,
system prompts — was in service of getting *better prose* out of a model.
The moment that output needs to flow into another piece of software
instead of being read by a person, "better prose" isn't good enough: code
that does `ticket["priority"]` needs a real key called `priority` to
exist, every single time, not "usually, if the model felt like it." This
lesson is about closing that gap — from a model that's probably going to
format things the way you asked, to a program that only ever sees data it
can actually trust.

## The concept

**Free-form text is unusable for software, directly.** An LLM's raw output
is a string. A human reading "the customer's priority is high" understands
it instantly; a line of code expecting `ticket.priority == "high"` does
not — it needs that fact to arrive in an exact, predictable shape, not
buried in a sentence that might also say "high priority" or "this is
urgent" or "priority: HIGH" depending on the day. The fix is making the
model's output *itself* machine-readable — almost always JSON — so a
program can parse it directly instead of trying to pattern-match prose.

**Three ways to get there, in increasing order of reliability.**

1. **Ask for JSON in the prompt.** Lesson 25's structured-output
   technique: describe the exact shape you want, ideally with an example,
   and ask the model to output only that. It usually works. "Usually" is
   the whole problem — the model can still wrap the JSON in a markdown
   code fence you didn't ask for, add a sentence of preamble before it, or
   rename a key it decided sounded better, because nothing is actually
   *enforcing* the shape — you're relying on the model choosing to comply.
2. **Native structured output / JSON mode.** Model providers increasingly
   offer an API parameter that constrains *decoding itself* — the model is
   mechanically prevented from generating a token that would produce
   invalid JSON or a value that doesn't match your schema, rather than
   being asked nicely and hoping. This is a guarantee about the output's
   *shape*, not about whether the content is correct.
3. **Function calling / tool calling.** Instead of asking the model to
   write JSON as text, you describe one or more **tools** — named
   operations with a schema for their arguments — and the model, when it
   decides a tool is the right next step, returns a structured "call this
   tool with these arguments" object instead of prose. Lesson 29 used this
   to let a model *choose and invoke* real functions; this lesson uses the
   exact same mechanism for a narrower purpose — treating "extract this
   data" itself as the tool call, purely to get a validated arguments
   object back, whether or not anything is actually invoked.

**Defining the shape with a schema.** All three approaches need the shape
spelled out somewhere, formally. **JSON Schema** is the standard,
language-independent way to describe it: field names, types, which fields
are required, allowed values for a field, numeric ranges. Writing raw JSON
Schema by hand is verbose, so in Python the standard shortcut is
**Pydantic**: define a normal-looking class with typed fields, and
Pydantic gives you two things from that one definition — a `.model_validate()`
method that checks a piece of data against it, and (via
`.model_json_schema()`) the equivalent JSON Schema, ready to hand to an
API that wants one. "In code" below uses Pydantic for exactly that reason:
one definition, both jobs.

**The essential engineering discipline: validate, then retry on
failure.** Even native structured output and tool calling only guarantee
the *shape* comes back correctly formed — they say nothing about whether
a value is sensible (an email field with no `@` in it, a priority of `9`
when only `1`–`5` are valid). And plain prompted JSON (option 1 above)
doesn't even guarantee the shape. The professional pattern, regardless of
which of the three approaches produced the output, is the same: **parse
it, validate it against your schema, and if it fails, catch that failure
explicitly** — either return a clear error instead of silently passing bad
data downstream, or send the error back to the model and ask it to correct
itself, then try again. A model returning something wrong on a given call
isn't rare; treating "the output might not validate" as the normal case
worth coding for, not an edge case worth ignoring, is what separates a
demo from something you'd trust in production.

## In code

A Pydantic model for a support-ticket extraction task, and a
`parse_or_retry`-style function that turns raw model output into either a
validated object or a clear reason it was rejected. Nothing here calls a
model — this is the validation layer that has to exist regardless of
*how* the JSON was produced, run against two literal strings standing in
for two different real outputs a model could hand back:

```python
import json
from pydantic import BaseModel, ValidationError

class SupportTicket(BaseModel):
    customer_name: str
    email: str
    plan: str
    priority: int

def parse_or_retry(raw_text, schema=SupportTicket):
    """Turn raw model output into a validated schema instance.
    Returns (instance, None) on success, or (None, error_message) on failure."""
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return None, f"invalid JSON syntax: {e}"
    try:
        return schema.model_validate(data), None
    except ValidationError as e:
        return None, f"schema validation failed: {e.errors()[0]['msg']}"

good_output = '{"customer_name": "Priya Shah", "email": "priya@example.com", "plan": "pro", "priority": 2}'

# A realistic LLM failure mode: output got cut off before the closing brace.
bad_output = '{"customer_name": "Alex Kim", "email": "alex@example.com", "plan": "pro", "priority": 2'

for label, raw in [("good_output", good_output), ("bad_output", bad_output)]:
    result, error = parse_or_retry(raw)
    if result is not None:
        print(f"{label}: ACCEPTED -> {result}")
    else:
        print(f"{label}: REJECTED -> {error}")
```

```
good_output: ACCEPTED -> customer_name='Priya Shah' email='priya@example.com' plan='pro' priority=2
bad_output: REJECTED -> invalid JSON syntax: Expecting ',' delimiter: line 1 column 88 (char 87)
```

`good_output` comes back as a real `SupportTicket` instance — typed,
attribute-accessible, safe to pass to the rest of your program.
`bad_output` is rejected with a specific, actionable reason instead of
either crashing the program or silently handing it a broken string. That
`error` message is exactly what you'd feed back to the model on a retry:
"your last response was invalid JSON — reason: `{error}` — please
resend."

That `.model_json_schema()` method mentioned in "The concept" turns the
same class into the schema a tool-calling API wants, with no separate
definition to keep in sync:

```python
import json as json_lib
print(json_lib.dumps(SupportTicket.model_json_schema(), indent=2))
```

```
{
  "properties": {
    "customer_name": {
      "title": "Customer Name",
      "type": "string"
    },
    "email": {
      "title": "Email",
      "type": "string"
    },
    "plan": {
      "title": "Plan",
      "type": "string"
    },
    "priority": {
      "title": "Priority",
      "type": "integer"
    }
  },
  "required": [
    "customer_name",
    "email",
    "plan",
    "priority"
  ],
  "title": "SupportTicket",
  "type": "object"
}
```

Sending that schema to a real model as a tool definition needs your own
API key — shown here to illustrate the idiomatic shape, **not run**, no
response text shown as if it were captured. Using the [Anthropic Python
SDK](https://pypi.org/project/anthropic/), the same reason for the pattern
as lesson 29's tool-calling loop, now used purely for extraction:

```python
# ILLUSTRATIVE -- requires your own ANTHROPIC_API_KEY. Not executed here.
import anthropic

client = anthropic.Anthropic()

tools = [
    {
        "name": "record_support_ticket",
        "description": "Record the structured details of a support ticket.",
        "input_schema": SupportTicket.model_json_schema(),
    }
]

response = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=1024,
    tools=tools,
    tool_choice={"type": "tool", "name": "record_support_ticket"},
    messages=[{
        "role": "user",
        "content": (
            "Hi, I'm Priya Shah, on the Pro plan, priority 2. "
            "You can reach me at priya@example.com."
        ),
    }],
)

tool_use = next(b for b in response.content if b.type == "tool_use")
# tool_use.input is already a parsed dict matching the schema -- validate it
# the same way as any other candidate output, since the API guarantees the
# *shape* came through, not that every value is sensible:
ticket, error = parse_or_retry(json.dumps(tool_use.input))
# ticket would now hold a validated SupportTicket -- not shown, since we
# have no key here to actually run this and produce a real one.
```

`tool_choice` forces the model to call this specific tool rather than
choosing freely, which is the right move when extraction — not open-ended
tool selection like lesson 29 — is the entire point of the call.
`tool_use.input` arrives already parsed into a Python dict (the API did
the JSON-syntax work), but it still gets passed through `parse_or_retry` —
native tool calling guarantees valid JSON, not a valid `priority`, so the
validation step from earlier in this section still earns its keep.

## Build this

Take the `SupportTicket` model above and add two constrained fields:
a `plan` restricted to a fixed set of values (`from typing import
Literal`, then `plan: Literal["starter", "pro", "enterprise"]`) and a
`priority` restricted to a sensible range (`from pydantic import Field`,
then `priority: int = Field(ge=1, le=5)`). Write two new candidate output
strings — one with `"plan": "ultra"` (not in the allowed set) and one with
`"priority": 9` (out of range) — run them through `parse_or_retry`, and
confirm each is rejected with a specific, readable reason rather than
either crashing or silently passing through with a nonsense value.

**Stretch:** write a small retry loop around `parse_or_retry`: given a
list of candidate raw-output strings (simulating a model's first attempt,
then a corrected second attempt after being told what was wrong), call
`parse_or_retry` on each in order, stop at the first one that validates,
and print which attempt succeeded and why the earlier one(s) failed. That
loop — try, catch the specific validation failure, feed it back, try again
— is the exact shape a real retry-on-invalid-output system uses in
production, just with your list standing in for the model's next
response.

## Go deeper

- [Anthropic: Tool use overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview) — the official reference for tool/function calling, including forcing a specific tool with `tool_choice`.
- [Pydantic documentation](https://docs.pydantic.dev/latest/) — the validation library used in this lesson, including `model_json_schema()` and all built-in constraint types.
- [JSON Schema](https://json-schema.org/learn/getting-started-step-by-step) — the underlying schema format every structured-output and tool-calling API is built on.
- [OpenAI: Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) — a second provider's take on the same "guarantee the shape" idea, useful for seeing what's shared across providers versus specific to one.

**Next:** [Model Context Protocol (MCP)](43-model-context-protocol.md)
</content>
