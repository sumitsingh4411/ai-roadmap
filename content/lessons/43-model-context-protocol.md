---
title: "Model Context Protocol (MCP)"
stage: 5
order: 43
minutes: 45
difficulty: advanced
prerequisites: ["agent-harness"]
tags: ["mcp", "agents", "tools"]
summary: "The USB-C of AI tools — a standard protocol that lets any agent connect to any tool or data source without custom glue for each one."
---

# Model Context Protocol (MCP)

## Why this matters

In [lesson 35](35-agent-harness.md) you built a harness by hand: a `TOOLS`
dict, a loop that dispatches on a tool name, and a model that decides which
one to call. That works great for one agent with a handful of tools you
wrote yourself. Now imagine you maintain ten different agents — an IDE
assistant, a Slack bot, a desktop app — and you want all of them to read
your calendar, query your database, and search your files. Without a
shared standard, that's N agents times M tools worth of bespoke
integration code, and every new tool means writing another adapter for
every agent that wants it. **MCP (Model Context Protocol)** exists to
collapse that N×M problem into N+M: build a tool once as an MCP server,
and any MCP-compatible agent can use it with zero custom glue.

## The concept

**The problem, precisely.** Before a shared protocol, "give my agent
access to GitHub" meant writing GitHub-specific code inside *that specific
agent's* codebase — parsing its API, matching your harness's tool-call
shape, handling its auth. Do that for GitHub, Slack, Postgres, and your
local filesystem, across three different agent products, and you've
written twelve integrations instead of four tools and three agents. Every
pair needs its own glue code.

**MCP is an open standard** for the interface between an agent and the
tools/data it uses — not a specific tool, model, or company's product.
It defines a **client–server protocol**: an **MCP client** (an IDE, Claude
Desktop, or your own agent harness) connects to an **MCP server** (a
process exposing some capability — GitHub, a database, your filesystem)
and speaks a shared, standardized message format to it. The server author
never needs to know which client will connect; the client author never
needs to know which server it'll talk to, beyond the protocol itself. That
is exactly what turns N×M integrations into N servers plus M clients.

**Three things a server can expose:**

- **Tools** — actions the client can invoke, each with a name, a
  description, and an input schema. This is the same idea as the `TOOLS`
  dict from lesson 35's harness, just described in a standard, discoverable
  shape instead of hardcoded into one program.
- **Resources** — data the client can read (a file's contents, a database
  row, a document) without it counting as an "action" with side effects.
- **Prompts** — reusable, parameterized prompt templates the server offers,
  so common tasks against that server don't need to be reinvented by every
  client that connects to it.

**How this relates to your harness.** Lesson 35's loop still applies
unchanged: the model decides to call a tool, your code executes it, the
result comes back as an observation, repeat. MCP doesn't replace that loop
— it standardizes *one piece* of it: how the harness discovers what tools
exist and how it calls them. Instead of you hand-writing a `TOOLS` dict
for every integration, an MCP client asks a server "what tools do you
have?" and gets back names, descriptions, and schemas it can hand straight
to the model — the same information your harness needed, just fetched over
a standard protocol instead of typed in by hand.

**The request/response shape.** Two operations carry most of the protocol:

1. **List tools** — the client asks the server "what can you do?" The
   server replies with a list of tool names, descriptions, and input
   schemas.
2. **Call a tool** — the client sends a tool name plus arguments; the
   server actually runs it and replies with a result (or an error).

Real MCP runs this over JSON-RPC, typically across stdio (a local
subprocess) or HTTP (a remote server). The mock below skips the transport
entirely and implements only the *shape* — two request types and how a
server answers them — in plain Python function calls, so you can see the
protocol's logic with nothing else in the way.

## In code

A tiny mock MCP-style server and client. No network, no external packages
— just the request/response shape MCP formalizes.

```python
# A tiny mock MCP-style server and client, in pure Python. No network, no deps.

class MockMCPServer:
    """Stands in for a real MCP server: it registers tools and answers
    two request types, `list_tools` and `call_tool` -- the same two
    operations a real MCP client uses against any real MCP server."""

    def __init__(self, name):
        self.name = name
        self.tools = {}  # tool name -> (description, function)

    def register_tool(self, name, description, func):
        self.tools[name] = (description, func)

    def handle_request(self, request):
        """A single entry point that dispatches on `request["type"]`,
        exactly like a real MCP server dispatching JSON-RPC requests."""
        if request["type"] == "list_tools":
            return {
                "tools": [
                    {"name": name, "description": desc}
                    for name, (desc, _fn) in self.tools.items()
                ]
            }
        if request["type"] == "call_tool":
            name = request["name"]
            if name not in self.tools:
                return {"error": f"unknown tool {name!r}"}
            _desc, fn = self.tools[name]
            result = fn(**request["arguments"])
            return {"result": result}
        return {"error": f"unknown request type {request['type']!r}"}


class MockMCPClient:
    """Stands in for an MCP client (an IDE, Claude Desktop, or an agent
    harness like lesson 35's) talking to a server over the protocol."""

    def __init__(self, server):
        self.server = server  # a real client would hold a connection, not the object

    def list_tools(self):
        return self.server.handle_request({"type": "list_tools"})["tools"]

    def call_tool(self, name, **arguments):
        response = self.server.handle_request(
            {"type": "call_tool", "name": name, "arguments": arguments}
        )
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["result"]


# --- Set up a server with one tool and drive it from a client ---

server = MockMCPServer("demo-server")
server.register_tool(
    "add",
    "Add two numbers.",
    lambda a, b: a + b,
)

client = MockMCPClient(server)

print("Tools available:")
for tool in client.list_tools():
    print(f"  - {tool['name']}: {tool['description']}")

result = client.call_tool("add", a=4, b=5)
print(f"\ncall_tool('add', a=4, b=5) -> {result}")
```

```
Tools available:
  - add: Add two numbers.

call_tool('add', a=4, b=5) -> 9
```

Read that against lesson 35's harness: `list_tools` is the discoverable
version of the `TOOLS` dict, and `call_tool` is the dispatch step (`tool =
TOOLS[decision["action"]]; result = tool(decision["input"])`), just phrased
as a request/response pair instead of a Python lookup. A real agent's loop
would call `client.list_tools()` once to build the tool-schema list it
hands the model, then call `client.call_tool(...)` every time the model
decides to act — the rest of the loop from lesson 35 is unchanged.

**A real MCP server.** The official Python SDK (`pip install mcp`) gives
you the real protocol — JSON-RPC over stdio or HTTP, a `FastMCP` server
class, and clients built into products like Claude Desktop and various
IDEs. Shown here for shape only; it needs the package installed and a
client to connect to it, so it is **not run**:

```python
# ILLUSTRATIVE -- needs `pip install mcp` and a real MCP client to connect.
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

if __name__ == "__main__":
    mcp.run()  # starts listening for a client, e.g. Claude Desktop
```

Compare it to the mock above: `@mcp.tool()` is doing the same job as
`server.register_tool(...)` — registering a Python function, its name, and
its description so a client can discover and call it — just with the real
JSON-RPC transport and schema generation handled for you instead of a
plain dict.

## Build this

1. Add a second tool to the mock server — `def multiply(a, b): return a *
   b` — register it with `server.register_tool`, then call
   `client.list_tools()` again and confirm both tools appear. Call it with
   `client.call_tool("multiply", a=6, b=7)` and confirm the result.
2. Call `client.call_tool("subtract", a=1, b=1)` — a tool name that was
   never registered — and confirm your client raises the `unknown tool`
   error instead of crashing somewhere unrelated. That's the same
   "handle a bad tool call gracefully" concern from lesson 35's harness,
   now happening at the protocol boundary.

**Stretch:** add a third request type, `list_resources`/`read_resource`,
to `MockMCPServer` — register a resource as a name mapped to a fixed
string (standing in for a file's contents), answer `list_resources` with
the names, and answer `read_resource` with the stored content for a given
name. This is the same pattern as `tools`, applied to MCP's second
capability: data the client can read without it being an "action."

## Go deeper

- [modelcontextprotocol.io](https://modelcontextprotocol.io/) — the official spec and docs, including the architecture overview this lesson's client/server split follows.
- [Anthropic: Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) — the announcement explaining the problem MCP solves and why it's open.
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) — the real `mcp` package, `FastMCP`, and working example servers.
- [MCP specification — server concepts](https://modelcontextprotocol.io/specification/2025-06-18/server/tools) — the formal definitions of tools, resources, and prompts.

**Next:** [AI Ethics & Responsible AI](44-ai-ethics.md)
