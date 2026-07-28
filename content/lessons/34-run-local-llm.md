---
title: "Run Open LLMs Locally with Ollama"
stage: 5
order: 34
minutes: 45
difficulty: intermediate
prerequisites: ["how-llms-work"]
tags: ["llm", "local-models", "ollama"]
summary: "Run real language models on your own machine — private, free, offline — and call them from Python like an API."
---

# Run Open LLMs Locally with Ollama

## Why this matters

Everything you've done with LLMs so far has gone through someone else's API: a
key, a bill, a rate limit, and your data leaving your machine. But there are now
excellent **open-weight** models you can download and run entirely on your own
computer — no key, no cost, no internet required, and nothing you type ever
leaves the room. For learning, prototyping, and anything private, running a model
locally is a superpower. It's also the same next-token prediction from
[how LLMs work](24-how-llms-work.md) — just on your hardware instead of a data centre.

## The concept

**Open weights vs. an API.** Models like Meta's **Llama**, Mistral, Alibaba's
**Qwen**, and Google's **Gemma** publish their actual weights. Anyone can download
them and run them. That's different from a closed API model, where you only ever
send text in and get text out.

**A runner does the heavy lifting.** You don't load raw weights yourself — a
runner handles it. The friendliest is **[Ollama](https://ollama.com/)**: one
command downloads a model and starts a local server. Alternatives are
**llama.cpp** (the engine under the hood) and **LM Studio** (a desktop app).

**Quantization is why this fits on a laptop.** A model's weights are numbers.
Storing each in full precision (16 bits) is accurate but big. **Quantization**
rounds them to fewer bits — often 4 — which shrinks the model ~4× with only a
small quality hit. Most local models ship as **GGUF** files at 4-bit. The rough
memory a model needs is just *parameters × bytes-per-weight*:

```python
def model_ram_gb(params_billion, bits):
    # Each weight takes `bits / 8` bytes. Memory ~ params * bytes-per-weight.
    bytes_per_param = bits / 8
    return round(params_billion * bytes_per_param, 1)  # GB, params in billions

print(f"{'model':>8} | {'fp16':>7} | {'8-bit':>7} | {'4-bit':>7}")
for p in [1, 3, 7, 8, 13, 70]:
    print(f"{str(p)+'B':>8} | {model_ram_gb(p,16):>5}GB | {model_ram_gb(p,8):>5}GB | {model_ram_gb(p,4):>5}GB")
```

```
   model |    fp16 |   8-bit |   4-bit
      1B |   2.0GB |   1.0GB |   0.5GB
      3B |   6.0GB |   3.0GB |   1.5GB
      7B |  14.0GB |   7.0GB |   3.5GB
      8B |  16.0GB |   8.0GB |   4.0GB
     13B |  26.0GB |  13.0GB |   6.5GB
     70B | 140.0GB |  70.0GB |  35.0GB
```

So a 7–8B model at 4-bit needs about 4 GB — comfortable on a modern laptop.
Add a bit of headroom for the context window. That's why 7B/8B models are the
sweet spot for running locally.

## In code

**1. Install and run — two commands.** After installing Ollama from
[ollama.com](https://ollama.com/download):

```bash
# Download a small, capable model and start chatting in your terminal.
ollama run llama3.2
```

Ollama downloads the model once, then drops you into a chat prompt. Ask it
anything; type `/bye` to leave. The model runs on your machine — pull your
network cable and it still answers.

**2. Call it from Python.** While Ollama is running it also serves a local HTTP
API on port `11434`. This snippet is **illustrative** — it needs the Ollama
server running, so we don't capture its output here (the reply is different every
time anyway):

```python
import requests

# ILLUSTRATIVE — requires `ollama serve` running locally.
resp = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "llama3.2",
        "messages": [{"role": "user", "content": "Explain a neural net in one sentence."}],
        "stream": False,
    },
)
print(resp.json()["message"]["content"])
```

You get back JSON whose `message.content` holds the model's answer — the same
shape you'd parse from a hosted API. Ollama also exposes an **OpenAI-compatible**
endpoint at `http://localhost:11434/v1`, so code written for the OpenAI SDK works
against your local model by just changing the base URL and using any string as
the key.

## Build this

1. Install Ollama and run a model that fits your machine — use the estimator
   above to pick the biggest one your RAM can hold at 4-bit (try `llama3.2` for
   ~3B, or `llama3.1` / `qwen2.5` for ~7–8B).
2. Ask it the same question three times and notice the answers differ — that's
   the sampling temperature from [how LLMs work](24-how-llms-work.md), live on
   your own hardware.
3. Then call it from Python with the snippet above and print the reply.

**Stretch:** point your RAG project from [lesson 27](27-rag.md) at your local
model instead of a hosted API — now your whole pipeline runs offline and free.

## Go deeper

- [Ollama — library of models](https://ollama.com/library) — browse everything you can `ollama run`.
- [Ollama API docs](https://github.com/ollama/ollama/blob/main/docs/api.md) — the endpoints, streaming, and options.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — the engine under most local runners; worth understanding.
- [LM Studio](https://lmstudio.ai/) — a polished desktop app if you'd rather not use the terminal.
- [Hugging Face — GGUF models](https://huggingface.co/models?library=gguf) — thousands of quantized models to download.

**Next:** [Build an Agent Harness](35-agent-harness.md)
