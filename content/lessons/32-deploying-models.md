---
title: "Deploying a Model"
stage: 6
order: 32
minutes: 55
difficulty: intermediate
prerequisites: ["mlops-basics"]
tags: ["deployment", "fastapi", "docker"]
summary: "Wrapping a trained model behind a validated FastAPI endpoint, measuring latency and batching, Dockerizing it, and free hosting options."
---

# Deploying a Model

## Why this matters

Every model built so far in this roadmap has lived in the same process
that trained it — call `.predict()` on the same Python object, in the
same script, on the same machine. That's fine for a notebook, and it's
exactly where lesson 31's tracking and reproducibility work stops being
enough: none of that helps a mobile app, a teammate's script, or a
retention dashboard actually *use* the model you tracked and versioned.
Deployment is the step that turns "a model I can reproduce" into "a
service anything can call" — a small, boring HTTP wrapper around
`.predict()`, which turns out to be most of what deployment actually is.

## The concept

**Wrapping a model behind an API** means separating two concerns that
your notebook currently mixes together: training happens once, offline,
producing a fitted model object; serving happens continuously, in
response to requests, using that already-fitted object without retraining
it. A deployed service loads the model **once**, when it starts up, and
reuses it for every request after that — retraining or refitting per
request would be both pointlessly slow and a correctness bug (the whole
point of lesson 31's reproducibility work is that a given model version
is a fixed, versioned artifact, not something that quietly changes
between requests).

**Request/response schemas.** A model function expects specific,
correctly-typed inputs in a specific order — exactly the kind of contract
that's invisible and easy to violate once it's exposed over the network
to code you don't control. **Pydantic** models make that contract
explicit and enforced: define a class describing exactly what a valid
request looks like (field names, types, and constraints), and FastAPI
validates every incoming request against it automatically, before your
function body ever runs, rejecting anything malformed with a specific
error explaining what was wrong. The same idea applies to the response —
declaring its shape means callers, and FastAPI's auto-generated docs, know
exactly what they'll get back.

**Latency and batching.** A single prediction is typically fast — the
actual `.predict()` call is often the cheapest part of handling a
request. What's *not* free is everything around it: network round-trip
time, and, per request, a fixed amount of framework and validation
overhead. When many predictions are needed at once, sending one HTTP
request per row pays that fixed overhead every single time; sending one
request containing all the rows pays it exactly once. This is a real,
measurable effect, not a theoretical one — "In code" below measures it
directly. The tradeoff is latency for the *first* result: batching is a
throughput optimization, and it means no single prediction in the batch
comes back until the whole batch is processed, so it fits "score these
10,000 rows tonight" far better than "answer this one user's request
right now."

**Dockerizing.** A model that runs "on my machine" depends on a specific
Python version, specific library versions, and often OS-level libraries
underneath them — all invisible until someone else's machine doesn't have
them and the service fails in a way that's genuinely hard to debug
remotely. A **Docker container** packages the code, the exact dependency
versions, and a minimal OS layer into one image that runs identically
wherever Docker runs — the deployment-time equivalent of the pinned
dependencies lesson 31 covered for reproducibility. This lesson's
Dockerfile is shown as a real, idiomatic example, but building and
running it needs a Docker daemon this environment doesn't have — it's
presented honestly as illustrative, with no invented container output.

**Free hosting options.** A container needs somewhere to actually run.
**Hugging Face Spaces** hosts a Dockerfile directly, free, and is aimed
squarely at ML demos — the most direct route from "I have a Dockerfile"
to "there's a public URL." **Render**'s free web-service tier will run a
small FastAPI app directly from a repo, with a real, honest cost: free
services spin down after a period of inactivity and take up to about a
minute to cold-start on the next request, which is fine for a portfolio
demo and wrong for anything latency-sensitive. Free tiers change often,
so check current pricing before relying on one — **Google Cloud Run**
currently has a genuine perpetual free tier (2 million requests/month),
while providers like Railway and Fly.io have dropped their persistent
free tiers in favor of trial credit or a card requirement. Whichever
provider you pick, the pattern to evaluate is the same: how much compute
is actually free right now, and what happens to latency after idle time.

## In code

**Part 1 — the service, called in-process with `TestClient`.** No live
server, no separate terminal — `TestClient` sends real ASGI requests
straight into the FastAPI app object and returns real responses,
including real validation failures.

```python
import time
import warnings
from typing import List, Literal

warnings.filterwarnings("ignore")
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- "training" happens once, at import time -- standing in for loading
# an already-fitted, versioned model artifact on startup.
rng = np.random.default_rng(42)
n = 300
contract_is_month_to_month = rng.integers(0, 2, n)
tenure_months = np.clip(rng.exponential(24, n), 0, 72)
monthly_charges = np.clip(rng.normal(70, 25, n), 20, 150)
churn_logit = -0.6 + 1.3 * contract_is_month_to_month - 0.025 * tenure_months + rng.normal(0, 0.5, n)
churned = (rng.random(n) < 1 / (1 + np.exp(-churn_logit))).astype(int)

X_train = np.column_stack([tenure_months, monthly_charges, contract_is_month_to_month])
scaler = StandardScaler().fit(X_train)
model = LogisticRegression(max_iter=1000).fit(scaler.transform(X_train), churned)


# --- request/response schemas -------------------------------------------
class ChurnRequest(BaseModel):
    tenure_months: float = Field(ge=0, le=100, description="Months as a customer")
    monthly_charges: float = Field(ge=0, le=500)
    contract_type: Literal["month-to-month", "one-year", "two-year"]


class ChurnResponse(BaseModel):
    churn_probability: float
    will_churn: bool


# --- the service ----------------------------------------------------------
app = FastAPI(title="Churn Predictor")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest):
    is_mtm = 1.0 if request.contract_type == "month-to-month" else 0.0
    features = np.array([[request.tenure_months, request.monthly_charges, is_mtm]])
    scaled = scaler.transform(features)
    probability = float(model.predict_proba(scaled)[0, 1])
    return ChurnResponse(churn_probability=round(probability, 4), will_churn=probability >= 0.5)


client = TestClient(app)

print("=== GET /health ===")
r = client.get("/health")
print(r.status_code, r.json())

print("\n=== POST /predict (new month-to-month customer) ===")
r = client.post("/predict", json={
    "tenure_months": 2, "monthly_charges": 95.0, "contract_type": "month-to-month",
})
print(r.status_code, r.json())

print("\n=== POST /predict (loyal two-year customer) ===")
r = client.post("/predict", json={
    "tenure_months": 48, "monthly_charges": 60.0, "contract_type": "two-year",
})
print(r.status_code, r.json())

print("\n=== POST /predict with an invalid contract_type ===")
r = client.post("/predict", json={
    "tenure_months": 10, "monthly_charges": 70.0, "contract_type": "lifetime",
})
print(r.status_code)
print(r.json())

print("\n=== POST /predict missing a required field ===")
r = client.post("/predict", json={
    "monthly_charges": 70.0, "contract_type": "one-year",
})
print(r.status_code)
print(r.json())
```

```
=== GET /health ===
200 {'status': 'ok'}

=== POST /predict (new month-to-month customer) ===
200 {'churn_probability': 0.5484, 'will_churn': True}

=== POST /predict (loyal two-year customer) ===
200 {'churn_probability': 0.1933, 'will_churn': False}

=== POST /predict with an invalid contract_type ===
422
{'detail': [{'type': 'literal_error', 'loc': ['body', 'contract_type'], 'msg': "Input should be 'month-to-month', 'one-year' or 'two-year'", 'input': 'lifetime', 'ctx': {'expected': "'month-to-month', 'one-year' or 'two-year'"}}]}

=== POST /predict missing a required field ===
422
{'detail': [{'type': 'missing', 'loc': ['body', 'tenure_months'], 'msg': 'Field required', 'input': {'monthly_charges': 70.0, 'contract_type': 'one-year'}}]}
```

Both failures were rejected with a real `422 Unprocessable Entity` and a
specific, structured reason — `contract_type` had to be one of three
literal strings, and `tenure_months` was outright missing — without a
single line of hand-written validation code. That's the entire payoff of
the pydantic schema: bad input never reaches the model.

**Part 2 — measuring one-at-a-time versus batched requests for real**,
continuing in the same session with a second endpoint added to the same
app:

```python
class Customer(BaseModel):
    tenure_months: float
    monthly_charges: float
    contract_type: Literal["month-to-month", "one-year", "two-year"]


class BatchRequest(BaseModel):
    customers: List[Customer]


def _to_features(c):
    is_mtm = 1.0 if c.contract_type == "month-to-month" else 0.0
    return [c.tenure_months, c.monthly_charges, is_mtm]


@app.post("/predict_batch")
def predict_batch(request: BatchRequest):
    features = scaler.transform([_to_features(c) for c in request.customers])
    probabilities = model.predict_proba(features)[:, 1]
    return {"churn_probabilities": [round(float(p), 4) for p in probabilities]}


sample = {"tenure_months": 5, "monthly_charges": 80.0, "contract_type": "month-to-month"}
n_customers = 50

start = time.perf_counter()
for _ in range(n_customers):
    client.post("/predict", json=sample)
one_at_a_time = time.perf_counter() - start

start = time.perf_counter()
client.post("/predict_batch", json={"customers": [sample] * n_customers})
batched = time.perf_counter() - start

print(f"{n_customers} predictions, one request each:   {one_at_a_time * 1000:.2f} ms total, "
      f"{one_at_a_time / n_customers * 1000:.3f} ms/customer")
print(f"{n_customers} predictions, one batched request: {batched * 1000:.2f} ms total, "
      f"{batched / n_customers * 1000:.3f} ms/customer")
print(f"speedup: {one_at_a_time / batched:.1f}x")
```

```
50 predictions, one request each:   45.34 ms total, 0.907 ms/customer
50 predictions, one batched request: 1.17 ms total, 0.023 ms/customer
speedup: 38.8x
```

Roughly 39x faster per customer, batched — and this is measured with
`TestClient` calling straight into the app in-process, so it isolates
per-request framework and validation overhead specifically, without even
counting real network round-trip time. Over an actual network, each of
those 50 individual requests would also pay a full round trip, making the
real-world gap even larger than what's measured here.

**Illustrative — a Dockerfile for this service.** Building and running
this needs a Docker daemon, which this environment doesn't have, so
nothing below was executed — no container logs are shown, because none
were produced:

```dockerfile
# ILLUSTRATIVE -- not built or run here.
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
# main.py defines `app = FastAPI(...)`, same as Part 1 above

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# ILLUSTRATIVE -- not run here.
docker build -t churn-predictor .
docker run -p 8000:8000 churn-predictor
# then, from any other machine on the network:
# curl -X POST http://<host-ip>:8000/predict \
#   -H "Content-Type: application/json" \
#   -d '{"tenure_months": 2, "monthly_charges": 95.0, "contract_type": "month-to-month"}'
```

The `Dockerfile` mirrors the two-stage shape from `The concept`: pin
dependencies first so Docker can cache that layer, then copy in the
application code, which changes far more often. `uvicorn` is the ASGI
server that actually runs a FastAPI app outside of `TestClient` — it's
what `docker run` starts inside the container, and what you'd run
directly with `uvicorn main:app --reload` for local development too.

## Build this

Take the FastAPI service from Part 1 (or your own model from lesson 31's
exercise) and actually run it: save it as `main.py`, install `fastapi`
and `uvicorn`, and start it with `uvicorn main:app --reload`. Confirm it
works by opening the auto-generated docs at `http://127.0.0.1:8000/docs`
in a browser and calling `/predict` from there. Then call it over HTTP
from *outside* Python — `curl` from a terminal, or another machine on
your local network using your computer's LAN IP instead of `127.0.0.1` —
and confirm you get the same JSON response `TestClient` gave you above.

**Stretch:** write the `Dockerfile` for your own service, following the
pattern above, and if you have Docker installed, actually build and run
it (`docker build`, `docker run`), then hit it with `curl` exactly as you
did against the uncontainerized version. Confirm the response is
identical. If you don't have Docker available, deploy the
uncontainerized version to Hugging Face Spaces or Render's free tier
instead, and confirm you can reach it from a URL that isn't `localhost`.

## Go deeper

- [FastAPI Documentation](https://fastapi.tiangolo.com/) — the full framework reference, including the automatic `/docs` UI used in the exercise.
- [FastAPI: FastAPI in Containers - Docker](https://fastapi.tiangolo.com/deployment/docker/) — the official guide this lesson's Dockerfile pattern follows.
- [Docker: Get Started](https://docs.docker.com/get-started/) — the official introduction to building and running containers, for anyone doing the Docker stretch goal.
- [Hugging Face: Spaces](https://huggingface.co/docs/hub/spaces) — free hosting for ML demos, including Docker Spaces that run an arbitrary `Dockerfile` directly.
- [Render: Deploy for Free](https://render.com/docs/free) — the free web-service tier referenced in "The concept," including its inactivity spin-down behavior.

**Next:** [Portfolio and Career](33-portfolio-career.md)
