---
title: "Reinforcement Learning"
stage: 4
order: 39
minutes: 55
difficulty: advanced
prerequisites: ["pytorch"]
tags: ["reinforcement-learning", "q-learning"]
summary: "Learning from reward instead of labels — agents, states, actions, rewards, and Q-learning taught by making an agent solve a tiny grid world."
---

# Reinforcement Learning

## Why this matters

Every model this roadmap has trained so far learned from **labels**: a
correct answer sitting right next to each example, telling the model exactly
what it should have predicted. Reinforcement learning throws that away.
There is no labelled dataset of "correct moves" for a robot arm, a game of
Go, or a chatbot deciding how to respond — there is only a **reward** signal
that arrives, often late and noisy, after a sequence of decisions. The model
has to work out, by trial and error, which of its earlier actions the reward
was actually rewarding. That is a genuinely different learning problem, and
it is the one behind game-playing agents, robotics, and the reinforcement
learning step used to align modern LLMs (RLHF, which Stage 5 touches on).
This lesson builds the smallest complete version of that problem: a table,
a grid, and an agent that learns a working strategy from nothing but reward.

## The concept

**The vocabulary.** An **agent** takes actions inside an **environment**.
At any moment the environment is in some **state** — everything the agent
currently knows about the world. The agent picks an **action**; the
environment responds with a new state and a **reward**, a single number
saying how good or bad that action turned out to be right now. A **policy**
is the agent's strategy: a rule (possibly learned) for choosing an action
given a state. One full run from a starting state to some ending condition
(reaching a goal, falling in a pit, running out of time) is an **episode**.
The whole loop — state, action, reward, new state, repeat — is called the
agent-environment interaction, and it is the only source of learning
signal; nobody hands the agent a correct-action label.

**Explore vs. exploit.** Early on, the agent has no idea which actions are
good, so always picking its current favourite action (**exploit**) risks
locking in on a mediocre strategy it never questions. Always picking a
random action (**explore**) never uses what it has learned. The standard
fix is **epsilon-greedy**: with probability `epsilon` take a random action,
otherwise take the best action currently known. `epsilon` is often started
high (mostly explore, when nothing is known yet) and decayed over training
toward a small value (mostly exploit, once the agent knows something
worth exploiting) — this lesson's exercise section has you try that decay.

**The value of a state-action pair.** Some actions pay off immediately;
others only pay off several steps later (moving toward a distant goal
earns nothing *this* step but sets up a big reward later). To act well, the
agent needs a way to score "how good is taking action `a` in state `s`,
counting not just the immediate reward but everything that reasonably
follows from it." That score is called a **Q-value**, written `Q(s, a)`,
and the table of every `Q(s, a)` for every state-action pair is the
**Q-table**. If the agent had a perfect Q-table, acting optimally would
just be "in state `s`, pick the action with the highest `Q(s, a)`" — the
entire learning problem reduces to *learning that table*.

**Q-learning and the Bellman update.** Q-learning learns the table by
repeatedly nudging each estimate toward a slightly better estimate, using
the **Bellman equation**: the value of taking action `a` in state `s`
should equal the immediate reward, plus the discounted value of the best
action available from wherever that action lands you. As an update rule,
after taking action `a` in state `s`, landing in state `s'`, and receiving
reward `r`:

```
Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))
```

`alpha` is the learning rate (how much each observation moves the
estimate); `gamma` is the discount factor between 0 and 1 (how much future
reward matters compared to immediate reward — `gamma` close to 1 means "a
reward 10 steps away still matters a lot"). The term in parentheses is the
**TD (temporal-difference) error**: the gap between what the agent
currently believes `Q(s, a)` is worth, and a slightly-better-informed
estimate built from the reward it actually just received plus its current
best guess about what comes next. Repeat this update over thousands of
episodes and the Q-table converges toward values that make the greedy
policy — always pick `argmax_a Q(s, a)` — actually solve the task, entirely
from trial, error, and reward, with no labelled "correct action" ever
provided.

**Where deep RL comes in.** A Q-*table* only works when the number of
states is small enough to enumerate — 25 grid cells, as below, is nothing;
the raw pixels of an Atari screen or a real camera feed have far too many
possible states to tabulate. **Deep Q-Networks (DQN)** replace the table
with a neural network, `Q(s, a; theta)`, that takes a state as input and
outputs a Q-value per action — the same Bellman-error idea, but the update
is now a gradient step on `theta` (via the exact `loss.backward()` /
`optimizer.step()` loop from the PyTorch lesson) that pushes the network's
prediction toward `r + gamma * max_a' Q(s', a'; theta)`, instead of writing
straight into a table cell. `# ILLUSTRATIVE`: everything else about this
lesson's grid world — epsilon-greedy, the Bellman update, episodes,
convergence — carries over unchanged; only "table lookup" becomes "network
forward pass."

## In code

A 5x5 grid world: the agent starts top-left, must reach a goal bottom-right,
and there is one pit that ends the episode badly if stepped on. Reward is
`-1` per step (so the agent prefers short paths), `+10` for reaching the
goal, `-10` for falling in the pit. Everything is seeded, so this run is
exactly reproducible.

```python
import numpy as np

rng = np.random.default_rng(0)

# 5x5 grid. States numbered 0..24, row-major: state = row * 5 + col.
GRID_SIZE = 5
START = (0, 0)
GOAL = (4, 4)
PIT = (2, 2)

N_STATES = GRID_SIZE * GRID_SIZE
ACTIONS = ["up", "down", "left", "right"]
N_ACTIONS = len(ACTIONS)
MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def to_state(pos):
    r, c = pos
    return r * GRID_SIZE + c


def step(pos, action):
    """Apply an action, clipped to the grid, and return (new_pos, reward, done)."""
    dr, dc = MOVES[action]
    r, c = pos
    new_r = min(max(r + dr, 0), GRID_SIZE - 1)
    new_c = min(max(c + dc, 0), GRID_SIZE - 1)
    new_pos = (new_r, new_c)

    if new_pos == GOAL:
        return new_pos, 10.0, True
    if new_pos == PIT:
        return new_pos, -10.0, True
    return new_pos, -1.0, False  # -1 per step so the agent prefers short paths


# Q-table: one row per state, one column per action, all starting at 0
Q = np.zeros((N_STATES, N_ACTIONS))

alpha = 0.1       # learning rate
gamma = 0.9        # discount factor -- how much future reward matters now
epsilon = 0.2      # exploration rate: chance of a random action instead of the best one
n_episodes = 2000
max_steps = 50

for episode in range(n_episodes):
    pos = START
    for _ in range(max_steps):
        s = to_state(pos)

        if rng.random() < epsilon:
            action_idx = rng.integers(N_ACTIONS)   # explore: random action
        else:
            action_idx = int(np.argmax(Q[s]))      # exploit: best known action

        action = ACTIONS[action_idx]
        new_pos, reward, done = step(pos, action)
        s_next = to_state(new_pos)

        # Bellman update: move Q[s, a] toward reward + discounted best future value
        best_next = np.max(Q[s_next])
        Q[s, action_idx] += alpha * (reward + gamma * best_next - Q[s, action_idx])

        pos = new_pos
        if done:
            break

print("Training done:", n_episodes, "episodes")

# Extract the greedy policy: the best action in every state
policy_grid = []
arrows = {"up": "^", "down": "v", "left": "<", "right": ">"}
for r in range(GRID_SIZE):
    row_symbols = []
    for c in range(GRID_SIZE):
        if (r, c) == GOAL:
            row_symbols.append("G")
        elif (r, c) == PIT:
            row_symbols.append("X")
        else:
            s = to_state((r, c))
            best_action = ACTIONS[int(np.argmax(Q[s]))]
            row_symbols.append(arrows[best_action])
    policy_grid.append(" ".join(row_symbols))

print("\nLearned policy (S=start, G=goal, X=pit):")
for r, row in enumerate(policy_grid):
    prefix = "S " if (r, 0) == START else "  "
    print(prefix + row)

# Walk the greedy policy from start and confirm it reaches the goal
pos = START
path = [pos]
total_reward = 0.0
for _ in range(max_steps):
    s = to_state(pos)
    action = ACTIONS[int(np.argmax(Q[s]))]
    pos, reward, done = step(pos, action)
    total_reward += reward
    path.append(pos)
    if done:
        break

print("\nGreedy path from start:", path)
print("Reached goal:", pos == GOAL)
print("Total reward:", total_reward)
```

```
Training done: 2000 episodes

Learned policy (S=start, G=goal, X=pit):
S > > > v v
  > > > > v
  ^ v X > v
  > v v > v
  > > > > G

Greedy path from start: [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (1, 4), (2, 4), (3, 4), (4, 4)]
Reached goal: True
Total reward: 3.0
```

No part of this code was told the shortest path, or even that a shortest
path was desirable — that fell out entirely from `-1` per step pushing the
Bellman update toward shorter routes, `+10` pulling it toward the goal, and
`-10` pushing it away from the pit at `(2, 2)` (visible in the policy grid:
row 2's arrows route *around* the `X`, not into it). Total reward of `3.0`
over an 8-step path is exactly `-1 * 8 + 10`, confirming the agent is both
reaching the goal and doing it efficiently.

## Build this

Change the pit's position, or add a second pit, or change the step penalty
from `-1` to `-0.1`, and retrain. Print the new policy grid and check two
things: does the agent still route around every pit, and does a smaller
step penalty produce a *different*-looking (possibly less direct) path
than `-1` did? Explain in one sentence why the size of the step penalty
changes what the "optimal" path looks like.

**Stretch:** replace the fixed `epsilon = 0.2` with a decaying schedule —
for example `epsilon = max(0.01, 1.0 - episode / n_episodes)` — so the
agent explores heavily in early episodes and mostly exploits by the end.
Track the total reward per episode and print the average over the first
100 episodes versus the last 100; with decay, the late-training average
should be noticeably higher and more consistent than without it.

## Go deeper

- [Sutton & Barto, *Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/RLbook2020.pdf) — the free, canonical textbook; Chapter 6 covers Q-learning and TD methods directly.
- [OpenAI Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/) — a free, code-first introduction that goes from this lesson's tabular methods to modern deep RL algorithms.
- [DeepMind x UCL Reinforcement Learning Lecture Series](https://www.youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm) — free video lectures covering the same ideas with more formal depth.
- [Gymnasium documentation](https://gymnasium.farama.org/) — the standard toolkit of RL environments (the maintained successor to OpenAI Gym); a natural next step once a grid world feels too small.

**Next:** [Diffusion Models & Image Generation](40-diffusion-models.md)
