# MAPF-CBS + MARL — Multi-Agent Path Finding

> Conflict-Based Search (optimal baseline) **+** MAPPO with transformer communication,
> CBS-bootstrapped curriculum, and lifelong goals — trained on procedurally generated maps.
> Pure Python · PyTorch · No ROS · No Isaac Sim · Mac-native.

![demo](results/demo.gif)

---

## What this project does

**Multi-Agent Path Finding (MAPF):** given *N* agents on a shared grid, find collision-free
paths from each agent's start to its goal.

This repo contains two solvers and a full training pipeline:

| Component | What it is |
|-----------|-----------|
| **CBS** | Optimal classical solver — guaranteed minimum sum-of-costs |
| **MAPPO + Transformer Comm** | Learned decentralised policy — agents coordinate via attention |
| **CBS-Bootstrapped Curriculum** | Novel training schedule: CBS teaches RL cold-start, then steps away |
| **Lifelong MAPF** | Agents get new goals continuously — models real warehouse robotics |
| **Procedural maps** | Maze, room-corridor, and scatter generators — new terrain every episode |

---

## Architecture

```
                  ┌──────────────────────────────────────────┐
  Each agent i    │  Raw obs [150]                           │
  sees a 7×7      │      ↓                                   │
  local crop +    │  AgentEncoder (MLP)  ──→  embedding [128]│
  goal direction  │                                           │
                  │  All N embeddings  →  Multi-Head         │
  Shared weights  │  Self-Attention (×2 layers)              │  ← Communication
  across agents   │      ↓                                   │    (transformer)
                  │  Enhanced embedding [128] per agent      │
                  │      ↓                    ↓              │
                  │  SharedActor          CentralizedCritic  │
                  │  → action logits      → mean-pool → V(s) │  ← CTDE
                  └──────────────────────────────────────────┘

  Training only:  CBS oracle suggests greedy actions → extra reward (Phase A/B)
  At execution:   CBS is gone — purely emergent coordination from weights
```

### CBS-Bootstrapped Curriculum (novel)

```
Phase A  [steps 0 → 50k]      CBS weight = 1.0   agents learn basic navigation
Phase B  [steps 50k → 200k]   CBS weight 1.0→0.0  gradual handoff to RL
Phase C  [steps 200k → ∞]     CBS weight = 0.0   pure emergent coordination

Difficulty: easy(7×7, 2 agents) → medium(11×11, 4) → hard(15×15, 8) → expert(20×20, 12)
Auto-advances when success rate > 80%,  regresses when < 40%
```

---

## Quick start

```bash
git clone https://github.com/KyawLinnKhant/mapf-cbs
cd mapf-cbs
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the CBS demo (8 agents, produces results/demo.gif)
python main.py

# Train MARL (2M steps, auto-detects MPS/CUDA/CPU)
python train.py

# Evaluate trained policy vs CBS baseline
python eval.py --level medium --n-episodes 100

# CBS-only baseline (no checkpoint needed)
python eval.py --cbs-only --level hard
```

**Apple Silicon:** automatically uses MPS (`--device mps` or `--device auto`).

---

## Training options

```
python train.py --total-steps 5000000    # longer run for expert level
                --start-level medium     # skip easy warmup
                --hidden-dim 256         # larger model
                --n-heads 8              # more attention heads
                --warmup-steps 100000    # extend CBS guidance phase
                --device mps             # Apple Silicon GPU
```

---

## CBS demo results

8 agents on a 12×12 grid with 16 obstacle cells:

| Metric | Value |
|--------|-------|
| Agents | 8 |
| Sum-of-costs | 148 (optimal) |
| Makespan | 22 steps |

---

## Project structure

```
mapf-cbs/
├── src/
│   ├── grid.py         Grid environment — free cells, neighbors
│   ├── astar.py        Space-time A* with vertex + edge constraints
│   ├── cbs.py          Conflict-Based Search (high-level CT search)
│   ├── maps.py         Procedural map generator (maze / rooms / scatter)
│   ├── env.py          Multi-agent MAPF env — lifelong goals, hard collision
│   ├── comm.py         Transformer communication module (variable N agents)
│   ├── mappo.py        MAPPO — shared actor + mean-pool centralized critic
│   ├── curriculum.py   CBS annealer (3 phases) + difficulty scheduler
│   └── trainer.py      Training loop — rollout collection, PPO update, logging
├── main.py             CBS demo — 8 agents, saves results/demo.gif
├── train.py            MARL training entry point
├── eval.py             Evaluation: RL policy vs CBS baseline comparison table
└── requirements.txt    numpy, matplotlib, pillow, torch
```

---

## Why transformer communication?

Standard MARL treats each agent independently. Agents learn to avoid collisions
only through reward signals, which is slow and unstable in dense environments.

With self-attention over all agents' embeddings:
- Agent *i* can "see" what every other agent is doing (implicitly)
- The attention is permutation-equivariant — works for any N agents, any arrangement
- Shared encoder weights = one policy scales to 2 agents or 20 agents

The hardest engineering challenge: variable-length attention (N changes with curriculum)
handled with padding masks — each level uses a different N without reinitialising weights.

---

## Why CBS bootstrap instead of training from scratch?

Cold-start MARL on MAPF is notoriously broken: agents start random, immediately collide,
receive negative reward on every step, and the gradient signal is pure noise. Most runs
never learn to coordinate at all.

CBS bootstrap breaks the cold-start: in Phase A, the CBS oracle shows agents what
good behaviour looks like. As the agent improves, the CBS guidance phases out linearly.
By Phase C the agent is fully autonomous — CBS is gone from the policy entirely.

This is the contribution most comparable to work like PRIMAL (Sartoretti et al. 2019)
and MAPF-LNS2, but applied to the transformer-attention MARL setting.

---

## Related work

- Sharon et al. (2015). *Conflict-based search for optimal multi-agent pathfinding.* AI, 219, 40–66.
- Sartoretti et al. (2019). *PRIMAL: Pathfinding via reinforcement and imitation multi-agent learning.*
- Yu et al. (2022). *The surprising effectiveness of PPO in cooperative multi-agent games (MAPPO).*
- **EKF SLAM** — 2D LiDAR SLAM with Taubin circle fitting: [github.com/KyawLinnKhant/slam_turtlebot_ros](https://github.com/KyawLinnKhant/slam_turtlebot_ros)

---

## Author

**Kyaw Linn Khant** — Robotics & AI Engineer  
[Portfolio](https://kyawlinnkhant.github.io/my_portfolio/) · [LinkedIn](https://linkedin.com/in/kyawlinnkhant) · [GitHub](https://github.com/KyawLinnKhant)
