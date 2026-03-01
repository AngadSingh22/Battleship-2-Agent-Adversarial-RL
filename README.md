# Battleship Adversarial RL (IBR + C-Core)

This repository implements a two-player adversarial training framework for Battleship using Iterative Best Response (IBR). We train an Attacker policy (shot selection) against a Defender policy (ship placement), alternating optimization across generations to approximate a robust equilibrium-like behavior under distribution shift in placements.

The environment’s inner loop is implemented in a C shared library (`libbattleship`) and exposed to Python via `ctypes`, enabling high-throughput rollouts for RL training. Policies are trained in Python (PyTorch / Stable-Baselines3) using PPO with action masking.

---

## What this project is

Battleship can be modeled as a partially observable decision problem for the attacker: the ship layout is hidden, and the attacker only observes shot outcomes (miss, hit, sunk). We extend this to a two-player setting by making the ship placement an adversarial decision. This yields a zero-sum Markov game where the defender chooses the latent initial state and the attacker acts sequentially under partial observability.

The core experimental question is: how does an attacker trained only on “random placements” degrade when evaluated against biased or adversarial placements, and can IBR reduce that gap?

---

## Problem formulation (high level)

*   **Attacker (A)**: chooses shots until all ships are sunk.
*   **Defender (D)**: chooses a legal ship placement at episode start.

**Payoff**: we use turns-to-clear (shots-to-win) as the primary objective.
*   Attacker wants to *minimize* turns-to-clear.
*   Defender wants to *maximize* turns-to-clear.

Formal definitions of the state, actions, observation function, and reward are written in [`formulation/`](./formulation/).

---

## Training method: Iterative Best Response

We use a discrete generation loop. Each generation trains one side against a fixed opponent from the previous generation. This “frozen opponent” structure reduces non-stationarity and allows each training phase to be treated as a stationary RL problem.

Let $A_k$ denote the attacker policy after attacker-training in generation $k$, and $D_k$ the defender policy after defender-training in generation $k$.

1.  **Defender update** (train placement policy against frozen attacker):
    Maximize expected turns-to-clear induced by the attacker.
2.  **Attacker update** (train shooting policy against frozen defender):
    Minimize expected turns-to-clear under that defender distribution.

In practice, both phases are trained with PPO (Stable-Baselines3), using action masking to forbid invalid shots (already-fired cells).

*Note: this procedure is related to fictitious play and best-response dynamics, but it does not guarantee convergence to a Nash equilibrium in general. The repo is structured to make the dynamics measurable (gap vs biased/adversarial placements) rather than to claim theoretical convergence.*

---

## System architecture

The design goal is high rollout throughput and reproducibility.

| Component | Path | Description |
| :--- | :--- | :--- |
| **C Core** | `csrc/` | Environment state transitions, valid mask computation, minimal allocations. |
| **Bindings** | `bindings/` | `ctypes` interface. Zero-copy views into C-owned buffers exposed as NumPy arrays. |
| **Python RL** | `battleship_rl/` | Gymnasium env wrapper, PPO training (sb3), IBR orchestration. |
| **Scripts** | `training/`, `evaluation/`, `tools/` | Entrypoints, looping, and diagnostic utilities. |
| **Formulation** | `formulation/` | LaTeX sources for the formal POMDP/Markov game definition. |

---

## Build (C-core)

Windows (MSVC):

```powershell
.\build.bat
```

This produces a shared library (DLL) that Python loads via `ctypes`.

---

## Run

### 1. Train attacker vs uniform placements (sanity baseline)

```powershell
python -m scripts.train_ppo --total-timesteps 1000000
```

### 2. Train defender against a frozen attacker (one IBR defender step)

```powershell
python -m scripts.train_defender --attacker-path runs/adversarial/gen0_attacker.zip
```

### 3. Run the full IBR loop (multiple generations)

```powershell
python -m scripts.train_adversarial_loop --generations 5
```

---

## Evaluation protocol (what to report)

Primary metric:
*   **mean shots-to-win** (turns-to-clear)

Robustness metrics:
*   **90th percentile shots-to-win**
*   **generalization gap** = $\mu(\text{shots} | \text{biased}) - \mu(\text{shots} | \text{uniform})$

---

## Notes on performance claims

Throughput depends on board size, number of parallel envs, CPU, and vectorization method (SubprocVecEnv vs DummyVecEnv). The C-core is designed to remove Python overhead from the step function; we observe >40k FPS on standard consumer hardware.
