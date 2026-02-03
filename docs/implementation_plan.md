# Battleship RL Implementation Plan

**Goal**: Populate the repository with a working implementation of the Battleship POMDP, Baselines, and RL Training infrastructure, strictly following the specifications in `problem/main.tex` and `solution/main.tex`.

## User Review Required
> [!IMPORTANT]
> **Observation**: Shape `(3, H, W)`, Channel-First.
> **Invalid Actions**: `debug=True` $\to$ Raise `ValueError`; `debug=False` $\to$ `Truncated=True` + Penalty (State Unchanged).
> **Outcome Info**: `info["last_outcome"]` tuple: `("HIT", ship_id)`, `("SUNK", ship_id)`, `("MISS", None)`.
> **Seeding**: `reset(seed=...)` calls `super().reset(seed=seed)`. `self.np_random` passed to Placement/Defender.

## Proposed Changes

### 1. Core Environment (`battleship_rl/envs/`)

#### [NEW] [placement.py](file:///g:/Battleship2AgentAdvRL/Battleship-2-Agent-Adversarial-RL/battleship_rl/envs/placement.py)
- **Contract**: `ship_id_grid` is `(H, W)` int array. `-1` = Empty. `0..N-1` = Ship IDs (assigned by placement order).
- **Functions**:
    - `sample_placement(board_size, ships, rng) -> board`: Uses provided `rng` for determinism.

#### [NEW] [battleship_env.py](file:///g:/Battleship2AgentAdvRL/Battleship-2-Agent-Adversarial-RL/battleship_rl/envs/battleship_env.py)
- **Spaces**:
    - `action_space`: `Discrete(H * W)`. Action `a` maps to `(a // W, a % W)`.
    - `observation_space`: `Box(low=0.0, high=1.0, shape=(3, H, W), dtype=np.float32)`. Channel-first.
- **Seeding**:
    - `reset(seed=None, options=None)`: Must call `super().reset(seed=seed)`.
    - `self.np_random` is passed to `defender.sample_layout`.
- **Logic**:
    - `step(action)`:
        - **Invalid**: If `action` mask is False:
            - **State-Safe**: Do NOT mutate hits/misses.
            - If `debug`: Raise `ValueError`.
            - Else: Return `reward=-100`, `truncated=True`, `info["outcome_type"]="INVALID"`.
        - **Transition**:
            - If Hit: Update `hits_grid`.
            - If Sunk: Update `sunk_ships`. Emit `outcome_type="SUNK"`. (Note: SUNK implies HIT).
    - **Info Dict**:
        - `action_mask`: `np.array(H*W, dtype=bool)`.
        - `outcome_type`: "MISS", "HIT", "SUNK".
        - `outcome_ship_id`: `int` (if Hit/Sunk) or `None`.
    - `get_action_mask()`: Helper method for `ActionMasker`.

#### [NEW] [observations.py](file:///g:/Battleship2AgentAdvRL/Battleship-2-Agent-Adversarial-RL/battleship_rl/envs/observations.py)
- **Channel Order**: 0=Hit, 1=Miss, 2=Unknown.
- **Invariant**: `Unknown = 1.0 - (Hit + Miss)`.

#### [NEW] [rewards.py](file:///g:/Battleship2AgentAdvRL/Battleship-2-Agent-Adversarial-RL/battleship_rl/envs/rewards.py)
- **Default (Problem Objective)**: `StepPenaltyReward` ($r_t = -1$).
- **Ablation**: `ShapedReward` ($r_t = -1 + \alpha \cdot \text{hit} + \beta \cdot \text{sunk}$).

### 2. Agents & Training (`battleship_rl/agents/`)

#### [NEW] [defender.py](file:///g:/Battleship2AgentAdvRL/Battleship-2-Agent-Adversarial-RL/battleship_rl/agents/defender.py)
- **Implementations**:
    - `UniformRandomDefender`: Delegates to `placement.py` using `rng`.
    - `BiasedDefender`: Deterministic edge-bias. Weight = `1 / (dist_to_edge + 1)`.

#### [NEW] [sb3_train.py](file:///g:/Battleship2AgentAdvRL/Battleship-2-Agent-Adversarial-RL/battleship_rl/agents/sb3_train.py)
- **Algorithm**: `MaskablePPO` (sb3-contrib).
- **Wrapper**: `ActionMasker` applied **per-env** (inside `make_env` callable) before vectorization.
- **Vectorization**: `SubprocVecEnv` (Seeded: base + rank).

#### [NEW] [policies.py](file:///g:/Battleship2AgentAdvRL/Battleship-2-Agent-Adversarial-RL/battleship_rl/agents/policies.py)
- **MaskableActorCriticPolicy**: Custom CNN (Channel-First).

### 3. Baselines (`battleship_rl/baselines/`)

#### [NEW] [heuristic_probmap.py](file:///g:/Battleship2AgentAdvRL/Battleship-2-Agent-Adversarial-RL/battleship_rl/baselines/heuristic_probmap.py)
- **State Tracking**:
    - Baseline maintains its own `hit_grid`, `miss_grid`, `sunk_set`.
    - Updates from `obs` and `info` (`outcome_type == "SUNK"` adds to `sunk_set`).
- **Input**: Reads `info` only for events, not for hidden state.
- **Fallback**: Hunt/Target if sampling fails.

### 4. Evaluation (`battleship_rl/eval/`)

#### [NEW] [metrics.py](file:///g:/Battleship2AgentAdvRL/Battleship-2-Agent-Adversarial-RL/battleship_rl/eval/metrics.py)
- `mean_shots_to_win`: Timestep when `terminated=True`.
- `failure_rate`: Fraction where `truncated=True`.
- `shots_90th_percentile`.
- `generalization_gap`: `AvgShots(Biased) - AvgShots(Uniform)`. (Positive = Worse).

#### [NEW] [evaluate.py](file:///g:/Battleship2AgentAdvRL/Battleship-2-Agent-Adversarial-RL/battleship_rl/eval/evaluate.py)
- Reports strict table: **Mean | Std | 90th% | Fail Rate | Gap**.

### 5. Static Frontend (GitHub Pages)

#### [NEW] [site/](file:///g:/Battleship2AgentAdvRL/Battleship-2-Agent-Adversarial-RL/site/)
- **Purpose**: Low-dependency visualization interface.
- **Structure**:
    - `index.html`: Main SPA container.
    - `styles.css`: CSS (matches "Deep Navy" aesthetic).
    - `ui/`: `board.js` (Grid render), `replay.js` (Step logic), `metrics.js` (Table render).
    - `data/`: JSON artifacts (ignored by git, populated by script).

#### [NEW] [scripts/export_web_artifacts.py](file:///g:/Battleship2AgentAdvRL/Battleship-2-Agent-Adversarial-RL/scripts/export_web_artifacts.py)
- **Trigger**: Runs after `evaluate.py`.
- **Outputs**:
    - `site/data/config.json`: Metadata (H, W, Ships).
    - `site/data/metrics.json`: `{ "mean": ..., "std": ..., "p90": ..., "gap": ... }`.
    - `site/data/replays/*.json`: `{ "seed": 123, "steps": [ { "r": 0, "c": 0, "type": "MISS" }, ... ] }`.
- **Schema Lock**: Replay steps must match `info["outcome_type"]` enum.

#### [NEW] [.github/workflows/pages.yml](file:///g:/Battleship2AgentAdvRL/Battleship-2-Agent-Adversarial-RL/.github/workflows/pages.yml)
- **Job**: Checkout -> Upload Pages Artifact (`site/`) -> Deploy.
- **Trigger**: Manual workflow dispatch or push to main (optional).

### 6. Verification Plan & Traceability

#### [NEW] [tests/test_sb3_integration.py](file:///g:/Battleship2AgentAdvRL/Battleship-2-Agent-Adversarial-RL/tests/test_sb3_integration.py)
- Integrate `MaskablePPO` with `ActionMasker` wrapper.

#### [NEW] [tests/test_seed_reproducibility.py](file:///g:/Battleship2AgentAdvRL/Battleship-2-Agent-Adversarial-RL/tests/test_seed_reproducibility.py)
- Fix seed, run 2 episodes. Assert identical layouts and action outcomes.

#### [NEW] [tests/test_action_mapping.py](file:///g:/Battleship2AgentAdvRL/Battleship-2-Agent-Adversarial-RL/tests/test_action_mapping.py)
- Verify `(r, c) <-> Action ID` bijection.

#### [NEW] [docs/traceability.md](file:///g:/Battleship2AgentAdvRL/Battleship-2-Agent-Adversarial-RL/docs/traceability.md)
- **Encoding Commitments**:
    - [ ] Channel Order: Hit, Miss, Unknown.
    - [ ] Action Indexing: Row-Major `a = r*W + c`.
    - [ ] Sunk-by-ID Semantics.
    - [ ] Outcome Schema: `("TYPE", ID/None)`.
