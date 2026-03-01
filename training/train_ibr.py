"""
train_ibr.py — Stage 2: Iterative Best Response training loop.

Procedure per generation k:
  A) Train Defender D_k against frozen Attacker A_{k-1}
  B) Train Attacker A_k against frozen Defender mixture [D_k + Uniform + ...]
  C) Evaluate robustness and exploitability proxy

Usage:
  PYTHONPATH=. .venv/bin/python3 training/train_ibr.py \
      --init_attacker results/training/stage1/B/final_model.zip \\
      --generations 3 \\
      --attacker_steps 1000000 \\
      --defender_steps 300000

Artifacts per generation:
  results/training/stage2/
    attacker_gen_{k}.zip
    defender_gen_{k}.zip
    eval_gen_{k}.json
    tensorboard_attacker/
    tensorboard_defender/
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

# Performance knobs — must come before SB3/gym imports
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from sb3_contrib.common.wrappers import ActionMasker

from battleship_rl.agents.defender import (
    ClusteredDefender,
    EdgeBiasedDefender,
    ParityDefender,
    SpreadDefender,
    UniformRandomDefender,
)
from battleship_rl.agents.policies import BattleshipFeatureExtractor
from battleship_rl.envs.battleship_env import BattleshipEnv
from battleship_rl.envs.defender_env import DefenderEnv, build_layout_pool, evaluate_attacker_on_layout


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

BASE_SEED   = 42
N_ATCK_ENVS = 16
POOL_SIZE   = 50_000
# K_EVAL removed: k_eval is now a CLI arg (--k_eval), default 1
BOARD_SIZE  = 10
SHIPS       = [5, 4, 3, 3, 2]

SCRIPTED_DEFENDERS = {
    "UNIFORM":  UniformRandomDefender,
    "EDGE":     EdgeBiasedDefender,
    "CLUSTER":  ClusteredDefender,
    "SPREAD":   SpreadDefender,
    "PARITY":   ParityDefender,
}

ATTACKER_PPO_KWARGS = dict(
    policy="CnnPolicy",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=1024,
    n_epochs=5,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    max_grad_norm=0.5,
    policy_kwargs=dict(
        features_extractor_class=BattleshipFeatureExtractor,
        features_extractor_kwargs={"features_dim": 512},
        net_arch=dict(pi=[512, 512], vf=[512, 512]),
    ),
    device="cuda",
    verbose=0,
)


# ---------------------------------------------------------------------------
# Learned-defender wrapper — serializable, no GPU in subprocess workers
# ---------------------------------------------------------------------------

class LearnedLayoutDefender:
    """Defender that places ships by sampling from a pre-extracted set of
    layouts that were chosen by D_k.  Because it only holds a numpy array
    it is fully pickle-able and works safely inside SubprocVecEnv workers.

    The layouts array has shape (N, H, W) with integer ship-id cells.
    """

    def __init__(self, layouts: np.ndarray):
        if layouts.ndim != 3 or len(layouts) == 0:
            raise ValueError(f"layouts must be (N, H, W) with N>0, got {layouts.shape}")
        self._layouts = layouts

    def sample_layout(
        self,
        board_size: tuple,
        ships: list,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample one layout uniformly from D_k's extracted distribution."""
        idx = int(rng.integers(0, len(self._layouts)))
        return self._layouts[idx].copy()


def extract_dk_layouts(
    defender_model: PPO,
    layout_pool: np.ndarray,
    n_layouts: int = 2_000,
    seed: int = 0,
) -> np.ndarray:
    """Run D_k's policy stochastically to collect a representative sample of
    the layouts it would choose, then return those layouts as a numpy array.

    Strategy:
      - Run D_k predict() with N different dummy observations (slight noise).
      - Collect the chosen layout index each time.
      - Return the corresponding layouts from the pool.

    This is CPU-only after defender_model.policy is moved to CPU.
    """
    obs_shape = defender_model.observation_space.shape
    rng = np.random.default_rng(seed)

    # Move the policy to CPU temporarily for fast batched inference
    device_orig = defender_model.device
    defender_model.policy.set_training_mode(False)

    chosen_indices = []
    for i in range(n_layouts):
        # Use slight observation noise so we sample the full distribution,
        # not just the single deterministic argmax.
        obs = rng.uniform(0.0, 1.0, size=obs_shape).astype(np.float32)
        idx, _ = defender_model.predict(obs, deterministic=False)
        idx_clipped = int(idx) % len(layout_pool)  # guard against off-by-one
        chosen_indices.append(idx_clipped)

    unique_indices = list(set(chosen_indices))
    dk_layouts = layout_pool[unique_indices]
    print(f"  [D_k extract] {n_layouts} rolls → {len(unique_indices)} unique layouts "
          f"(pool coverage: {100 * len(unique_indices) / len(layout_pool):.1f}%)")
    return dk_layouts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mask_fn(env):
    return env.get_action_mask()


def make_attacker_env(defender, seed: int):
    """Single fixed-defender attacker training env — used for IBR attacker phase."""
    def _init():
        env = BattleshipEnv(defender=defender, debug=False)
        env = ActionMasker(env, mask_fn)
        env.reset(seed=seed)
        return env
    return _init


def make_attacker_vec_env(defenders: list, weights: list, base_seed: int):
    """Mixed-defender attacker training env for IBR attacker phase."""
    fns = []
    for rank in range(N_ATCK_ENVS):
        def _init(r=rank):
            env = BattleshipEnv(defenders=defenders, defender_weights=weights, debug=False)
            env = ActionMasker(env, mask_fn)
            env.reset(seed=base_seed + r)
            return env
        fns.append(_init)
    return VecMonitor(SubprocVecEnv(fns))


def evaluate_attacker_scripted(
    attacker: MaskablePPO,
    n_episodes: int = 100,
    seed: int = 9000,
) -> dict:
    """Evaluate attacker against all 5 scripted defender modes.

    BUG FIX: each mode creates its own env; close it after each mode.
    Previously env.close() was called once outside the loop, on the last env only.
    """
    results = {}
    for name, cls in SCRIPTED_DEFENDERS.items():
        lengths = []
        env = BattleshipEnv(defender=cls(), debug=False)
        try:
            for ep in range(n_episodes):
                obs, _ = env.reset(seed=seed + ep)
                done = False
                steps = 0
                while not done:
                    mask = env.get_action_mask()
                    action, _ = attacker.predict(
                        obs[np.newaxis], action_masks=mask[np.newaxis], deterministic=True
                    )
                    obs, _, term, trunc, _ = env.step(int(action[0]))
                    steps += 1
                    done = term or trunc
                lengths.append(steps)
        finally:
            env.close()  # always release — was leaking before
        arr = np.array(lengths)
        results[name] = {"mean": float(arr.mean()), "p90": float(np.percentile(arr, 90))}
    return results


def evaluate_attacker_on_learned_defender(
    attacker: MaskablePPO,
    defender_model: PPO,
    layout_pool: np.ndarray,
    n_episodes: int = 50,
    seed: int = 9999,
) -> dict:
    """Evaluate attacker against a learned defender (D_k).

    BUG FIX (L2): original used zero-obs + deterministic=True, so every episode
    queried the same layout — making n_episodes=50 a single-sample measurement.
    Now uses stochastic predict + seeded RNG noise so we sample D_k's actual
    layout distribution across n_episodes.

    BUG FIX: action_d is clamped to layout_pool bounds via modulo.
    """
    lengths = []
    pool_size = len(layout_pool)
    obs_shape = defender_model.observation_space.shape
    rng = np.random.default_rng(seed)

    for ep in range(n_episodes):
        # Sample a slightly-perturbed obs to get stochastic layout diversity.
        # D_k policy is stateless (obs = rolling stats only), so any obs explores
        # the layout distribution — noise prevents the same mode from always winning.
        obs_d = rng.uniform(0.0, 0.1, size=obs_shape).astype(np.float32)
        action_d, _ = defender_model.predict(obs_d, deterministic=False)
        layout_idx = int(action_d) % pool_size  # clamp: guard against pool size mismatch
        layout = layout_pool[layout_idx]

        mean_shots, ep_shots = evaluate_attacker_on_layout(
            layout=layout, attacker_policy=attacker,
            k_episodes=1, board_size=BOARD_SIZE, ships=SHIPS, seed=seed + ep
        )
        lengths.append(ep_shots[0])

    arr = np.array(lengths)
    return {"mean": float(arr.mean()), "p90": float(np.percentile(arr, 90))}



# ---------------------------------------------------------------------------
# Phase A: Train Defender D_k
# ---------------------------------------------------------------------------

def train_defender(
    generation: int,
    frozen_attacker: MaskablePPO,
    layout_pool: np.ndarray,
    steps: int,
    out_dir: Path,
    max_generations: int,
    seed: int = BASE_SEED,
    k_eval: int = 1,
) -> PPO:
    """Train Defender D_k against a frozen attacker. Returns the trained PPO model."""
    def_env = DefenderEnv(
        layout_pool=layout_pool,
        attacker_policy=frozen_attacker,
        k_eval_episodes=k_eval,
        generation=generation,
        max_generations=max_generations,
        board_size=BOARD_SIZE,
        ships=SHIPS,
        seed=seed,
    )
    def_venv = VecMonitor(DummyVecEnv([lambda: def_env]))

    # Defender uses plain PPO with MlpPolicy — runs faster on CPU (no CNN needed).
    # SB3 explicitly warns against running MlpPolicy on GPU.
    # verbose=1: prints iteration stats so the terminal is not silent for hours.
    defender_model = PPO(
        policy="MlpPolicy",
        env=def_venv,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=5,
        gamma=0.99,
        ent_coef=0.01,
        device="cpu",   # MlpPolicy: CPU is faster than GPU here
        verbose=1,      # print PPO iteration stats every n_steps
        tensorboard_log=str(out_dir / "tensorboard_defender"),
        seed=seed,
    )

    print(f"  [Gen {generation}] Training defender for {steps:,} steps (k_eval={k_eval}) ...")
    defender_model.learn(total_timesteps=steps, progress_bar=False)
    save_path = str(out_dir / f"defender_gen_{generation}")
    defender_model.save(save_path)
    print(f"  [Gen {generation}] Defender saved → {save_path}.zip")
    def_venv.close()
    return defender_model


# ---------------------------------------------------------------------------
# Phase B: Train Attacker A_k
# ---------------------------------------------------------------------------

def train_attacker(
    generation: int,
    init_attacker: Optional[MaskablePPO],
    defender_model: PPO,
    layout_pool: np.ndarray,
    steps: int,
    out_dir: Path,
    seed: int = BASE_SEED,
    uniform_weight: float = 0.5,
    dk_extract_n: int = 2_000,
) -> MaskablePPO:
    """Train Attacker A_k against D_k (real learned defender) and UNIFORM.

    Mixture: {LearnedLayoutDefender(D_k): 1-uniform_weight, UNIFORM: uniform_weight}

    LearnedLayoutDefender is a serializable defender backed by layouts extracted
    from D_k's policy — safe for SubprocVecEnv workers (no GPU needed there).

    The UNIFORM component is non-negotiable every generation to prevent attacker
    from forgetting the base case.
    """
    dk_weight = 1.0 - uniform_weight

    print(f"  [Gen {generation}] Extracting D_k layout distribution ({dk_extract_n} samples) ...")
    dk_layouts = extract_dk_layouts(defender_model, layout_pool, n_layouts=dk_extract_n, seed=seed)
    dk_defender = LearnedLayoutDefender(dk_layouts)

    defenders   = [UniformRandomDefender(), dk_defender]
    def_weights = [uniform_weight, dk_weight]

    print(f"  [Gen {generation}] Training attacker for {steps:,} steps ...")
    print(f"    Attacker mix: UNIFORM={uniform_weight:.2f}  D_k={dk_weight:.2f}")
    print(f"    D_k unique layouts: {len(dk_layouts)}")

    train_venv = make_attacker_vec_env(defenders, def_weights, base_seed=seed)

    attacker = MaskablePPO(
        env=train_venv,
        **{k: v for k, v in ATTACKER_PPO_KWARGS.items() if k not in ("device",)},
        device="cuda",
        verbose=0,
        seed=seed,
        tensorboard_log=str(out_dir / "tensorboard_attacker"),
    )
    if init_attacker is not None:
        # Warm-start from previous generation — preserves UNIFORM knowledge
        attacker.set_parameters(init_attacker.get_parameters())

    attacker.learn(total_timesteps=steps, progress_bar=False)
    save_path = str(out_dir / f"attacker_gen_{generation}")
    attacker.save(save_path)
    print(f"  [Gen {generation}] Attacker A_{generation} saved → {save_path}.zip")
    train_venv.close()
    return attacker


# ---------------------------------------------------------------------------
# Phase C: Evaluate generation
# ---------------------------------------------------------------------------

def evaluate_generation(
    generation: int,
    attacker_before: MaskablePPO,   # A_{k-1}
    attacker_after:  MaskablePPO,   # A_k
    defender_model:  PPO,
    layout_pool:     np.ndarray,
    out_dir:         Path,
    n_scripted_eps:  int = 100,
    n_learned_eps:   int = 50,
) -> dict:
    """Evaluate per generation.

    Reports:
      - A_{k-1} vs UNIFORM / SPREAD / D_k  (defender hardness baseline)
      - A_k     vs UNIFORM / SPREAD / D_k  (adaptation after training)
      - Δ = A_k_vs_D_k - A_{k-1}_vs_D_k   (negative = attacker improved)
    """
    print(f"  [Gen {generation}] Evaluating A_{{k-1}} (before) vs scripted modes ...")
    before_scripted = evaluate_attacker_scripted(attacker_before, n_episodes=n_scripted_eps)
    before_dk = evaluate_attacker_on_learned_defender(
        attacker_before, defender_model, layout_pool, n_episodes=n_learned_eps
    )

    print(f"  [Gen {generation}] Evaluating A_k (after) vs scripted modes ...")
    after_scripted = evaluate_attacker_scripted(attacker_after, n_episodes=n_scripted_eps)
    after_dk = evaluate_attacker_on_learned_defender(
        attacker_after, defender_model, layout_pool, n_episodes=n_learned_eps
    )

    # Defender adversariality: D_k must be harder than UNIFORM for A_{k-1}
    defender_adversarial = before_dk["mean"] - before_scripted["UNIFORM"]["mean"]
    # Attacker adaptation: A_k improvement over A_{k-1} against D_k
    attacker_adaptation  = after_dk["mean"] - before_dk["mean"]  # negative = improvement
    # UNIFORM drift: A_k vs UNIFORM relative to A_{k-1} vs UNIFORM
    uniform_drift = after_scripted["UNIFORM"]["mean"] - before_scripted["UNIFORM"]["mean"]

    # Exploitability proxy relative to UNIFORM
    exploit_proxy = (after_dk["mean"] - after_scripted["UNIFORM"]["mean"]) / \
                    max(after_scripted["UNIFORM"]["mean"], 1.0)

    results = {
        "generation":            generation,
        "before": {
            "scripted_modes":    before_scripted,
            "vs_D_k":            before_dk,
        },
        "after": {
            "scripted_modes":    after_scripted,
            "vs_D_k":            after_dk,
        },
        "checks": {
            "defender_adversarial":  defender_adversarial,   # >0 = D_k harder than UNIFORM
            "attacker_adaptation":   attacker_adaptation,    # <0 = improvement
            "uniform_drift":         uniform_drift,          # ~0 = UNIFORM preserved
            "exploitability_proxy":  exploit_proxy,
            "delta_spread_vs_uniform": after_scripted["SPREAD"]["mean"] - \
                                        after_scripted["UNIFORM"]["mean"],
        },
    }

    out_path = out_dir / f"eval_gen_{generation}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    un_b = before_scripted["UNIFORM"]["mean"]
    sp_b = before_scripted["SPREAD"]["mean"]
    dk_b = before_dk["mean"]
    un_a = after_scripted["UNIFORM"]["mean"]
    sp_a = after_scripted["SPREAD"]["mean"]
    dk_a = after_dk["mean"]

    print(f"""\n  --- Gen {generation} summary ---""")
    print(f"  {'':20s}  {'UNIFORM':>8}  {'SPREAD':>8}  {'vs_D_k':>8}")
    print(f"  {'A_{{k-1}} (before)':20s}  {un_b:>8.1f}  {sp_b:>8.1f}  {dk_b:>8.1f}")
    print(f"  {'A_k (after)':20s}  {un_a:>8.1f}  {sp_a:>8.1f}  {dk_a:>8.1f}")
    print(f"  {'delta':20s}  {un_a-un_b:>+8.1f}  {sp_a-sp_b:>+8.1f}  {dk_a-dk_b:>+8.1f}")
    print()
    adv_flag = 'OK' if defender_adversarial > 0 else 'WARN: D_k not harder than UNIFORM'
    ada_flag = 'OK' if attacker_adaptation < 0 else 'WARN: attacker did not improve vs D_k'
    uni_flag = 'OK' if abs(uniform_drift) < 3 else f'WARN: UNIFORM drift={uniform_drift:+.1f}'
    print(f"  [check A] defender_adversarial={defender_adversarial:+.1f}  ({adv_flag})")
    print(f"  [check B] attacker_adaptation={attacker_adaptation:+.1f}  ({ada_flag})")
    print(f"  [check C] uniform_drift={uniform_drift:+.1f}  ({uni_flag})")
    print(f"  [eval saved] {out_path}")
    return results


# ---------------------------------------------------------------------------
# Stage 1 auto-selector: pick regime with best SPREAD, tie-break by UNIFORM
# ---------------------------------------------------------------------------

ST1_ROOT = Path("results/training/stage1")


def select_best_stage1_model() -> Path:
    """Pick the Stage 1 model with lowest SPREAD mean (tie-break: UNIFORM mean).

    Preference order for eval file:
      1. final_eval_corrected.json  (mask-aware, authoritative)
      2. final_eval.json            (fallback)

    Corruption guard: if any mode mean ≤ 5 the file is treated as invalid
    (likely produced by the broken no-mask eval that gave mean≈2) and skipped.
    """
    CORRUPTION_THRESHOLD = 5.0

    candidates = []
    for regime_dir in sorted(ST1_ROOT.iterdir()):
        if not regime_dir.is_dir():
            continue
        model_zip = regime_dir / "final_model.zip"
        if not model_zip.exists():
            continue

        # Prefer corrected eval; fall back to original
        eval_file = regime_dir / "final_eval_corrected.json"
        if not eval_file.exists():
            eval_file = regime_dir / "final_eval.json"
        if not eval_file.exists():
            continue

        data = json.loads(eval_file.read_text())
        spread_mean  = data.get("SPREAD",  {}).get("mean", float("inf"))
        uniform_mean = data.get("UNIFORM", {}).get("mean", float("inf"))

        # Corruption guard: suspicious mean indicates mask-broken eval
        if spread_mean <= CORRUPTION_THRESHOLD or uniform_mean <= CORRUPTION_THRESHOLD:
            print(f"  [auto-select] Regime {regime_dir.name}: eval looks corrupted "
                  f"(SPREAD={spread_mean:.2f}, UNIFORM={uniform_mean:.2f}) — skipping")
            continue

        src = eval_file.name
        candidates.append((spread_mean, uniform_mean, regime_dir.name, model_zip, src))

    if not candidates:
        raise FileNotFoundError(
            f"No valid (non-corrupted) eval files found under {ST1_ROOT}. "
            "Re-run eval with masks, or pass --init_attacker explicitly."
        )

    # Sort: primary = SPREAD mean ascending, secondary = UNIFORM mean ascending
    candidates.sort(key=lambda x: (x[0], x[1]))
    best = candidates[0]
    print(f"\n[auto-select] Best Stage 1 regime: {best[2]} (from {best[4]})")
    print(f"  SPREAD mean={best[0]:.2f}  UNIFORM mean={best[1]:.2f}")
    print(f"  Model: {best[3]}")
    for s, u, name, _, src in candidates:
        marker = " ← selected" if name == best[2] else ""
        print(f"  Regime {name}: SPREAD={s:.2f}  UNIFORM={u:.2f}  [{src}]{marker}")
    return best[3]


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main IBR loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Iterative Best Response training loop."
    )
    parser.add_argument(
        "--init_attacker", default=None,
        help="Path to Stage 1 attacker checkpoint (.zip). "
             "If omitted, auto-selects the regime with lowest SPREAD mean "
             "from results/training/stage1/*/final_eval.json.",
    )
    parser.add_argument("--generations",     type=int, default=3)
    parser.add_argument("--attacker_steps",  type=int, default=1_000_000)
    parser.add_argument("--defender_steps",  type=int, default=30_000,
                        help="PPO steps for defender training per generation. "
                             "Keep small (30k-50k) — each step runs k_eval attacker rollouts.")
    parser.add_argument("--k_eval",          type=int, default=1,
                        help="Attacker rollouts per defender env step (default 1). "
                             "Higher = more signal but O(k_eval) slower defender training.")
    parser.add_argument("--pool_size",       type=int, default=POOL_SIZE)
    parser.add_argument("--seed",            type=int, default=BASE_SEED)
    parser.add_argument("--n_eval_eps",      type=int, default=100)
    args = parser.parse_args()

    out_dir = Path("results/training/stage2")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve which Stage 1 model to use
    if args.init_attacker is None:
        init_path = select_best_stage1_model()
    else:
        init_path = Path(args.init_attacker)
        if not init_path.exists():
            raise FileNotFoundError(f"--init_attacker not found: {init_path}")

    # Write run metadata for reproducibility
    meta = {
        "git_hash":      _git_hash(),
        "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%S"),
        "init_attacker": str(init_path),
        "args":          vars(args),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[run_meta] {out_dir / 'run_meta.json'}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build layout pool once, reuse across all generations
    print(f"\nBuilding layout pool ({args.pool_size:,} layouts) ...")
    t0 = time.time()
    layout_pool = build_layout_pool(args.pool_size, board_size=BOARD_SIZE,
                                    ships=SHIPS, seed=args.seed)
    print(f"Done in {time.time() - t0:.1f}s  shape={layout_pool.shape}")

    # Load initial attacker A_0
    print(f"\nLoading A_0 from {init_path} ...")
    dummy_env = make_attacker_vec_env([UniformRandomDefender()], [1.0], base_seed=args.seed)
    attacker = MaskablePPO.load(str(init_path), env=dummy_env, device="cuda")
    dummy_env.close()

    learned_defenders: List[PPO] = []
    all_eval_results = []

    print(f"\n{'='*60}")
    print(f"  IBR Training: {args.generations} generations")
    print(f"  Attacker budget: {args.attacker_steps:,} steps/gen")
    print(f"  Defender budget: {args.defender_steps:,} steps/gen")
    print(f"{'='*60}\n")

    for gen in range(1, args.generations + 1):
        t_gen = time.time()
        print(f"\n{'—'*40}")
        print(f" Generation {gen}/{args.generations}")
        print(f"{'—'*40}")

        # === Phase A: Train Defender ===
        defender_model = train_defender(
            generation=gen,
            frozen_attacker=attacker,
            layout_pool=layout_pool,
            steps=args.defender_steps,
            out_dir=out_dir,
            max_generations=args.generations,
            seed=args.seed + gen * 1000,
            k_eval=args.k_eval,
        )
        learned_defenders.append(defender_model)

        # === Phase B: Train Attacker — trains against REAL D_k ===
        attacker_before = attacker   # keep ref for before/after eval
        attacker = train_attacker(
            generation=gen,
            init_attacker=attacker,        # warm-start from previous gen
            defender_model=defender_model,  # actual D_k (not scripted proxy)
            layout_pool=layout_pool,
            steps=args.attacker_steps,
            out_dir=out_dir,
            seed=args.seed + gen * 2000,
        )

        # === Phase C: Evaluate — before (A_{k-1}) and after (A_k) ===
        eval_row = evaluate_generation(
            generation=gen,
            attacker_before=attacker_before,
            attacker_after=attacker,
            defender_model=defender_model,
            layout_pool=layout_pool,
            out_dir=out_dir,
            n_scripted_eps=args.n_eval_eps,
            n_learned_eps=50,
        )
        all_eval_results.append(eval_row)

        elapsed = time.time() - t_gen
        print(f"  Generation {gen} done in {elapsed/60:.1f} min")

    # Save combined summary
    summary_path = out_dir / "ibr_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_eval_results, f, indent=2)
    print(f"\nIBR complete. Summary → {summary_path.resolve()}")

    # Print final table
    print("\n=== FINAL IBR RESULTS ===")
    print(f"{'Gen':>5}  {'A_k UNIFORM':>11}  {'A_k SPREAD':>10}  {'A_k vs D_k':>10}  {'UNIFORM drift':>13}  {'Exploit':>9}")
    for r in all_eval_results:
        a = r["after"]
        chk = r["checks"]
        print(
            f"  {r['generation']:>3}  "
            f"{a['scripted_modes']['UNIFORM']['mean']:>11.1f}  "
            f"{a['scripted_modes']['SPREAD']['mean']:>10.1f}  "
            f"{a['vs_D_k']['mean']:>10.1f}  "
            f"{chk['uniform_drift']:>+13.1f}  "
            f"{chk['exploitability_proxy']:>+9.3f}"
        )


if __name__ == "__main__":
    main()
