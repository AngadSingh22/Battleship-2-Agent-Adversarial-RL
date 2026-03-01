"""
Train an adversarial Battleship attacker policy under three distinct regimes:

  A - UNIFORM only
  B - Fixed defender mixture (UNIFORM + EDGE + PARITY + SPREAD, equal weights)
  C - IBR-lite alternation between UNIFORM and SPREAD every N PPO updates

Usage:
  PYTHONPATH=. .venv/bin/python training/train_attacker.py --regime A
  PYTHONPATH=. .venv/bin/python training/train_attacker.py --regime B
  PYTHONPATH=. .venv/bin/python training/train_attacker.py --regime C --ibr_switch_n 10

Each regime produces:
  results/training/<regime>/final_model.zip
  results/training/<regime>/checkpoints/         (every 50k steps)
  results/training/<regime>/eval_log.json        (mean + p90 per defender mode)
  results/training/<regime>/tensorboard/
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

# ---- Performance knobs (must come before any SB3/gym imports) ---------------
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
# -----------------------------------------------------------------------------

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from battleship_rl.agents.defender import (
    ClusteredDefender,
    EdgeBiasedDefender,
    ParityDefender,
    SpreadDefender,
    UniformRandomDefender,
)
from battleship_rl.agents.policies import BattleshipFeatureExtractor
from battleship_rl.envs.battleship_env import BattleshipEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFENDER_MAP = {
    "UNIFORM": UniformRandomDefender,
    "EDGE":    EdgeBiasedDefender,
    "CLUSTER": ClusteredDefender,
    "SPREAD":  SpreadDefender,
    "PARITY":  ParityDefender,
}

BASE_SEED   = 42
N_ENVS      = 16
N_STEPS     = 2048
BATCH_SIZE  = 1024
N_EPOCHS    = 5
TOTAL_STEPS = 2_000_000
EVAL_FREQ   = 200_000      # default: eval every 200k (not 50k) to save time
N_EVAL_EPS  = 30           # default: 30 eps/mode for intermediate evals
FINAL_EVAL_EPS = 200       # full eval at end: 200 eps per mode, all 5 modes
ALL_MODES   = ["UNIFORM", "EDGE", "CLUSTER", "SPREAD", "PARITY"]
FAST_MODES  = ["UNIFORM", "SPREAD"]  # default intermediate eval modes


def mask_fn(env):
    return env.get_action_mask()


def make_single_env(defenders, weights, seed: int, debug: bool = False):
    """Returns a thunk of a fully-seeded BattleshipEnv + ActionMasker."""
    def _init():
        env = BattleshipEnv(
            defenders=defenders,
            defender_weights=weights,
            debug=debug,
        )
        env = ActionMasker(env, mask_fn)
        env.reset(seed=seed)
        return env
    return _init


def make_training_vec_env(defenders, weights, base_seed=BASE_SEED, n_envs=N_ENVS):
    fns = [make_single_env(defenders, weights, base_seed + rank) for rank in range(n_envs)]
    venv = SubprocVecEnv(fns)
    return VecMonitor(venv)


def make_eval_env_for_defender(defender_cls, seed: int):
    """Fixed single-defender env for deterministic evaluation (no mixture)."""
    def _init():
        env = BattleshipEnv(defender=defender_cls(), debug=False)
        env = ActionMasker(env, mask_fn)
        env.reset(seed=seed)
        return env
    return _init


# ---------------------------------------------------------------------------
# Evaluation callback: runs full evaluation suite at fixed intervals
# ---------------------------------------------------------------------------

class BenchmarkEvalCallback(BaseCallback):
    """Evaluates policy on selected defender modes; writes JSON eval_log."""

    def __init__(
        self,
        eval_freq: int,
        n_episodes: int,
        log_dir: Path,
        eval_modes: List[str],
        n_envs: int = 4,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_freq  = eval_freq
        self.n_episodes = n_episodes
        self.log_dir    = log_dir
        self.eval_modes = eval_modes
        self.n_envs     = n_envs
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path   = log_dir / "eval_log.json"
        self.results    = []

    def _on_step(self) -> bool:
        if self.n_calls % max(1, self.eval_freq // N_ENVS) == 0:
            ts  = self.num_timesteps
            row = {"timesteps": ts, "modes": {}}
            for name in self.eval_modes:
                cls = DEFENDER_MAP[name]
                eval_fns = [
                    make_eval_env_for_defender(cls, seed=BASE_SEED + 9000 + i)
                    for i in range(self.n_envs)
                ]
                eval_venv = SubprocVecEnv(eval_fns)
                eval_venv = VecMonitor(eval_venv)

                ep_lengths:   list = []
                trunc_count:  int  = 0
                total_eps:    int  = 0

                n_batches = max(1, self.n_episodes // self.n_envs)
                for _ in range(n_batches):
                    obs = eval_venv.reset()
                    dones   = [False] * self.n_envs
                    steps   = [0]     * self.n_envs
                    truncs  = [False] * self.n_envs
                    while not all(dones):
                        # KEY FIX: pass action masks so MaskablePPO picks valid cells
                        masks = get_action_masks(eval_venv)
                        actions, _ = self.model.predict(
                            obs, deterministic=True, action_masks=masks
                        )
                        obs, _, new_dones, infos = eval_venv.step(actions)
                        for i, (d, nd) in enumerate(zip(dones, new_dones)):
                            if not d:
                                steps[i] += 1
                                if nd:
                                    ep_lengths.append(steps[i])
                                    # SB3 sets TimeLimit.truncated for truncation
                                    if infos[i].get("TimeLimit.truncated", False):
                                        trunc_count += 1
                                    total_eps += 1
                                    dones[i] = True
                eval_venv.close()

                arr       = np.array(ep_lengths)
                fail_rate = trunc_count / max(total_eps, 1)
                row["modes"][name] = {
                    "mean":      float(arr.mean()),
                    "p90":       float(np.percentile(arr, 90)),
                    "fail_rate": fail_rate,
                }
                if self.verbose:
                    print(f"  [{name:7s}] mean={arr.mean():.1f}  "
                          f"p90={np.percentile(arr, 90):.1f}  "
                          f"fail_rate={fail_rate:.3f}")
                    if fail_rate > 0.01:
                        print(f"    WARNING: fail_rate={fail_rate:.1%} — "
                              f"masks may still be broken")

            self.results.append(row)
            with open(self.log_path, "w") as f:
                json.dump(self.results, f, indent=2)
            if self.verbose and "SPREAD" in row["modes"] and "UNIFORM" in row["modes"]:
                sp = row["modes"]["SPREAD"]
                un = row["modes"]["UNIFORM"]
                print(f"\n[eval @ {ts:,}] UNIFORM={un['mean']:.1f}  "
                      f"SPREAD={sp['mean']:.1f}  Δ={sp['mean']-un['mean']:+.1f}\n")
        return True


# ---------------------------------------------------------------------------
# IBR-lite callback: alternate defender pool every N PPO updates
# ---------------------------------------------------------------------------

class IBRSwitchCallback(BaseCallback):
    """Regime C: alternate defender pool between UNIFORM-only and SPREAD-only
    every switch_n PPO updates by calling set_defender_weights() on all workers.
    Workers apply the new weights on their next episode reset.
    """

    def __init__(self, switch_n: int, verbose: int = 0):
        super().__init__(verbose)
        self.switch_n = switch_n
        self._phase   = 0   # 0 = UNIFORM-only, 1 = SPREAD-only
        self._update  = 0

    def _on_rollout_end(self) -> None:
        self._update += 1
        if self._update % self.switch_n == 0:
            self._phase ^= 1
            # Regime C env has defenders=[UniformRandomDefender, SpreadDefender]
            # weights index 0 = UNIFORM, index 1 = SPREAD
            new_weights = [0.0, 1.0] if self._phase else [1.0, 0.0]
            name = "SpreadDefender" if self._phase else "UniformRandomDefender"
            # Broadcast to all SubprocVecEnv workers; takes effect next reset()
            self.training_env.env_method("set_defender_weights", new_weights)
            if self.verbose:
                print(f"\n[IBR @update {self._update}] Active defender: {name}\n")

    def _on_step(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_regime(regime: str, ibr_switch_n: int):
    if regime == "A":
        defenders = [UniformRandomDefender()]
        weights   = [1.0]
    elif regime == "B":
        defenders = [UniformRandomDefender(), EdgeBiasedDefender(),
                     ParityDefender(),        SpreadDefender()]
        weights   = [0.25, 0.25, 0.25, 0.25]
    elif regime == "C":
        # Start with UNIFORM only; IBRSwitchCallback swaps it at runtime
        defenders = [UniformRandomDefender(), SpreadDefender()]
        weights   = [1.0, 0.0]
    else:
        raise ValueError(f"Unknown regime: {regime!r}")
    return defenders, weights


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _write_run_meta(out_dir: Path, args) -> None:
    """Write git hash + CLI args into run_meta.json for reproducibility."""
    meta = {
        "git_hash":  _git_hash(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "args":      vars(args),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))


def _final_eval(model, out_dir: Path, n_episodes: int, n_envs: int = 4) -> None:
    """Run full 5-mode mask-aware evaluation and write final_eval.json."""
    print("\n[final eval] Running full 5-mode evaluation (mask-aware) ...")
    results = {}
    for name, cls in DEFENDER_MAP.items():
        eval_fns = [
            make_eval_env_for_defender(cls, seed=BASE_SEED + 8000 + i)
            for i in range(n_envs)
        ]
        eval_venv = VecMonitor(SubprocVecEnv(eval_fns))
        ep_lengths:  list = []
        trunc_count: int  = 0
        total_eps:   int  = 0

        n_batches = max(1, n_episodes // n_envs)
        for _ in range(n_batches):
            obs = eval_venv.reset()
            dones  = [False] * n_envs
            steps  = [0]     * n_envs
            while not all(dones):
                # KEY FIX: pass action masks so MaskablePPO picks valid cells
                masks = get_action_masks(eval_venv)
                actions, _ = model.predict(
                    obs, deterministic=True, action_masks=masks
                )
                obs, _, new_dones, infos = eval_venv.step(actions)
                for i, (d, nd) in enumerate(zip(dones, new_dones)):
                    if not d:
                        steps[i] += 1
                        if nd:
                            ep_lengths.append(steps[i])
                            if infos[i].get("TimeLimit.truncated", False):
                                trunc_count += 1
                            total_eps += 1
                            dones[i] = True
        eval_venv.close()

        arr       = np.array(ep_lengths)
        fail_rate = trunc_count / max(total_eps, 1)
        results[name] = {
            "mean":      float(arr.mean()),
            "p90":       float(np.percentile(arr, 90)),
            "fail_rate": fail_rate,
        }
        flag = "  ⚠ fail_rate > 0" if fail_rate > 0 else ""
        print(f"  [{name:7s}] mean={arr.mean():.1f}  p90={np.percentile(arr, 90):.1f}  "
              f"fail_rate={fail_rate:.3f}{flag}")
    (out_dir / "final_eval.json").write_text(json.dumps(results, indent=2))
    print(f"  Saved → {out_dir / 'final_eval.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime",       choices=["A", "B", "C"], required=True)
    parser.add_argument("--total_steps",  type=int, default=TOTAL_STEPS)
    parser.add_argument("--n_envs",       type=int, default=N_ENVS)
    parser.add_argument("--ibr_switch_n", type=int, default=10,
                        help="(Regime C) Switch defender every N PPO updates")
    parser.add_argument("--seed",         type=int, default=BASE_SEED)
    # Eval cost controls (the main lever for wall-time)
    parser.add_argument("--eval_freq",    type=int, default=EVAL_FREQ,
                        help="Evaluate every N global env steps")
    parser.add_argument("--eval_eps",     type=int, default=N_EVAL_EPS,
                        help="Episodes per defender mode per intermediate eval")
    parser.add_argument("--eval_modes",   nargs="+", default=FAST_MODES,
                        choices=ALL_MODES, metavar="MODE",
                        help="Defender modes for intermediate eval (default: %(default)s)")
    parser.add_argument("--final_eval_eps", type=int, default=FINAL_EVAL_EPS,
                        help="Episodes/mode for final full 5-mode eval")
    args = parser.parse_args()

    # Create output dirs explicitly so tee never fails
    out_dir = Path("results/training/stage1") / args.regime
    out_dir.mkdir(parents=True, exist_ok=True)
    (Path("results/training/stage2")).mkdir(parents=True, exist_ok=True)

    # Write reproducibility metadata before anything else
    _write_run_meta(out_dir, args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    defenders, weights = build_regime(args.regime, args.ibr_switch_n)

    print(f"\n{'='*60}")
    print(f"  Regime {args.regime} | {args.total_steps:,} steps | {args.n_envs} envs | seed={args.seed}")
    print(f"  Defenders: {[type(d).__name__ for d in defenders]}")
    print(f"  Weights:   {weights}")
    print(f"{'='*60}\n")

    # ------ Training env (mixed) --------------------------------------------
    train_venv = make_training_vec_env(
        defenders, weights, base_seed=args.seed, n_envs=args.n_envs
    )

    # ------ Model -----------------------------------------------------------
    policy_kwargs = dict(
        features_extractor_class=BattleshipFeatureExtractor,
        features_extractor_kwargs={"features_dim": 512},
        net_arch=dict(pi=[512, 512], vf=[512, 512]),
    )

    model = MaskablePPO(
        policy="CnnPolicy",
        env=train_venv,
        learning_rate=3e-4,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(out_dir / "tensorboard"),
        device="cuda",
        verbose=1,
        seed=args.seed,
    )

    # ------ Callbacks -------------------------------------------------------
    callbacks = []

    # Checkpoint every 200k steps
    callbacks.append(CheckpointCallback(
        save_freq=200_000 // args.n_envs,
        save_path=str(out_dir / "checkpoints"),
        name_prefix="model",
    ))

    # Full benchmark eval at --eval_freq; only --eval_modes defenders
    callbacks.append(BenchmarkEvalCallback(
        eval_freq=args.eval_freq,
        n_episodes=args.eval_eps,
        eval_modes=args.eval_modes,
        log_dir=out_dir,
        verbose=1,
    ))

    # IBR alternation (Regime C only)
    if args.regime == "C":
        callbacks.append(IBRSwitchCallback(
            switch_n=args.ibr_switch_n, verbose=1
        ))

    # ------ Train -----------------------------------------------------------
    t0 = time.time()
    model.learn(
        total_timesteps=args.total_steps,
        callback=callbacks,
        progress_bar=False,
    )
    elapsed = time.time() - t0

    model.save(str(out_dir / "final_model"))
    train_venv.close()

    # Full 5-mode final eval → final_eval.json (used by auto-selector in train_ibr.py)
    _final_eval(model, out_dir, n_episodes=args.final_eval_eps)

    print(f"\nRegime {args.regime} done in {elapsed/3600:.1f}h  "
          f"({args.total_steps / elapsed:.0f} steps/sec)")
    print(f"Artifacts: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
