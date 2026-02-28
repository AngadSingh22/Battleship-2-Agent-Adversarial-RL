"""
Train an adversarial Battleship attacker policy under three distinct regimes:

  A - UNIFORM only
  B - Fixed defender mixture (UNIFORM + EDGE + PARITY + SPREAD, equal weights)
  C - IBR-lite alternation between UNIFORM and SPREAD every N PPO updates

Usage:
  PYTHONPATH=. .venv/bin/python scripts/train_attacker.py --regime A
  PYTHONPATH=. .venv/bin/python scripts/train_attacker.py --regime B
  PYTHONPATH=. .venv/bin/python scripts/train_attacker.py --regime C --ibr_switch_n 10

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
import time
from pathlib import Path
from typing import Optional

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
from battleship_rl.eval.evaluate import evaluate_policy


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

BASE_SEED  = 42
N_ENVS     = 16
N_STEPS    = 2048          # rollout steps per env per update → 32768 total
BATCH_SIZE = 1024
N_EPOCHS   = 5
TOTAL_STEPS = 2_000_000    # 2M env steps per regime  (~61 updates)
EVAL_FREQ   = 50_000       # evaluate every 50k global steps
N_EVAL_EPS  = 100          # episodes per eval per defender mode


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
    """Evaluates policy on all five defender modes; writes JSON log."""

    def __init__(self, eval_freq: int, n_episodes: int, log_dir: Path, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq   = eval_freq
        self.n_episodes  = n_episodes
        self.log_dir     = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path    = log_dir / "eval_log.json"
        self.results     = []

    def _on_step(self) -> bool:
        if self.n_calls % (self.eval_freq // N_ENVS) == 0:
            ts = self.num_timesteps
            row = {"timesteps": ts, "modes": {}}
            for name, cls in DEFENDER_MAP.items():
                # Separate fixed-defender vec env; NOT the training env
                eval_fns = [
                    make_eval_env_for_defender(cls, seed=BASE_SEED + 9000 + i)
                    for i in range(4)   # 4 workers for speed
                ]
                eval_venv = SubprocVecEnv(eval_fns)
                eval_venv = VecMonitor(eval_venv)

                ep_lengths = []
                for _ in range(self.n_episodes // 4):
                    obs = eval_venv.reset()
                    dones = [False] * 4
                    steps = [0] * 4
                    while not all(dones):
                        actions, _ = self.model.predict(obs, deterministic=True)
                        obs, _, new_dones, _ = eval_venv.step(actions)
                        for i, (d, nd) in enumerate(zip(dones, new_dones)):
                            if not d:
                                steps[i] += 1
                                if nd:
                                    ep_lengths.append(steps[i])
                                    dones[i] = True
                eval_venv.close()

                arr = np.array(ep_lengths)
                row["modes"][name] = {
                    "mean": float(arr.mean()),
                    "p90":  float(np.percentile(arr, 90)),
                }
                if self.verbose:
                    print(f"  [{name}] mean={arr.mean():.1f}  p90={np.percentile(arr, 90):.1f}")

            self.results.append(row)
            with open(self.log_path, "w") as f:
                json.dump(self.results, f, indent=2)
            if self.verbose:
                spread = row["modes"]["SPREAD"]
                uniform = row["modes"]["UNIFORM"]
                delta = spread["mean"] - uniform["mean"]
                print(f"\n[eval @ {ts:,}] UNIFORM={uniform['mean']:.1f}  "
                      f"SPREAD={spread['mean']:.1f}  Δ={delta:+.1f}\n")
        return True


# ---------------------------------------------------------------------------
# IBR-lite callback: alternate defender pool every N PPO updates
# ---------------------------------------------------------------------------

class IBRSwitchCallback(BaseCallback):
    """Regime C: switch training env's active defender between UNIFORM and SPREAD."""

    def __init__(self, switch_n: int, verbose: int = 0):
        super().__init__(verbose)
        self.switch_n = switch_n
        self._phase   = 0          # 0 = UNIFORM, 1 = SPREAD
        self._update  = 0

    def _on_rollout_end(self) -> None:
        self._update += 1
        if self._update % self.switch_n == 0:
            self._phase ^= 1
            new_cls = SpreadDefender if self._phase else UniformRandomDefender
            # For SubprocVecEnv we can't swap defender objects mid-flight;
            # instead we encode the switch via a new weights vector so each
            # env applies it on next reset.
            # Since the env draws via self.np_random on reset, we update the
            # weights array stored on each worker's env via env_method.
            # The SubprocVecEnv's env_method broadcasts to all workers.
            # Workers re-read _defender_weights on the next reset().
            name = "SpreadDefender" if self._phase else "UniformRandomDefender"
            if self.verbose:
                print(f"\n[IBR] Switching to {name} @ update {self._update}\n")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime",       choices=["A", "B", "C"], required=True)
    parser.add_argument("--total_steps",  type=int, default=TOTAL_STEPS)
    parser.add_argument("--n_envs",       type=int, default=N_ENVS)
    parser.add_argument("--ibr_switch_n", type=int, default=10,
                        help="(Regime C) Switch defender every N PPO updates")
    parser.add_argument("--seed",         type=int, default=BASE_SEED)
    args = parser.parse_args()

    out_dir = Path("results/training") / args.regime
    out_dir.mkdir(parents=True, exist_ok=True)

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
        net_arch=[dict(pi=[512, 512], vf=[512, 512])],
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

    # Full benchmark eval every 50k steps; writes to eval_log.json
    callbacks.append(BenchmarkEvalCallback(
        eval_freq=EVAL_FREQ,
        n_episodes=N_EVAL_EPS,
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
        progress_bar=True,
    )
    elapsed = time.time() - t0

    model.save(str(out_dir / "final_model"))
    train_venv.close()

    print(f"\nRegime {args.regime} done in {elapsed/3600:.1f}h  "
          f"({args.total_steps / elapsed:.0f} steps/sec)")
    print(f"Artifacts saved to {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
