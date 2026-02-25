from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import yaml
from stable_baselines3.common.callbacks import BaseCallback

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from battleship_rl.agents.policies import BattleshipCnnPolicy
from battleship_rl.envs.battleship_env import BattleshipEnv


def _load_yaml(path: str | None) -> dict:
    if path is None:
        return {}
    data = Path(path)
    if not data.exists():
        return {}
    with data.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _mask_fn(env: BattleshipEnv):
    return env.get_action_mask()


def make_env(rank: int, seed: int, env_config: dict, defender_path: str | None = None) -> Callable[[], BattleshipEnv]:
    def _init():
        # Handle defender loading
        defender = None
        if defender_path:
            try:
                from battleship_rl.agents.defender import AdversarialDefender
                defender = AdversarialDefender(model_path=defender_path)
            except Exception as e:
                print(f"Failed to load defender from {defender_path}: {e}")

        # Pass defender directly so it is set before any reset() call.
        env = BattleshipEnv(config=env_config, defender=defender)
        env = ActionMasker(env, _mask_fn)
        env.reset(seed=seed + rank)
        return env

    return _init


class AttackerEarlyStop(BaseCallback):
    """Stop attacker training when training metrics plateau.

    Uses policy_gradient_loss magnitude and value_loss as convergence signals
    (ep_rew_mean is not available without Monitor wrapper). Stops when:
      - At least min_timesteps have elapsed (warm-up guard), AND
      - |policy_gradient_loss| < pg_loss_threshold for `patience` checks, AND
        value_loss < value_loss_threshold for `patience` checks.

    This catches the point where the policy has converged against the current
    (fixed) defender — any further training is diminishing returns.
    """

    def __init__(
        self,
        patience: int = 5,
        pg_loss_threshold: float = 0.02,
        value_loss_threshold: float = 1.0,
        min_timesteps: int = 200_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.patience = patience
        self.pg_loss_threshold = pg_loss_threshold
        self.value_loss_threshold = value_loss_threshold
        self.min_timesteps = min_timesteps
        self._plateau_count = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.num_timesteps < self.min_timesteps:
            return

        try:
            pg_loss = self.model.logger.name_to_value.get("train/policy_gradient_loss")
            val_loss = self.model.logger.name_to_value.get("train/value_loss")
        except Exception:
            return

        if pg_loss is None or val_loss is None:
            return

        pg_converged = abs(pg_loss) < self.pg_loss_threshold
        val_converged = val_loss < self.value_loss_threshold

        if pg_converged and val_converged:
            self._plateau_count += 1
            if self.verbose >= 1:
                print(
                    f"[AttackerEarlyStop] Plateau check "
                    f"({self._plateau_count}/{self.patience}): "
                    f"pg_loss={pg_loss:.4f}, val_loss={val_loss:.3f}"
                )
        else:
            if self._plateau_count > 0 and self.verbose >= 2:
                print(
                    f"[AttackerEarlyStop] Reset plateau counter. "
                    f"pg_loss={pg_loss:.4f}, val_loss={val_loss:.3f}"
                )
            self._plateau_count = 0

        if self._plateau_count >= self.patience:
            steps_saved = self.model._total_timesteps - self.num_timesteps
            print(
                f"\n[AttackerEarlyStop] Stopping at {self.num_timesteps:,} steps "
                f"(saved ~{steps_saved:,} steps). "
                f"pg_loss={pg_loss:.4f}, val_loss={val_loss:.3f}\n"
            )
            self.model.stop_training = True


def train(
    total_timesteps: int,
    num_envs: int,
    seed: int,
    env_config: dict,
    ppo_config: dict,
    save_path: str | None = None,
    defender_path: str | None = None,
    load_path: str | None = None,
    tensorboard_log: str | None = "runs/tb_logs",
    early_stop: bool = True,
    early_stop_patience: int = 5,
    early_stop_min_steps: int = 200_000,
) -> MaskablePPO:
    set_random_seed(seed)
    env_fns = [make_env(rank, seed, env_config, defender_path) for rank in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)

    # Opt 1: Warm-start from previous generation if available
    if load_path and Path(load_path).exists():
        print(f"Warm-starting attacker from {load_path}")
        model = MaskablePPO.load(load_path, env=vec_env, tensorboard_log=tensorboard_log)
        # Re-apply any ppo_config overrides from yaml (lr etc.) that may differ
        if "learning_rate" in ppo_config:
            model.learning_rate = ppo_config["learning_rate"]
    else:
        model = MaskablePPO(
            policy=BattleshipCnnPolicy,
            env=vec_env,
            tensorboard_log=tensorboard_log,
            **ppo_config,
        )

    # Opt 2: Early stopping callback
    callbacks = []
    if early_stop:
        callbacks.append(
            AttackerEarlyStop(
                patience=early_stop_patience,
                min_timesteps=early_stop_min_steps,
                verbose=1,
            )
        )

    model.learn(total_timesteps=total_timesteps, callback=callbacks or None)

    if save_path:
        model.save(save_path)

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MaskablePPO on Battleship.")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-config", default="configs/env.yaml")
    parser.add_argument("--ppo-config", default="configs/ppo.yaml")
    parser.add_argument("--save-path", default="runs/battleship_maskable_ppo")
    parser.add_argument("--defender-path", type=str, default=None,
                        help="Path to adversarial defender model zip")
    parser.add_argument("--load-path", type=str, default=None,
                        help="Warm-start from this model checkpoint (zip path)")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable attacker early stopping")
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-steps", type=int, default=200_000)
    args = parser.parse_args()

    env_config = _load_yaml(args.env_config)
    ppo_config = _load_yaml(args.ppo_config)

    train(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        seed=args.seed,
        env_config=env_config,
        ppo_config=ppo_config,
        save_path=args.save_path,
        defender_path=args.defender_path,
        load_path=args.load_path,
        early_stop=not args.no_early_stop,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_steps=args.early_stop_min_steps,
    )


if __name__ == "__main__":
    main()
