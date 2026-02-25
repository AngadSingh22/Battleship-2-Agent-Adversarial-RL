import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from battleship_rl.envs.placement_env import BattleshipPlacementEnv


class DefenderEarlyStop(BaseCallback):
    """Stop defender training when ep_rew_mean stops improving.

    Checks after every rollout collection. Stops when:
      - At least min_timesteps have been used (warm-up guard), AND
      - The best ep_rew_mean has not improved by > min_delta
        for `patience` consecutive rollout checks.

    Also monitors entropy_loss: if entropy is still falling rapidly,
    we add 1 extra check before stopping (policy may still be diversifying).
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.3,
        min_timesteps: int = 100_000,
        verbose: int = 1,
        save_path: str = None,
    ):
        super().__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.min_timesteps = min_timesteps
        self.save_path = save_path
        self._best_mean = -np.inf
        self._no_improve_count = 0
        self._last_entropy = None

    def _on_step(self) -> bool:
        # Return False to abort training if flagged.
        return not getattr(self.model, "stop_training", False)

    def _on_rollout_end(self) -> None:
        # Warm-up guard: don't stop too early
        if self.num_timesteps < self.min_timesteps:
            return

        # Need episode data to evaluate
        if len(self.model.ep_info_buffer) == 0:
            return

        # --- Mean episode reward (= shots defender survives) ---
        mean_rew = float(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))

        # --- Entropy from logger (indicates policy confidence) ---
        current_entropy = None
        try:
            current_entropy = self.model.logger.name_to_value.get("train/entropy_loss")
        except Exception:
            pass

        # Check for meaningful entropy drop (policy still actively changing)
        entropy_dropping = False
        if current_entropy is not None and self._last_entropy is not None:
            entropy_dropping = (self._last_entropy - current_entropy) > 0.1
        self._last_entropy = current_entropy

        if mean_rew > self._best_mean + self.min_delta:
            self._best_mean = mean_rew
            self._no_improve_count = 0
            if self.verbose >= 2:
                print(f"[EarlyStop] New best ep_rew_mean: {self._best_mean:.2f}")
        else:
            self._no_improve_count += 1
            if self.verbose >= 1:
                print(
                    f"[EarlyStop] No reward improvement "
                    f"({self._no_improve_count}/{self.patience}). "
                    f"Best: {self._best_mean:.2f}, Current: {mean_rew:.2f}"
                    + (f", entropy still dropping" if entropy_dropping else "")
                )

        # If entropy is still actively dropping, give one extra round
        effective_patience = self.patience + (1 if entropy_dropping else 0)

        if self._no_improve_count >= effective_patience:
            steps_saved = (
                self.model._total_timesteps - self.num_timesteps
            )
            print(
                f"\n[EarlyStop] Stopping defender at {self.num_timesteps:,} steps "
                f"(saved ~{steps_saved:,} steps). "
                f"Best ep_rew_mean: {self._best_mean:.2f}\n"
            )
            if self.save_path:
                self.model.save(self.save_path)
                print(f"[EarlyStop] Force-saved model to {self.save_path}.zip before halting.")
            self.model.stop_training = True


def make_env(rank, attacker_path):
    def _init():
        env = BattleshipPlacementEnv(attacker_model=attacker_path)
        env = ActionMasker(env, lambda e: e.action_masks())
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train adversarial defender (BattleshipPlacementEnv).")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--attacker-path", type=str, required=True, help="Path to opponent attacker model zip")
    parser.add_argument("--save-path", type=str, required=True, help="Path to save defender model (no extension)")
    parser.add_argument("--load-path", type=str, default=None,
                        help="Warm-start defender from this checkpoint zip (previous gen)")
    parser.add_argument("--tensorboard-log", type=str, default="runs/tb_logs", help="TensorBoard log directory")
    # Early stopping
    parser.add_argument("--early-stop-patience", type=int, default=5,
                        help="Rollout checks with no improvement before stopping (0 = disabled)")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.3,
                        help="Minimum ep_rew_mean improvement to reset patience counter")
    parser.add_argument("--early-stop-min-steps", type=int, default=100_000,
                        help="Minimum timesteps before early stopping can trigger")
    args = parser.parse_args()

    # Create VecEnv
    env = make_vec_env(
        make_env(0, args.attacker_path),
        n_envs=args.num_envs,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"},
    )

    # Opt 1: Warm-start from previous defender generation if available
    if args.load_path and Path(args.load_path).exists():
        print(f"Warm-starting defender from {args.load_path}")
        model = MaskablePPO.load(args.load_path, env=env,
                                 tensorboard_log=args.tensorboard_log)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=args.tensorboard_log,
            ent_coef=0.01,
            learning_rate=3e-4,
            batch_size=64,
            gamma=0.99,
        )

    callbacks = []
    if args.early_stop_patience > 0:
        callbacks.append(
            DefenderEarlyStop(
                patience=args.early_stop_patience,
                min_delta=args.early_stop_min_delta,
                min_timesteps=args.early_stop_min_steps,
                verbose=1,
                save_path=args.save_path,
            )
        )

    print(f"Training Defender against Attacker: {args.attacker_path}")
    if callbacks:
        print(
            f"Early stopping: patience={args.early_stop_patience}, "
            f"min_delta={args.early_stop_min_delta}, "
            f"min_steps={args.early_stop_min_steps:,}"
        )

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks or None)

    # Save
    model.save(args.save_path)
    print(f"Saved Defender model to {args.save_path}.zip")
    env.close()


if __name__ == "__main__":
    main()
