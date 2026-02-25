from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import yaml

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


def train(
    total_timesteps: int,
    num_envs: int,
    seed: int,
    env_config: dict,
    ppo_config: dict,
    save_path: str | None = None,
    defender_path: str | None = None,
    tensorboard_log: str | None = None,
) -> MaskablePPO:
    set_random_seed(seed)
    env_fns = [make_env(rank, seed, env_config, defender_path) for rank in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)

    model = MaskablePPO(
        policy=BattleshipCnnPolicy,
        env=vec_env,
        tensorboard_log=tensorboard_log,
        **ppo_config,
    )
    model.learn(total_timesteps=total_timesteps)

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
    )


if __name__ == "__main__":
    main()
