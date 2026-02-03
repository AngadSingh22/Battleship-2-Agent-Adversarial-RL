#!/usr/bin/env python3
"""
Training script for Adversarial Defender (Phase 2).
Trains an RL agent on BattleshipPlacementEnv to generate hard layouts.
"""

import argparse
from pathlib import Path
import sys
import yaml
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from battleship_rl.envs.placement_env import BattleshipPlacementEnv
from battleship_rl.agents.policies import BattleshipCnnPolicy


def _load_yaml(path: str | None) -> dict:
    if path is None:
        return {}
    data = Path(path)
    if not data.exists():
        return {}
    with data.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def make_env(rank: int, seed: int, env_config: dict, attacker_path: str | None = None):
    def _init():
        # Clean config for placement env (only needs board/ships)
        board_size = env_config.get("board_size", 10)
        ships = env_config.get("ships", [5, 4, 3, 3, 2])
        ship_config = env_config.get("ship_config")
        if ship_config:
            ships = list(ship_config.values())
            
        env = BattleshipPlacementEnv(
            board_size=board_size, 
            ships=ships,
            attacker_path=attacker_path
        )
        env = ActionMasker(env, lambda e: e.get_action_mask())
        env.reset(seed=seed + rank)
        return env
    return _init


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Adversarial Defender.")
    parser.add_argument("--total-timesteps", type=int, default=100_000, help="Total steps.")
    parser.add_argument("--num-envs", type=int, default=4, help="Parallel environments.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--env-config", default="configs/env.yaml", help="Path to env config.")
    parser.add_argument("--ppo-config", default="configs/ppo.yaml", help="Path to PPO config.")
    parser.add_argument("--save-path", default="runs/defender_ppo", help="Model save path.")
    parser.add_argument("--attacker-path", default=None, help="Path to trained Attacker PPO model.")
    args = parser.parse_args()

    env_config = _load_yaml(args.env_config)
    ppo_config = _load_yaml(args.ppo_config)

    print(f"Starting Defender training: {args.total_timesteps} steps, {args.num_envs} envs")
    
    set_random_seed(args.seed)
    env_fns = [make_env(rank, args.seed, env_config, args.attacker_path) for rank in range(args.num_envs)]
    vec_env = SubprocVecEnv(env_fns)

    model = MaskablePPO(
        policy=BattleshipCnnPolicy, # Reuse CNN policy
        env=vec_env,
        **ppo_config,
    )
    
    model.learn(total_timesteps=args.total_timesteps)
    model.save(args.save_path)
    print(f"Defender training complete. Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
