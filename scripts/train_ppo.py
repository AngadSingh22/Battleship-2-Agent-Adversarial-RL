#!/usr/bin/env python3
"""
Training script for Battleship RL (Phase 2).
Launches MaskablePPO training using configs/env.yaml and configs/ppo.yaml.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from battleship_rl.agents.sb3_train import train, _load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MaskablePPO on Battleship.")
    parser.add_argument("--total-timesteps", type=int, default=500_000, help="Total steps.")
    parser.add_argument("--num-envs", type=int, default=8, help="Parallel environments.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--env-config", default="configs/env.yaml", help="Path to env config.")
    parser.add_argument("--ppo-config", default="configs/ppo.yaml", help="Path to PPO config.")
    parser.add_argument("--save-path", default="runs/battleship_baseline", help="Model save path.")
    parser.add_argument("--defender-path", default=None, help="Path to adversarial defender model.")
    args = parser.parse_args()

    print(f"Loading env config from {args.env_config}...")
    env_config = _load_yaml(args.env_config)
    
    print(f"Loading PPO config from {args.ppo_config}...")
    ppo_config = _load_yaml(args.ppo_config)

    print(f"Starting training: {args.total_timesteps} steps, {args.num_envs} envs, Seed {args.seed}")
    model = train(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        seed=args.seed,
        env_config=env_config,
        ppo_config=ppo_config,
        save_path=args.save_path,
        defender_path=args.defender_path,
    )
    
    print(f"Training complete. Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
