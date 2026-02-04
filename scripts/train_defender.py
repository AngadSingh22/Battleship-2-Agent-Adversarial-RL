import argparse
import sys
from pathlib import Path
import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from battleship_rl.envs.placement_env import BattleshipPlacementEnv

def make_env(rank, attacker_path):
    def _init():
        env = BattleshipPlacementEnv(attacker_model=attacker_path)
        env = ActionMasker(env, lambda e: e.action_masks()) # Helper wrapper
        # Seed logic is handled by gym typically, or explicitly
        # env.reset(seed=rank) 
        return env
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--attacker-path", type=str, required=True, help="Path to opponent attacker model zip")
    parser.add_argument("--save-path", type=str, required=True, help="Path to save defender model (no extension)")
    args = parser.parse_args()

    # Create VecEnv
    env = make_vec_env(
        make_env(0, args.attacker_path),
        n_envs=args.num_envs,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"}
    )

    model = MaskablePPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        ent_coef=0.01, # Encourage exploration in placement
        learning_rate=3e-4,
        batch_size=64,
        gamma=0.99, # Discount factor not super critical as episodes are short (5 steps), but set for stability
    )

    print(f"Training Defender against Attacker: {args.attacker_path}")
    model.learn(total_timesteps=args.total_timesteps)
    
    # Save
    model.save(args.save_path)
    print(f"Saved Defender model to {args.save_path}.zip")
    env.close()

if __name__ == "__main__":
    main()
