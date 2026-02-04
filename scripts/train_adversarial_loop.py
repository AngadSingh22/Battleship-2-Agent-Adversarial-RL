#!/usr/bin/env python3
"""
Phase 4: Adversarial Training Loop.
Iteratively trains Attacker and Defender against each other.
"""
import argparse
import subprocess
import sys
import shutil
from pathlib import Path

import os

def run_command(cmd, cwd=None):
    print(f"Running: {cmd}")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() # Ensure current dir is in path
    try:
        subprocess.check_call(cmd, shell=True, cwd=cwd, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        # Build robustness: Don't exit loop, just warn? 
        # For training, failure is critical.
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=5, help="Number of adversarial generations.")
    parser.add_argument("--attacker-steps", type=int, default=1_000_000, help="Steps per attacker generation.")
    parser.add_argument("--defender-steps", type=int, default=500_000, help="Steps per defender generation.")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments.")
    parser.add_argument("--base-path", default="runs/adversarial", help="Base path for artifacts.")
    args = parser.parse_args()
    
    run_dir = Path(args.base_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Gen 0: Uniform Defender (Implicit) -> Train Attacker
    # Gen 1: Train Defender (vs Attacker 0)
    # ...
    
    # We assume Phase 2 might have run, but we start fresh or continue?
    # Let's start Gen 1 assuming Gen 0 Attacker exists?
    # Or just start from scratch.
    
    # Initial: Train Attacker vs Uniform (Gen 0)
    print("\n=== GEN 0: Bootstrapping Attacker vs Uniform ===")
    gen0_attacker_path = run_dir / "gen0_attacker"
    current_attacker = f"{gen0_attacker_path}.zip"
    if Path(current_attacker).exists():
        print(f"Skipping Gen 0 (Found {current_attacker})")
    else:
        run_command(
            f"python -m scripts.train_ppo --total-timesteps {args.attacker_steps} "
            f"--num-envs {args.num_envs} "
            f"--save-path {gen0_attacker_path}"
        )
    
    for gen in range(1, args.generations + 1):
        print(f"\n=== GEN {gen}: Adversarial Round ===")
        
        # 1. Train Defender vs Current Attacker
        print(f"--- Training Defender (Gen {gen}) vs Attacker (Gen {gen-1}) ---")
        defender_path = run_dir / f"gen{gen}_defender"
        current_defender_path = f"{defender_path}.zip"

        if Path(current_defender_path).exists():
            print(f"Skipping Gen {gen} Defender (Found {current_defender_path})")
        else:
            run_command(
                f"python -m scripts.train_defender --total-timesteps {args.defender_steps} "
                f"--num-envs {args.num_envs} "
                f"--attacker-path {current_attacker} "
                f"--save-path {defender_path}"
            )
        
        # 2. Train Attacker vs New Defender
        print(f"--- Training Attacker (Gen {gen}) vs Defender (Gen {gen}) ---")
        attacker_path = run_dir / f"gen{gen}_attacker"
        next_attacker = f"{attacker_path}.zip"
        
        if Path(next_attacker).exists():
             print(f"Skipping Gen {gen} Attacker (Found {next_attacker})")
             current_attacker = next_attacker
        else:
            run_command(
                f"python -m scripts.train_ppo --total-timesteps {args.attacker_steps} "
                f"--num-envs {args.num_envs} "
                f"--defender-path {current_defender_path} "
                f"--save-path {attacker_path}"
            )
            current_attacker = next_attacker
        

        
    print("\n=== Adversarial Loop Complete ===")

if __name__ == "__main__":
    main()
