#!/usr/bin/env python3
"""
Orchestration script for Phase 2 Training.
Sequentially trains the Attacker and then the Adversarial Defender.
"""
import subprocess
import sys
from pathlib import Path

def run_command(cmd, cwd=None):
    print(f"Running: {cmd}")
    try:
        subprocess.check_call(cmd, shell=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)

def main():
    root = Path(__file__).parent.parent
    
    # 1. Train Attacker (200k steps)
    print("\n=== STEP 1: Training Attacker (200k steps) ===")
    run_command(
        "python scripts/train_ppo.py --total-timesteps 200000 --num-envs 8 --save-path runs/battleship_ppo_200k",
        cwd=root
    )
    
    # 2. Train Defender (100k steps) using the trained Attacker
    # Note: 100k steps here is more expensive due to simulation
    print("\n=== STEP 2: Training Adversarial Defender (100k steps) ===")
    # We append .zip if strictly needed, or let SB3 handle it. 
    # train_defender.py loads via MaskablePPO.load() which handles missing .zip
    run_command(
        "python scripts/train_defender.py --total-timesteps 100000 --num-envs 4 "
        "--attacker-path runs/battleship_ppo_200k "
        "--save-path runs/defender_ppo_100k",
        cwd=root
    )

    print("\n=== PHASE 2 COMPLETE ===")
    print("Models saved to:")
    print("  - runs/battleship_ppo_200k.zip")
    print("  - runs/defender_ppo_100k.zip")

if __name__ == "__main__":
    main()
