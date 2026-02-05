"""
Queue Gen 3 through Gen 5 adversarial training.
This will be the FIRST training run to use the formulation fixes:
- 4-channel observations (ActiveHit, Miss, Sunk, Unknown)
- -1.0 step penalty (harsh time pressure)
- Dense rewards (+1.0 for hits, +5.0 for sinks)
"""

import subprocess
from pathlib import Path

def main():
    print("=== Gen 3-5 Training Queue ===")
    print("Objective: Train Gen 3, 4, and 5 with formulation fixes")
    print()
    print("Expected improvements:")
    print("  - Gen 3 Attacker: DRAMATIC improvement with dense rewards")
    print("  - Gen 3+ Defenders: Continue to evolve defensive strategies")
    print()
    
    # The adversarial loop will:
    # - Skip Gen 0, 1, 2 (already exist)
    # - Train Gen 3 (Defender + Attacker)
    # - Train Gen 4 (Defender + Attacker)
    # - Train Gen 5 (Defender + Attacker)
    
    cmd = [
        "python", "scripts/train_adversarial_loop.py",
        "--generations", "5",  # Trains up to and including Gen 5
        "--attacker-steps", "200000",  # 200k per attacker
        "--defender-steps", "100000",  # 100k per defender
        "--num-envs", "4"
    ]
    
    log_path = Path("runs/gen3_to_5_queue.log")
    
    with open(log_path, "w") as f:
        print(f"Launching process. Logging to {log_path}...")
        print("This will train 6 models (3 Defenders + 3 Attackers).")
        print("Estimated time: ~2-3 hours")
        print()
        subprocess.check_call(cmd, stdout=f, stderr=subprocess.STDOUT)
    
    print()
    print("=== Queue Complete ===")
    print(f"Check logs: {log_path}")

if __name__ == "__main__":
    main()
