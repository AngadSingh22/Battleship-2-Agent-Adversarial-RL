
import subprocess
import sys
from pathlib import Path

def main():
    print("=== Final Fast Track Queue ===")
    print("Objective: Complete Gen 1 Attacker -> Gen 2 Enforce")
    
    # We purposefully set generations=2.
    # Logic:
    # 0 (Skipped)
    # 1 (Defender Skipped, Attacker Trains)
    # 2 (Defender Trains, Attacker Trains)
    # End.
    
    cmd = [
        "python", "scripts/train_adversarial_loop.py",
        "--generations", "2",
        "--attacker-steps", "200000",
        "--defender-steps", "100000",
        "--num-envs", "4"
    ]
    
    log_path = Path("runs/final_queue.log")
    
    with open(log_path, "w") as f:
        print(f"Launching process. Looping logs to {log_path}...")
        subprocess.check_call(cmd, stdout=f, stderr=subprocess.STDOUT)
    
    print("Queue Complete.")

if __name__ == "__main__":
    main()
