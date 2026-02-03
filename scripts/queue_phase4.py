import psutil
import time
import subprocess
import sys
from pathlib import Path

def find_training_process():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower():
                cmdline = proc.info['cmdline'] or []
                # Check for train_phase2_full.py
                if any("train_phase2_full.py" in arg for arg in cmdline):
                    return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return None

def main():
    print("Monitor: Looking for active Phase 2 training...")
    proc = find_training_process()
    
    if proc:
        print(f"Monitor: Found Phase 2 process (PID {proc.info['pid']}). Waiting for completion...")
        try:
            # Poll until gone
            while proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                time.sleep(10)
        except psutil.NoSuchProcess:
            pass
        print("Monitor: Phase 2 process finished.")
    else:
        print("Monitor: No active Phase 2 training found. Assuming finished.")

    # Launch Phase 4
    print("Monitor: Launching Phase 4 Adversarial Loop...")
    log_file = Path("runs/phase4_log.txt")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, "w") as f:
        subprocess.call(
            ["python", "scripts/train_adversarial_loop.py", "--generations", "5"],
            stdout=f, stderr=subprocess.STDOUT
        )
    print("Monitor: Phase 4 Launched.")

if __name__ == "__main__":
    main()
