
import numpy as np
import matplotlib.pyplot as plt
from battleship_rl.agents.defender import AdversarialDefender

def main():
    model_path = "runs/adversarial/gen1_defender.zip"
    print(f"Loading Defender: {model_path}")
    
    try:
        defender = AdversarialDefender(model_path=model_path, deterministic=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    n_samples = 1000
    board_size = 10
    height, width = 10, 10
    ships = [5, 4, 3, 3, 2]
    rng = np.random.default_rng(42)
    
    occupancy_grid = np.zeros((height, width), dtype=np.float32)
    
    print(f"Sampling {n_samples} layouts...")
    for _ in range(n_samples):
        # Sample layout
        board = defender.sample_layout((height, width), ships, rng)
        # Binarize (ship present = 1, empty = 0)
        mask = (board != -1).astype(np.float32)
        occupancy_grid += mask
        
    # Normalize
    heatmap = occupancy_grid / n_samples
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, cmap="inferno", vmin=0, vmax=1)
    plt.colorbar(label="Ship Probability")
    plt.title(f"Gen 1 Defender Placement Strategy\n(AVG over {n_samples} games)")
    plt.xlabel("Column")
    plt.ylabel("Row")
    
    # Save
    output_path = "defender_gen1_heatmap.png"
    plt.savefig(output_path)
    print(f"Heatmap saved to {output_path}")

if __name__ == "__main__":
    main()
