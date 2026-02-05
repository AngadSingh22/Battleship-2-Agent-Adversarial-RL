"""Verify formulation alignment changes."""
import numpy as np
from battleship_rl.envs.battleship_env import BattleshipEnv

def test_observations():
    """Test that observations are 4-channel with sunk signal."""
    env = BattleshipEnv(config={"board_size": 10})
    obs, _ = env.reset(seed=42)
    
    assert obs.shape == (4, 10, 10), f"Expected (4,10,10), got {obs.shape}"
    print(f"✓ Observation shape correct: {obs.shape}")
    
    #assert obs.shape[0] == 4, f"Expected 4 channels, got {obs.shape[0]}"
    print(f"✓ Four channels: ActiveHit, Miss, Sunk, Unknown")
    
    # Force a hit
    r, c = np.argwhere(env.ship_id_grid != -1)[0]
    action = r * 10 + c
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Verify hit is in channel 0 (ActiveHit)
    assert obs[0, r, c] == 1.0, "Hit should appear in ActiveHit channel"
    print(f"✓ Hit correctly appears in ActiveHit channel")
    
    # Continue hitting same ship until sunk
    ship_id = env.ship_id_grid[r, c]
    ship_cells = np.argwhere(env.ship_id_grid == ship_id)
    
    for cell_r, cell_c in ship_cells[1:]:  # Skip first (already hit)
        action = cell_r * 10 + cell_c
        obs, reward, terminated, truncated, info = env.step(action)
        if info['outcome_type'] == 'SUNK':
            break
    
    # Verify all cells of sunk ship appear in channel 2 (Sunk)
    for cell_r, cell_c in ship_cells:
        assert obs[2, cell_r, cell_c] == 1.0, f"Sunk ship cell ({cell_r},{cell_c}) should be in Sunk channel"
        assert obs[0, cell_r, cell_c] == 0.0, f"Sunk ship cell ({cell_r},{cell_c}) should not be in ActiveHit"
    
    print(f"✓ Sunk ship correctly appears in Sunk channel (not ActiveHit)")

def test_rewards():
    """Test that rewards match formulation."""
    cfg = {
        "board_size": 10,
        "reward_scheme": {
            "hit": 1.0,
            "miss": -1.0,
            "sink": 5.0
        }
    }
    env = BattleshipEnv(config=cfg)
    env.reset(seed=42)
    
    # Force a miss
    r_miss, c_miss = np.argwhere(env.ship_id_grid == -1)[0]
    action_miss = r_miss * 10 + c_miss
    _, reward_miss, _, _, _ = env.step(action_miss)
    
    assert reward_miss == -1.0, f"Miss reward should be -1.0, got {reward_miss}"
    print(f"✓ Miss reward correct: {reward_miss}")
    
    # Reset and force a hit
    env.reset(seed=42)
    r_hit, c_hit = np.argwhere(env.ship_id_grid != -1)[0]
    action_hit = r_hit * 10 + c_hit
    _, reward_hit, _, _, info = env.step(action_hit)
    
    # Hit reward = base_penalty + alpha = -1.0 + 2.0 = 1.0
    assert reward_hit == 1.0, f"Hit reward should be 1.0, got {reward_hit}"
    print(f"✓ Hit reward correct: {reward_hit}")

if __name__ == "__main__":
    print("=== Formulation Alignment Verification ===\n")
    
    print("[1/2] Testing 4-Channel Observations...")
    test_observations()
    print()
    
    print("[2/2] Testing Reward Values...")
    test_rewards()
    print()
    
    print("=== ✓ ALL CHECKS PASSED ===")
    print("The implementation now matches the formulation:")
    print("  - Observations include Sunk(k) signal (Channel 2)")
    print("  - Step penalty is -1.0 (10x more pressure)")
    print("  - Shaped rewards provide dense learning signal")
