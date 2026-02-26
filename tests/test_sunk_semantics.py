import pytest
import numpy as np
from battleship_rl.envs.battleship_env import BattleshipEnv

def test_sunk_semantics():
    """Verify sunk events are emitted in info keys exactly once per ship, and tracking is correct."""
    env = BattleshipEnv(board_size=10)
    obs, info = env.reset(seed=42)
    
    # We will cheat by knowing where the ships are via ship_id_grid to sink one
    ship_id_target = 0
    ship_cells = np.argwhere(env.ship_id_grid == ship_id_target)
    
    sunk_emitted = 0
    
    for i, (r, c) in enumerate(ship_cells):
        action = r * 10 + c
        obs, reward, terminated, truncated, info = env.step(action)
        
        # It should only emit SUNK on the final hit of the ship
        is_last_hit = (i == len(ship_cells) - 1)
        
        if is_last_hit:
            assert info["outcome_type"] == "SUNK"
            assert info["outcome_ship_id"] == ship_id_target
            assert ship_id_target in env.sunk_ships
            sunk_emitted += 1
        else:
            assert info["outcome_type"] == "HIT"
            assert info["outcome_ship_id"] == ship_id_target
            assert ship_id_target not in env.sunk_ships
            
    assert sunk_emitted == 1, "SUNK event should be emitted exactly once for the ship."
