import pytest
import numpy as np
from battleship_rl.envs.battleship_env import BattleshipEnv

def test_action_mapping():
    """Ensure action -> (r, c) mapping matches the spec: r = action // W, c = action % W"""
    env = BattleshipEnv(board_size=(10, 10))
    env.reset()
    
    # We can't directly intercept the action inside step without mocking,
    # but we can verify the mapping by checking which cell in the backend was hit.
    # A cleaner way is to mock backend.step and see what r, c it infers, 
    # but the logic in `step` manually computes r, c for ship id retrieval.
    
    # Let's hit a specific cell that we know has a ship (if we mock the layout)
    layout = np.zeros((10, 10), dtype=int) - 1
    layout[3, 4] = 0 # Ship 0 at (3,4)
    env.ship_id_grid = layout
    env.backend.set_board(layout)
    
    # Fire at index 34 -> r=3, c=4
    action = 34
    obs, reward, terminated, truncated, info = env.step(action)
    
    # The C backend uses exactly this mapping (1D array internally, or row major).
    # If the mapping was wrong, it wouldn't hit ship 0.
    assert info["outcome_ship_id"] == 0, f"Expected to hit ship 0 at r=3, c=4. Got {info['outcome_ship_id']}"
    
    r, c = divmod(action, env.width)
    assert r == 3
    assert c == 4
    
