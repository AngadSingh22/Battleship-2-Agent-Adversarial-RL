import pytest
import numpy as np
from battleship_rl.envs.battleship_env import BattleshipEnv

def test_seeding_reproducibility():
    """Verify that same seed = same exact placement and outcomes."""
    seed = 12345
    env1 = BattleshipEnv(board_size=10)
    env2 = BattleshipEnv(board_size=10)
    
    obs1, info1 = env1.reset(seed=seed)
    obs2, info2 = env2.reset(seed=seed)
    
    # 1. Placement equality
    np.testing.assert_array_equal(env1.ship_id_grid, env2.ship_id_grid)
    np.testing.assert_array_equal(obs1, obs2)
    
    # 2. Outcome equality for a fixed action sequence
    actions = [5, 12, 45, 99, 0, 77]
    
    for a in actions:
        o1, r1, term1, trunc1, i1 = env1.step(a)
        o2, r2, term2, trunc2, i2 = env2.step(a)
        
        np.testing.assert_array_equal(o1, o2)
        assert r1 == r2
        assert term1 == term2
        assert trunc1 == trunc2
        assert i1["outcome_type"] == i2["outcome_type"]
        assert i1["outcome_ship_id"] == i2["outcome_ship_id"]
        np.testing.assert_array_equal(i1["action_mask"], i2["action_mask"])
