import pytest
import numpy as np
from battleship_rl.envs.battleship_env import BattleshipEnv

def test_observation_consistency():
    """Verify channel order and the Unknown invariant after each step."""
    env = BattleshipEnv(board_size=10)
    obs, info = env.reset()
    
    assert obs.shape == (3, 10, 10), "Observation must be exactly 3 channels (Hit, Miss, Unknown)"
    assert obs.dtype == np.float32, "Observation must be float32"
    
    for _ in range(20):
        # Pick a random valid action
        mask = info["action_mask"]
        valid_actions = np.where(mask)[0]
        if len(valid_actions) == 0:
            break
            
        action = np.random.choice(valid_actions)
        obs, reward, terminated, truncated, info = env.step(action)
        
        hit_channel = obs[0]
        miss_channel = obs[1]
        unknown_channel = obs[2]
        
        # Invariant 1: Hit and Miss are disjoint
        overlap = np.sum(hit_channel * miss_channel)
        assert overlap == 0.0, "Hit and Miss channels must be strictly disjoint."
        
        # Invariant 2: Unknown = 1 - (Hit + Miss)
        expected_unknown = 1.0 - (hit_channel + miss_channel)
        np.testing.assert_array_almost_equal(unknown_channel, expected_unknown)
        
        if terminated:
            break
