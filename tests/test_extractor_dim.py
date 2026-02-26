import pytest
import torch as th
from gymnasium import spaces
import numpy as np
from battleship_rl.agents.policies import BattleshipFeatureExtractor

def test_feature_extractor_dimension():
    """Verify the explicitly requested 512-dim constraint.
    
    The plan states: "conv stack -> pooling -> linear -> 512".
    We must mathematically test that giving it a batch of 3-channel
    observations yields exactly shape (batch_size, 512).
    """
    # 3 channels (Hit, Miss, Unknown), 10x10 board
    observation_space = spaces.Box(
        low=0.0, high=1.0, shape=(3, 10, 10), dtype=np.float32
    )
    
    # 512 explicitly required
    features_dim = 512
    extractor = BattleshipFeatureExtractor(observation_space, features_dim=features_dim)
    
    # Create dummy batch of 4 observations
    batch_size = 4
    dummy_obs = th.zeros((batch_size, 3, 10, 10), dtype=th.float32)
    
    # Forward pass
    with th.no_grad():
        output = extractor(dummy_obs)
        
    assert output.shape == (batch_size, features_dim), \
        f"Extractor must return exactly shape (batch_size, 512). Got {output.shape}"
