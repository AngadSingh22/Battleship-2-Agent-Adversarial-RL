import gymnasium as gym
import numpy as np

class BattleshipEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, config=None):
        super().__init__()
        # Placeholder
        pass
        
    def reset(self, seed=None, options=None):
        pass
        
    def step(self, action):
        pass
        
    def render(self):
        pass
