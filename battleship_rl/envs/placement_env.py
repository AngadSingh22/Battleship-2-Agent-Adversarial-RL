from __future__ import annotations

from typing import Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from battleship_rl.envs.placement import _enumerate_candidates, _normalize_board_size, _normalize_ships
from battleship_rl.envs.battleship_env import BattleshipEnv
from stable_baselines3 import PPO

class BattleshipPlacementEnv(gym.Env):
    """
    Environment where the agent places ships sequentially.
    Reward is derived from playing a game against a fixed Attacker agent.
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        board_size: int | Sequence[int] = 10,
        ships: Optional[Sequence[int]] = None,
        attacker_model: Optional[str] = None, # Path to attacker model
    ):
        super().__init__()
        
        self.height, self.width = _normalize_board_size(board_size)
        
        if ships is None:
            ships = [5, 4, 3, 3, 2]
        self.ships = list(ships) # List of lengths
        
        self.attacker_model_path = attacker_model
        
        # Load Attacker Logic
        # Ideally we want a fast inference attacker.
        # If None, use Random Attacker (Uniform)
        self.attacker_agent = None
        self.attacker_needs_legacy_obs = False  # Default: assume new format
        if attacker_model:
            from sb3_contrib import MaskablePPO
            # Load lazily or now?
            try:
                self.attacker_agent = MaskablePPO.load(attacker_model)
                # Check if attacker expects old 3-channel observations
                attacker_obs_shape = self.attacker_agent.observation_space.shape
                self.attacker_needs_legacy_obs = (attacker_obs_shape[0] == 3)
                if self.attacker_needs_legacy_obs:
                    print(f"[COMPAT] Attacker expects 3-channel obs, will slice 4-channel to 3")
            except Exception as e:
                print(f"Warning: Could not load attacker {attacker_model}: {e}")
        
        # State
        self.current_ship_idx = 0
        self.board = np.zeros((self.height, self.width), dtype=np.int32) - 1 # -1 Empty
        
        # Action Space: Flattened (Orientation=2, H, W) -> 2*H*W
        # 0..(HW-1): Horizontal at (r,c)
        # HW..(2HW-1): Vertical at (r,c)
        self.action_space = spaces.Discrete(2 * self.height * self.width)
        
        # Observation Space: (3, H, W)
        # 0: Occupied Mask (0/1)
        # 1: Current Ship Length (Normalized)
        # 2: Current Ship Index (Normalized)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3, self.height, self.width), dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_ship_idx = 0
        self.board.fill(-1)
        return self._get_obs(), self._get_info()

    def _get_obs(self):
        obs = np.zeros((3, self.height, self.width), dtype=np.float32)
        # Ch 0: Occupied
        obs[0] = (self.board != -1).astype(np.float32)
        
        if self.current_ship_idx < len(self.ships):
            # Ch 1: Length
            length = self.ships[self.current_ship_idx]
            obs[1] = float(length) / max(self.ships)
            # Ch 2: Index
            obs[2] = float(self.current_ship_idx) / len(self.ships)
            
        return obs

    def _get_info(self):
        return {"action_mask": self.action_masks()}

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)
        if self.current_ship_idx >= len(self.ships):
            return mask # No actions possible
            
        length = self.ships[self.current_ship_idx]
        
        # Helper to check placement
        for r in range(self.height):
            for c in range(self.width):
                # Horizontal (Action 0..HW-1)
                idx_h = r * self.width + c
                if c + length <= self.width:
                    if np.all(self.board[r, c : c + length] == -1):
                        mask[idx_h] = True
                
                # Vertical (Action HW..2HW-1)
                idx_v = self.height * self.width + idx_h
                if r + length <= self.height:
                    if np.all(self.board[r : r + length, c] == -1):
                        mask[idx_v] = True
        return mask

    def step(self, action):
        # Apply action
        # Assumes action is valid (MaskedPPO will ensure, but we check)
        mask = self.action_masks()
        if not mask[action]:
             return self._get_obs(), -50.0, True, False, self._get_info() # Invalid death
             
        action = int(action)
        length = self.ships[self.current_ship_idx]
        
        offset = self.height * self.width
        if action < offset:
            # Horizontal
            r, c = divmod(action, self.width)
            self.board[r, c : c + length] = self.current_ship_idx
        else:
            # Vertical
            r, c = divmod(action - offset, self.width)
            self.board[r : r + length, c] = self.current_ship_idx
            
        self.current_ship_idx += 1
        
        if self.current_ship_idx >= len(self.ships):
            # Done! Evaluate layout against attacker
            reward = self._evaluate_layout()
            return self._get_obs(), reward, True, False, self._get_info()
        
        return self._get_obs(), 0.0, False, False, self._get_info()

    def _evaluate_layout(self) -> float:
        """Run a game with this layout against the Attacker Model."""
        # Create a temporary simulation env
        # Note: We can create a lightweight simulator or use BattleshipEnv
        # Using BattleshipEnv with FixedDefender logic
        
        # Define Fixed Defender for this specific layout
        class OneOffDefender:
            def __init__(self, arr): self.arr = arr
            def sample_layout(self, *args): return self.arr
            
        # Create Env
        # Important: Use C-Core for speed if available!
        env = BattleshipEnv(
            board_size=(self.height, self.width),
            ships=self.ships,
            defender=OneOffDefender(self.board.copy()), # Copy strictly needed?
            debug=False
        )
        obs, _ = env.reset()
        
        # Run Episode
        # Max steps = HW
        max_steps = self.height * self.width
        shots = 0
        
        terminated = False
        truncated = False
        
        # Rollout
        # If self.attacker_agent is None, we run pure random.
        # If Agent is loaded, we predict.
        
        while not (terminated or truncated) and shots < max_steps:
             if self.attacker_agent:
                 mask = env.get_action_mask()
                 # Backward compatibility: slice 4-channel obs to 3 if attacker expects 3
                 obs_for_attacker = obs[:3] if self.attacker_needs_legacy_obs else obs
                 action, _ = self.attacker_agent.predict(obs_for_attacker, action_masks=mask, deterministic=True)
             else:
                 # Random
                 mask = env.get_action_mask()
                 candidates = np.flatnonzero(mask)
                 action = np.random.choice(candidates) # Simple random
            
             obs, _, terminated, truncated, _ = env.step(action)
             shots += 1
             
        # Reward:
        # We want to MAXIMIZE shots (Defender wants to survive long)
        # Shots = Time to kill or Steps survived
        return float(shots)
