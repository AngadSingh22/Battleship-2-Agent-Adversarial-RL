from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces




class BattleshipPlacementEnv(gym.Env):
    """
    Environment specifically for training a Defender to place ships.
    
    State:
        - Board occupancy (H, W)
        - Current ship index (scalar)
    Action:
        - Box(3): (r, c, orientation). Or Discrete(H*W*2).
        - Simplified: Discrete(H * W * 2) where 0..H*W-1 is Horizontal, H*W..2H*W-1 is Vertical.
    """
    
    def __init__(
        self,
        board_size: int | tuple[int, int] = (10, 10),
        ships: list[int] = [5, 4, 3, 3, 2],
        attacker_path: str | None = None,
    ):
        super().__init__()
        self.attacker_model = None
        if attacker_path:
            try:
                from sb3_contrib import MaskablePPO
                self.attacker_model = MaskablePPO.load(attacker_path)
            except ImportError:
                 pass # Warning printed usage
            except Exception:
                 pass

        if isinstance(board_size, int):
            self.height = board_size
            self.width = board_size
        else:
            self.height, self.width = board_size
            
        self.ships = ships
        self.num_ships = len(ships)
        
        # Action: (r, c, orientation) flattened.
        # Orientation 0: Horizontal, 1: Vertical
        self.action_space = spaces.Discrete(self.height * self.width * 2)
        
        # Observation:
        # Channel 0: Occupancy (0=Empty, 1=Occupied)
        # Channel 1: Current Ship Length (normalized)
        # Channel 2: Current Ship Index (normalized)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, self.height, self.width), dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_ship_idx = 0
        self.board = np.zeros((self.height, self.width), dtype=np.int32) - 1 # -1 Empty
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.zeros((3, self.height, self.width), dtype=np.float32)
        # Channel 0: Occupancy mask
        obs[0] = (self.board != -1).astype(np.float32)
        
        if self.current_ship_idx < self.num_ships:
            length = self.ships[self.current_ship_idx]
            # Channel 1: Length plane
            obs[1] = float(length) / max(self.ships)
            # Channel 2: Index plane (progress)
            obs[2] = float(self.current_ship_idx) / self.num_ships
            
        return obs

    def step(self, action):
        if self.current_ship_idx >= self.num_ships:
            return self._get_obs(), 0.0, True, False, {"error": "All ships placed"}

        orientation = int(action // (self.height * self.width))
        flat_coord = int(action % (self.height * self.width))
        r = flat_coord // self.width
        c = flat_coord % self.width
        length = self.ships[self.current_ship_idx]
        
        # Validate using helper (note: helper wants grid with -1 for empty)
        # We need to adapt _is_valid_placement logic or import it.
        # For speed/simplicity here, inline check:
        # Check bounds
        if orientation == 0: # Horizontal
            if c + length > self.width:
                return self._get_obs(), -10.0, True, False, {"error": "OOB"}
            if np.any(self.board[r, c : c + length] != -1):
                return self._get_obs(), -10.0, True, False, {"error": "Overlap"}
        else: # Vertical
            if r + length > self.height:
                return self._get_obs(), -10.0, True, False, {"error": "OOB"}
            if np.any(self.board[r : r + length, c] != -1):
                return self._get_obs(), -10.0, True, False, {"error": "Overlap"}
                
        # Place ship
        if orientation == 0:
            self.board[r, c : c + length] = self.current_ship_idx
        else:
            self.board[r : r + length, c] = self.current_ship_idx
            
        self.current_ship_idx += 1
        terminated = self.current_ship_idx >= self.num_ships
        
        # Reward: 0 until end? or small positive for placement?
        # Real reward comes from Attacker difficulty, so this is just the 'validity' env.
        # When training Adversarial, we'd wrap this or provide reward from the Attacker game.
        # Reward: 
        # If done, simulate game to see how hard it is for the attacker.
        # If no attacker model, fallback to valid placement reward.
        reward = 0.0
        if terminated:
            if self.attacker_model:
                hardness = self._simulate_attacker_game()
                # Scaling: e.g. steps / 10.0 or just raw steps
                # If attacker takes 50 steps, reward 50. If 17 (min), reward 17.
                # Maximizing steps creates harder boards.
                reward = hardness
            else:
                 reward = 1.0 
        else:
            reward = 0.1
        
        return self._get_obs(), reward, terminated, False, {}

    def get_action_mask(self):
        # Return boolean mask of valid actions for current ship
        if self.current_ship_idx >= self.num_ships:
            return np.zeros(self.action_space.n, dtype=bool)
            
        length = self.ships[self.current_ship_idx]
        mask = np.zeros(self.action_space.n, dtype=bool)
        
        # Iterate all positions? Vectorized check better?
        # For 10x10, 200 actions. Iteration is fine.
        for action in range(self.action_space.n):
            orientation = int(action // (self.height * self.width))
            flat_coord = int(action % (self.height * self.width))
            r = flat_coord // self.width
            c = flat_coord % self.width
            
            valid = True
            if orientation == 0:
                if c + length > self.width: valid = False
                elif np.any(self.board[r, c : c + length] != -1): valid = False
            else:
                if r + length > self.height: valid = False
                elif np.any(self.board[r : r + length, c] != -1): valid = False
            
            if valid:
                mask[action] = True
                
        return mask

    def _simulate_attacker_game(self) -> float:
        """
        Simulates a game using the current placed board against the attacker model.
        Returns: Difficulty Score (e.g. number of steps taken to win).
        """
        if self.attacker_model is None:
            return 0.0

        # We need to construct a BattleshipEnv.
        # Ideally, we reuse a single instance to avoid overhead?
        # For simplicity in this implementation, we instantiate or reuse.
        # Note: We need to inject OUR board into it.
        
        # Checking for circular import needed?
        try:
            from battleship_rl.envs.battleship_env import BattleshipEnv
            # We need a Defender that just returns our current board.
            class FixedDefender:
                def __init__(self, board): self.board = board
                def sample_layout(self, *args): return self.board
                
            # Create Env (or reuse if we cached it)
            # We use the same config as self (board_size, ships)
            env = BattleshipEnv(
                board_size=(self.height, self.width),
                ships=self.ships,
                defender=FixedDefender(self.board),
                debug=False
            )
            
            # Run Episode
            obs, info = env.reset()
            terminated = False
            truncated = False
            steps = 0
            
            while not (terminated or truncated):
                # Predict action
                action, _ = self.attacker_model.predict(
                    obs, action_masks=env.get_action_mask(), deterministic=True
                )
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                
                # Safety break
                if steps >= self.height * self.width:
                    break
            
            # Reward: Higher steps = Better Defender
            # Normalize? Steps / (H*W)
            return float(steps)
            
        except ImportError:
            return 0.0
        except Exception as e:
            # print(f"Simulation warning: {e}")
            return 0.0

