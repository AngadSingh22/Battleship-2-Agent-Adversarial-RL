from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from battleship_rl.agents.defender import UniformRandomDefender
from battleship_rl.envs.masks import compute_action_mask
from battleship_rl.envs.observations import build_observation
from battleship_rl.envs.rewards import StepPenaltyReward
from bindings.c_api import CBattleshipFactory


class BattleshipEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        config: Optional[dict] = None,
        board_size: int | Sequence[int] = 10,
        ships: Optional[Sequence[int] | dict] = None,
        defender: Optional[object] = None,
        reward_fn: Optional[object] = None,
        debug: bool = False,
    ) -> None:
        super().__init__()

        cfg = config or {}
        board_size = cfg.get("board_size", board_size)
        ships = cfg.get("ship_config", ships)
        ships = cfg.get("ships", ships)

        if ships is None:
            ships = [5, 4, 3, 3, 2]

        if isinstance(board_size, int):
            height, width = board_size, board_size
        else:
            if len(board_size) != 2:
                raise ValueError("board_size must be int or length-2 sequence")
            height, width = int(board_size[0]), int(board_size[1])

        self.height = height
        self.width = width
        
        # Normalize ships to list of ints
        self.ship_lengths = [
            int(length)
            for length in (list(ships.values()) if isinstance(ships, dict) else list(ships))
        ]
        self.ships = self.ship_lengths # Alias
        self.num_ships = len(self.ship_lengths)

        self.defender = defender or UniformRandomDefender()
        
        # Reward Logic
        if reward_fn is not None:
            self.reward_fn = reward_fn
        elif "reward_scheme" in cfg:
            rs = cfg["reward_scheme"]
            # Parse YAML scheme to ShapedReward params
            # Expected YAML: hit, miss, sink
            # ShapedReward = step_penalty + (alpha if hit) + (beta if sink)
            step_penalty = float(rs.get("miss", -0.1))
            target_hit = float(rs.get("hit", 1.0))
            target_sink = float(rs.get("sink", 5.0))
            
            # alpha = Target - Base
            alpha = target_hit - step_penalty
            beta = target_sink - step_penalty
            
            from battleship_rl.envs.rewards import ShapedReward
            self.reward_fn = ShapedReward(alpha=alpha, beta=beta, step_penalty=step_penalty)
        else:
            self.reward_fn = StepPenaltyReward()
        self.debug = bool(debug)
        self.invalid_action_penalty = -100.0

        self.action_space = spaces.Discrete(self.height * self.width)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4, self.height, self.width),  # 4 channels: ActiveHit, Miss, Sunk, Unknown
            dtype=np.float32,
        )

        # Initialize Backend (C-Core or Python Fallback)
        self.backend = CBattleshipFactory(self.height, self.width, self.ship_lengths)

        # Views (Will be bound to backend memory)
        self.ship_id_grid: Optional[np.ndarray] = None
        self.hits_grid: Optional[np.ndarray] = None
        self.miss_grid: Optional[np.ndarray] = None
        self.sunk_ships: set[int] = set()

    def _build_info(self, outcome_type: Optional[str], outcome_ship_id: Optional[int]) -> Dict[str, Any]:
        return {
            "action_mask": self.get_action_mask(),
            "outcome_type": outcome_type,
            "outcome_ship_id": outcome_ship_id,
            "last_outcome": (outcome_type, outcome_ship_id),
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        # 1. Sample Layout (Python Side)
        self.ship_id_grid = self.defender.sample_layout(
            (self.height, self.width), self.ships, self.np_random
        )
        
        # 2. Sync to Backend
        # Reset backend state (clears hits/misses)
        self.backend.reset(seed if seed is not None else 0)
        # Copy layout to backend
        self.backend.set_board(self.ship_id_grid)
        
        # 3. Bind Views directly to backend memory
        self.hits_grid = self.backend.hits
        self.miss_grid = self.backend.misses
        self.sunk_ships = set() # Tracked for convenience, though backend has its own

        # Initial Obs
        obs = build_observation(self.hits_grid, self.miss_grid, self.ship_id_grid, self.backend.ship_sunk)
        info = self._build_info(None, None)
        return obs, info

    def step(self, action):
        action = int(action)
        mask = self.get_action_mask()
        
        # invalid Check
        if action < 0 or action >= mask.size or not mask[action]:
            if self.debug:
                raise ValueError("Invalid action: %s" % action)
            obs = build_observation(self.hits_grid, self.miss_grid, self.ship_id_grid, self.backend.ship_sunk)
            info = {
                "action_mask": mask,
                "outcome_type": "INVALID",
                "outcome_ship_id": None,
                "last_outcome": ("INVALID", None),
            }
            return obs, self.invalid_action_penalty, False, True, info

        # Delegate to Backend
        # Return code: 0=Miss, 1=Hit, 2=Sunk
        res = self.backend.step(action)
        
        outcome_ship_id = None
        outcome_type = "MISS"
        
        if res == 1: # Hit
            outcome_type = "HIT"
            r, c = divmod(action, self.width)
            outcome_ship_id = int(self.ship_id_grid[r, c])
        elif res == 2: # Sunk
            outcome_type = "SUNK"
            r, c = divmod(action, self.width)
            outcome_ship_id = int(self.ship_id_grid[r, c])
            self.sunk_ships.add(outcome_ship_id)
            
        # Check Termination
        # Use backend logic for sunk ships
        terminated = bool(np.all(self.backend.ship_sunk))
        
        reward = self.reward_fn(outcome_type, terminated)
        
        obs = build_observation(self.hits_grid, self.miss_grid, self.ship_id_grid, self.backend.ship_sunk)
        info = self._build_info(outcome_type, outcome_ship_id)
        
        return obs, reward, terminated, False, info

    def get_action_mask(self) -> np.ndarray:
        return compute_action_mask(self.hits_grid, self.miss_grid)

    def render(self):
        if self.hits_grid is None or self.miss_grid is None:
            return ""
        symbols = np.full((self.height, self.width), ".", dtype="<U1")
        symbols[self.miss_grid] = "o"
        symbols[self.hits_grid] = "X"
        lines = [" ".join(row.tolist()) for row in symbols]
        board_str = "\n".join(lines)
        return board_str
