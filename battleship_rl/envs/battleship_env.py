from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from battleship_rl.agents.defender import UniformRandomDefender
from battleship_rl.envs.masks import compute_action_mask
from battleship_rl.envs.observations import build_observation
from battleship_rl.envs.rewards import StepPenaltyReward


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
        self.ships = ships
        self.ship_lengths = [
            int(length)
            for length in (list(ships.values()) if isinstance(ships, dict) else list(ships))
        ]
        self.num_ships = len(self.ship_lengths)

        self.defender = defender or UniformRandomDefender()
        self.reward_fn = reward_fn or StepPenaltyReward()
        self.debug = bool(debug)
        self.invalid_action_penalty = -100.0

        self.action_space = spaces.Discrete(self.height * self.width)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, self.height, self.width),
            dtype=np.float32,
        )

        self.ship_id_grid: Optional[np.ndarray] = None
        self.hits_grid: Optional[np.ndarray] = None
        self.miss_grid: Optional[np.ndarray] = None
        self.sunk_ships: set[int] = set()
        self.ship_cells: dict[int, np.ndarray] = {}

    def _build_info(self, outcome_type: Optional[str], outcome_ship_id: Optional[int]) -> Dict[str, Any]:
        return {
            "action_mask": self.get_action_mask(),
            "outcome_type": outcome_type,
            "outcome_ship_id": outcome_ship_id,
            "last_outcome": (outcome_type, outcome_ship_id),
        }

    def _check_sunk(self, ship_id: int) -> bool:
        cells = self.ship_cells[ship_id]
        return bool(np.all(self.hits_grid[cells[:, 0], cells[:, 1]]))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.ship_id_grid = self.defender.sample_layout(
            (self.height, self.width), self.ships, self.np_random
        )
        self.hits_grid = np.zeros((self.height, self.width), dtype=bool)
        self.miss_grid = np.zeros((self.height, self.width), dtype=bool)
        self.sunk_ships = set()

        self.ship_cells = {}
        for ship_id in range(self.num_ships):
            coords = np.argwhere(self.ship_id_grid == ship_id)
            if coords.size == 0:
                raise ValueError("Placement missing ship_id %d" % ship_id)
            self.ship_cells[ship_id] = coords

        obs = build_observation(self.hits_grid, self.miss_grid)
        info = self._build_info(None, None)
        return obs, info

    def step(self, action):
        action = int(action)
        mask = self.get_action_mask()
        if action < 0 or action >= mask.size or not mask[action]:
            if self.debug:
                raise ValueError("Invalid action: %s" % action)
            obs = build_observation(self.hits_grid, self.miss_grid)
            info = {
                "action_mask": mask,
                "outcome_type": "INVALID",
                "outcome_ship_id": None,
                "last_outcome": ("INVALID", None),
            }
            return obs, self.invalid_action_penalty, False, True, info

        row = action // self.width
        col = action % self.width

        ship_id = int(self.ship_id_grid[row, col])
        outcome_ship_id: Optional[int] = None

        if ship_id == -1:
            self.miss_grid[row, col] = True
            outcome_type = "MISS"
        else:
            self.hits_grid[row, col] = True
            outcome_ship_id = ship_id
            if ship_id not in self.sunk_ships and self._check_sunk(ship_id):
                self.sunk_ships.add(ship_id)
                outcome_type = "SUNK"
            else:
                outcome_type = "HIT"

        terminated = len(self.sunk_ships) == self.num_ships
        reward = self.reward_fn(outcome_type, terminated)
        obs = build_observation(self.hits_grid, self.miss_grid)
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
