from __future__ import annotations

from typing import Sequence

import numpy as np

from battleship_rl.envs.placement import (
    _enumerate_candidates,
    _normalize_board_size,
    _normalize_ships,
    decode_placement_action,
    sample_placement,
)


class BaseDefender:
    def sample_layout(
        self,
        board_size: int | Sequence[int],
        ships: Sequence[int] | dict,
        rng: np.random.Generator,
    ) -> np.ndarray:
        raise NotImplementedError


class UniformRandomDefender(BaseDefender):
    def sample_layout(
        self,
        board_size: int | Sequence[int],
        ships: Sequence[int] | dict,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return sample_placement(board_size, ships, rng)


class BiasedDefender(BaseDefender):
    """Edge-biased defender using deterministic weighting with RNG sampling."""

    def _edge_weight_grid(self, height: int, width: int) -> np.ndarray:
        weights = np.zeros((height, width), dtype=np.float32)
        for r in range(height):
            for c in range(width):
                dist = min(r, c, height - 1 - r, width - 1 - c)
                weights[r, c] = 1.0 / (dist + 1)
        return weights

    def sample_layout(
        self,
        board_size: int | Sequence[int],
        ships: Sequence[int] | dict,
        rng: np.random.Generator,
    ) -> np.ndarray:
        height, width = _normalize_board_size(board_size)
        ship_lengths = _normalize_ships(ships)
        ship_id_grid = -np.ones((height, width), dtype=np.int32)
        weight_grid = self._edge_weight_grid(height, width)

        for ship_id, length in enumerate(ship_lengths):
            candidates = _enumerate_candidates(ship_id_grid, length)
            if not candidates:
                raise ValueError("No legal placements remaining for ship length.")
            weights = np.array(
                [sum(weight_grid[r, c] for r, c in candidate) for candidate in candidates],
                dtype=np.float32,
            )
            if float(weights.sum()) <= 0.0:
                choice_idx = int(rng.integers(0, len(candidates)))
            else:
                probs = weights / weights.sum()
                choice_idx = int(rng.choice(len(candidates), p=probs))
            for r, c in candidates[choice_idx]:
                ship_id_grid[r, c] = ship_id


        return ship_id_grid


class AdversarialDefender(BaseDefender):
    """
    Defender that uses an RL policy to place ships.
    Requires a pre-trained SB3 model (trained on BattleshipPlacementEnv).
    """

    def __init__(self, model_path: str | None = None, deterministic: bool = True):
        self.deterministic = deterministic
        self.model = None
        if model_path:
            try:
                from sb3_contrib import MaskablePPO
                self.model = MaskablePPO.load(model_path)
            except ImportError:
                print("Warning: sb3-contrib not installed or model load failed.")
            except Exception as e:
                print(f"Warning: Failed to load adversarial model: {e}")

    def sample_layout(
        self,
        board_size: int | Sequence[int],
        ships: Sequence[int] | dict,
        rng: np.random.Generator,
    ) -> np.ndarray:
        # If no model, fallback to Uniform
        if self.model is None:
            return sample_placement(board_size, ships, rng)

        height, width = _normalize_board_size(board_size)
        ship_lengths = _normalize_ships(ships)
        
        # Instantiate a temporary environment logic to track state
        # (We mimic BattleshipPlacementEnv logic here purely for inference)
        board = np.zeros((height, width), dtype=np.int32) - 1
        
        for ship_idx, length in enumerate(ship_lengths):
            # Construct observation
            obs = np.zeros((3, height, width), dtype=np.float32)
            obs[0] = (board != -1).astype(np.float32)
            obs[1] = float(length) / max(ship_lengths)
            obs[2] = float(ship_idx) / len(ship_lengths)
            
            # Construct mask
            # We need a proper mask function. Ideally reuse one from placement_env logic or placement_logic
            # For strictness, let's reimplement simple mask check here or rely on the env if we instantiated it?
            # Creating a full env is cleaner but heavier. Let's do mask check manually.
            mask = np.zeros(height * width * 2, dtype=bool)
            
            # This loop is slightly expensive for inference (200 checks * 5 ships), but generally < 1ms
            for r in range(height):
                for c in range(width):
                    # Horizontal
                    if c + length <= width and np.all(board[r, c : c + length] == -1):
                        mask[r * width + c] = True
                    # Vertical
                    if r + length <= height and np.all(board[r : r + length, c] == -1):
                        mask[height * width + r * width + c] = True
            
            if not np.any(mask):
                # Should not happen in placement unless trapped. Fallback to random search for this ship?
                # Or restart?
                return sample_placement(board_size, ships, rng)

            action, _ = self.model.predict(
                obs, action_masks=mask, deterministic=self.deterministic
            )

            # Apply action using the shared decode utility (Gap 10).
            r, c, orientation = decode_placement_action(int(action), height, width)
            if orientation == 0:  # Horizontal
                board[r, c : c + length] = ship_idx
            else:                 # Vertical
                board[r : r + length, c] = ship_idx
                
        return board
