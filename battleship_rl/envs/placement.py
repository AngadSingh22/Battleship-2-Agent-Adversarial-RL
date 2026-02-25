from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def _normalize_board_size(board_size: int | Sequence[int]) -> Tuple[int, int]:
    if isinstance(board_size, int):
        return board_size, board_size
    if len(board_size) != 2:
        raise ValueError("board_size must be int or length-2 sequence")
    return int(board_size[0]), int(board_size[1])


def _normalize_ships(ships: Sequence[int] | dict) -> List[int]:
    if isinstance(ships, dict):
        lengths = list(ships.values())
    else:
        lengths = list(ships)
    if not lengths:
        raise ValueError("ships must be a non-empty sequence")
    for length in lengths:
        if int(length) <= 0:
            raise ValueError("ship lengths must be positive")
    return [int(length) for length in lengths]


def _enumerate_candidates(
    grid: np.ndarray, length: int
) -> List[List[Tuple[int, int]]]:
    height, width = grid.shape
    candidates: List[List[Tuple[int, int]]] = []

    for r in range(height):
        for c in range(width - length + 1):
            if np.all(grid[r, c : c + length] == -1):
                candidates.append([(r, c + i) for i in range(length)])

    for c in range(width):
        for r in range(height - length + 1):
            if np.all(grid[r : r + length, c] == -1):
                candidates.append([(r + i, c) for i in range(length)])

    return candidates


def decode_placement_action(
    action: int, height: int, width: int
) -> Tuple[int, int, int]:
    """Decode a flat placement action into (row, col, orientation).

    Gap 10: This shared utility replaces the three divergent copies of the
    same decode logic that existed in ``BattleshipPlacementEnv.step``,
    ``BattleshipPlacementEnv.action_masks``, and
    ``AdversarialDefender.sample_layout``.

    Action space layout:
        0 .. H*W-1       → horizontal placement at (r, c)
        H*W .. 2*H*W-1   → vertical placement at (r, c)

    Returns:
        (row, col, orientation)  where orientation 0=horizontal, 1=vertical.
    """
    offset = height * width
    orientation = 0 if action < offset else 1
    flat_coord = action if action < offset else action - offset
    r, c = divmod(flat_coord, width)
    return r, c, orientation


def sample_placement(
    board_size: int | Sequence[int],
    ships: Sequence[int] | dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a legal ship placement using the provided RNG.

    Gap 14: The placement order is now randomised per call so the marginal
    distribution over layouts is closer to uniform (previously the fixed
    placement order introduced a bias favouring earlier ships).

    Returns:
        ship_id_grid: (H, W) int array. -1 = empty, 0..N-1 = ship IDs.
    """
    height, width = _normalize_board_size(board_size)
    ship_lengths = _normalize_ships(ships)
    num_ships = len(ship_lengths)
    ship_id_grid = -np.ones((height, width), dtype=np.int32)

    # Randomise placement order to reduce ordering bias (Gap 14).
    placement_order = rng.permutation(num_ships).tolist()

    for ship_id in placement_order:
        length = ship_lengths[ship_id]
        candidates = _enumerate_candidates(ship_id_grid, length)
        if not candidates:
            raise ValueError(
                f"No legal placements remaining for ship {ship_id} (length={length})."
            )
        choice_idx = int(rng.integers(0, len(candidates)))
        for r, c in candidates[choice_idx]:
            ship_id_grid[r, c] = ship_id

    return ship_id_grid
