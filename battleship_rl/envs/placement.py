from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

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


def sample_placement(
    board_size: int | Sequence[int],
    ships: Sequence[int] | dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a legal ship placement using the provided RNG.

    Returns:
        ship_id_grid: (H, W) int array. -1 = empty, 0..N-1 = ship IDs.
    """
    height, width = _normalize_board_size(board_size)
    ship_lengths = _normalize_ships(ships)
    ship_id_grid = -np.ones((height, width), dtype=np.int32)

    for ship_id, length in enumerate(ship_lengths):
        candidates = _enumerate_candidates(ship_id_grid, length)
        if not candidates:
            raise ValueError("No legal placements remaining for ship length.")
        choice_idx = int(rng.integers(0, len(candidates)))
        for r, c in candidates[choice_idx]:
            ship_id_grid[r, c] = ship_id

    return ship_id_grid
