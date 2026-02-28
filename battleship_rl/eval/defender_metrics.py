"""
battleship_rl/eval/defender_metrics.py
========================================
Placement-distribution statistics for defender ladder evaluation.

Metrics (all computed from a sample of `n` ship-occupancy grids):
- edge_occupancy   : fraction of ship cells on boundary rows/cols
- mean_dist_edge   : mean minimum distance of ship cells to any edge
- cell_entropy     : mean per-cell binary entropy of the marginal occupancy map
- kl_from_uniform  : KL divergence of the marginal distribution vs. uniform

All functions accept a list of 2-D numpy arrays (H x W) where non-zero
entries mark ship-occupied cells.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def edge_occupancy(grids: Sequence[np.ndarray]) -> float:
    """Fraction of ship cells that lie on the board's boundary row/col."""
    total_edge = 0
    total_cells = 0
    for grid in grids:
        locs = np.argwhere(grid != 0)
        if len(locs) == 0:
            continue
        H, W = grid.shape
        for r, c in locs:
            if min(r, H - 1 - r, c, W - 1 - c) == 0:
                total_edge += 1
        total_cells += len(locs)
    return total_edge / total_cells if total_cells > 0 else 0.0


def mean_dist_to_edge(grids: Sequence[np.ndarray]) -> float:
    """Mean minimum Manhattan distance from a ship cell to the nearest edge."""
    total_dist = 0.0
    total_cells = 0
    for grid in grids:
        locs = np.argwhere(grid != 0)
        if len(locs) == 0:
            continue
        H, W = grid.shape
        for r, c in locs:
            total_dist += int(min(r, H - 1 - r, c, W - 1 - c))
        total_cells += len(locs)
    return total_dist / total_cells if total_cells > 0 else 0.0


def marginal_occupancy(grids: Sequence[np.ndarray]) -> np.ndarray:
    """Compute the marginal probability of each cell being ship-occupied."""
    if not grids:
        raise ValueError("grids must be non-empty")
    h, w = grids[0].shape
    freq = np.zeros((h, w), dtype=np.float64)
    for grid in grids:
        freq += (grid != 0).astype(np.float64)
    return freq / len(grids)


def cell_entropy(grids: Sequence[np.ndarray]) -> float:
    """Mean per-cell binary entropy H(p) of the marginal occupancy map.

    A uniform defender has maximum entropy; a concentrated defender has lower.
    """
    p = marginal_occupancy(grids)
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return float(np.mean(h))


def kl_from_uniform(grids: Sequence[np.ndarray]) -> float:
    """KL divergence D_KL(P_defender || P_uniform) over the cell occupancy.

    Uses the marginal of each cell being occupied vs. a flat uniform where
    every cell has equal probability (total_ship_area / H*W).
    """
    p = marginal_occupancy(grids)
    h, w = p.shape
    # Uniform reference: same total probability mass, spread uniformly
    q = np.full_like(p, p.mean())
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log2(p / q)))


def summarize_defender(
    defender,
    board_size: int | tuple[int, int],
    ships: list[int],
    n_samples: int = 500,
    seed: int = 0,
) -> dict:
    """Sample `n_samples` layouts from `defender` and compute all metrics.

    Parameters
    ----------
    defender : BaseDefender instance with `sample_layout(board_size, ships, rng)`
    board_size : int or (H, W) tuple
    ships : list of ship lengths
    n_samples : number of layouts to sample
    seed : RNG seed for reproducibility

    Returns
    -------
    dict with keys: edge_occupancy, mean_dist_edge, cell_entropy, kl_from_uniform
    """
    rng = np.random.default_rng(seed)
    grids = [defender.sample_layout(board_size, ships, rng) for _ in range(n_samples)]
    return {
        "edge_occupancy": edge_occupancy(grids),
        "mean_dist_edge": mean_dist_to_edge(grids),
        "cell_entropy": cell_entropy(grids),
        "kl_from_uniform": kl_from_uniform(grids),
    }


def print_comparison(results: dict[str, dict]) -> None:
    """Pretty-print a side-by-side comparison table of defender metrics."""
    keys = ["edge_occupancy", "mean_dist_edge", "cell_entropy", "kl_from_uniform"]
    labels = ["Edge Occupancy", "Mean Dist to Edge", "Cell Entropy", "KL From Uniform"]
    defenders = list(results.keys())
    col_w = 12

    header = f"{'Metric':<20}" + "".join(f"{d:>{col_w}}" for d in defenders)
    print(header)
    print("-" * len(header))
    for k, label in zip(keys, labels):
        row = f"{label:<20}" + "".join(f"{results[d].get(k, float('nan')):>{col_w}.4f}" for d in defenders)
        print(row)
