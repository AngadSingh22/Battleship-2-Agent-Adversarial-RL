"""
battleship_rl/eval/defender_metrics.py
=======================================
Placement-distribution metrics for defender ladder evaluation.

All metrics are computed from N samples of ship-occupancy grids.

Mathematical definitions (per manager review):
  p(c)  = empirical P(cell c occupied)  -- NOT a distribution, sums to expected ship area
  q(c)  = p(c) / sum_c p(c)             -- normalized to a proper probability distribution
  u(c)  = 1 / (H*W)                     -- uniform reference

Metrics:
  edge_occupancy    fraction of ship cells on board boundary
  mean_dist_to_edge mean min-dist to nearest edge, normalized by floor(min(H,W)/2)
  occupancy_entropy H(q) = -sum_c q(c) log2 q(c)   [bits]
  kl_from_uniform   D_KL(q||u) = sum_c q(c) log2(q(c)/u(c))   [bits]

Fast sampler: uses stride-trick vectorized numpy. Generates K=500 legal placements
per ship via sliding-window validity check — no Python loops over cells.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Vectorized placement validity (stride tricks, zero Python loops)
# ---------------------------------------------------------------------------

def _valid_horizontal(occupied: np.ndarray, blocked: np.ndarray, length: int) -> np.ndarray:
    """(H, W-L+1) bool array: True where horizontal ship of `length` can go."""
    H, W = occupied.shape
    if W < length:
        return np.zeros((H, 0), dtype=bool)
    combined = occupied | blocked
    windows = np.lib.stride_tricks.sliding_window_view(combined, length, axis=1)  # (H, W-L+1, L)
    return ~windows.any(axis=2)


def _valid_vertical(occupied: np.ndarray, blocked: np.ndarray, length: int) -> np.ndarray:
    """(H-L+1, W) bool array: True where vertical ship of `length` can go."""
    H, W = occupied.shape
    if H < length:
        return np.zeros((0, W), dtype=bool)
    combined = occupied | blocked
    windows = np.lib.stride_tricks.sliding_window_view(combined, length, axis=0)  # (H-L+1, W, L)
    return ~windows.any(axis=2)


def _all_valid_positions(occupied: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (rs, cs, orients) arrays of all valid placement starts.

    orients: 0 = horizontal, 1 = vertical
    """
    dummy_blocked = np.zeros_like(occupied)
    vh = _valid_horizontal(occupied, dummy_blocked, length)
    vv = _valid_vertical(occupied, dummy_blocked, length)
    rh, ch = np.nonzero(vh)
    rv, cv = np.nonzero(vv)
    rs = np.concatenate([rh, rv])
    cs = np.concatenate([ch, cv])
    ors = np.concatenate([np.zeros(len(rh), dtype=np.int8), np.ones(len(rv), dtype=np.int8)])
    return rs, cs, ors


# ---------------------------------------------------------------------------
# Fast independent placement sampler for metrics
# ---------------------------------------------------------------------------

def _fast_sample_layout(rng: np.random.Generator, H: int, W: int, ships: list[int],
                        weights_fn=None) -> np.ndarray:
    """Sample a layout using fully vectorized candidate selection.

    weights_fn: callable(H, W, rs, cs, ors, ship_id, occupied) -> 1-D weight array
                If None, uniform sampling.
    Returns occupied (H, W) bool array.
    """
    occupied = np.zeros((H, W), dtype=bool)

    for ship_id, length in enumerate(ships):
        rs, cs, ors = _all_valid_positions(occupied, length)
        if len(rs) == 0:
            # fallback: skip (shouldn't happen with standard Battleship configs)
            continue

        if weights_fn is not None:
            w = weights_fn(H, W, rs, cs, ors, ship_id, occupied)
            w = np.maximum(w, 0.0)
            total = w.sum()
            if total > 0:
                p = w / total
            else:
                p = np.ones(len(rs)) / len(rs)
        else:
            p = None  # uniform

        idx = rng.choice(len(rs), p=p)
        r, c, o = int(rs[idx]), int(cs[idx]), int(ors[idx])
        if o == 0:
            occupied[r, c:c + length] = True
        else:
            occupied[r:r + length, c] = True

    return occupied


def _uniform_layout(rng, H, W, ships):
    return _fast_sample_layout(rng, H, W, ships)


# ---------------------------------------------------------------------------
# Metric computation (vectorized)
# ---------------------------------------------------------------------------

def _compute_raw_marginal(grids: list[np.ndarray]) -> np.ndarray:
    """p(c) = mean occupancy over N samples."""
    stack = np.stack(grids, axis=0).astype(np.float32)  # (N, H, W)
    return stack.mean(axis=0)  # (H, W)


def _normalize_marginal(p: np.ndarray) -> np.ndarray:
    """q(c) = p(c) / sum p(c)  — proper distribution over cell mass."""
    total = p.sum()
    if total <= 0.0:
        H, W = p.shape
        return np.full_like(p, 1.0 / (H * W))
    return p / total


def edge_occupancy(p: np.ndarray) -> float:
    """Fraction of expected ship cells on boundary rows/cols. Computed on raw p."""
    H, W = p.shape
    boundary = np.zeros((H, W), dtype=bool)
    boundary[0, :] = boundary[-1, :] = True
    boundary[:, 0] = boundary[:, -1] = True
    edge_mass = p[boundary].sum()
    total_mass = p.sum()
    return float(edge_mass / total_mass) if total_mass > 0 else 0.0


def mean_dist_to_edge_normalized(p: np.ndarray) -> float:
    """Mean min-distance to nearest edge, weighted by p(c), normalized to [0,1].

    Normalization factor: floor(min(H,W)/2).
    """
    H, W = p.shape
    max_dist = max(1, math.floor(min(H, W) / 2))
    rs, cs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    dist_map = np.minimum(  # (H, W)
        np.minimum(rs, H - 1 - rs),
        np.minimum(cs, W - 1 - cs)
    ).astype(np.float32)
    total = p.sum()
    if total <= 0:
        return 0.0
    raw_mean = float((p * dist_map).sum() / total)
    return raw_mean / max_dist


def occupancy_entropy(q: np.ndarray) -> float:
    """H(q) = -sum_c q(c) log2 q(c)  [bits]. q must be normalized."""
    eps = 1e-12
    q_safe = np.clip(q, eps, 1.0)
    return float(-np.sum(q_safe * np.log2(q_safe)))


def kl_from_uniform(q: np.ndarray) -> float:
    """D_KL(q || u) where u = 1/(H*W).  [bits]. q must be normalized.
    NOTE: This is a descriptive stat, NOT the shift metric.
    Even the UNIFORM defender will have nonzero KL because ships create
    non-uniform occupancy. Use jsd_from_reference for shift detection.
    """
    H, W = q.shape
    u = 1.0 / (H * W)
    eps = 1e-12
    q_safe = np.clip(q, eps, 1.0)
    return float(np.sum(q_safe * np.log2(q_safe / u)))


def jsd_from_reference(q: np.ndarray, q_ref: np.ndarray) -> float:
    """Jensen-Shannon divergence [bits] between q and reference q_ref.

    JSD(q || q_ref) = 0.5 * KL(q || m) + 0.5 * KL(q_ref || m)
    where m = 0.5*(q+q_ref).

    When q_ref is the UNIFORM-defender marginal, JSD=0 for UNIFORM
    by construction and positive for any distinct distribution.
    Always in [0, 1] (in natural-log bits, upper bound = 1 bit).
    """
    eps = 1e-12
    q_s = np.clip(q.ravel(), eps, 1.0)
    r_s = np.clip(q_ref.ravel(), eps, 1.0)
    # Re-normalize both (they should already sum to 1)
    q_s = q_s / q_s.sum()
    r_s = r_s / r_s.sum()
    m = 0.5 * (q_s + r_s)
    kl_qm = float(np.sum(q_s * np.log2(q_s / m)))
    kl_rm = float(np.sum(r_s * np.log2(r_s / m)))
    return 0.5 * kl_qm + 0.5 * kl_rm


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_metrics(grids: list[np.ndarray], q_ref: np.ndarray | None = None) -> dict:
    """Compute all four metrics from a list of occupancy grids.

    q_ref: normalized marginal from the UNIFORM defender (same board/ships).
           When provided, `shift_metric` = JSD(q_mode || q_ref), which is 0
           for UNIFORM by construction. When None, falls back to KL vs flat.
    """
    p = _compute_raw_marginal(grids)
    q = _normalize_marginal(p)
    if q_ref is not None:
        shift = jsd_from_reference(q, q_ref)
    else:
        shift = kl_from_uniform(q)  # descriptive fallback
    return {
        "edge_occupancy":    edge_occupancy(p),
        "mean_dist_to_edge": mean_dist_to_edge_normalized(p),
        "occupancy_entropy": occupancy_entropy(q),
        "kl_from_uniform":   kl_from_uniform(q),   # kept as descriptive stat
        "shift_metric":      shift,                 # JSD vs UNIFORM-defender (0 for UNIFORM)
        "q_marginal":        q,                     # expose for downstream JSD computation
    }


def summarize_defender(
    defender,
    board_size: int | tuple[int, int],
    ships: list[int],
    n_samples: int = 500,
    seed: int = 0,
    q_ref: np.ndarray | None = None,
) -> dict:
    """Sample `n_samples` layouts from `defender` and return all metrics.

    q_ref: normalized marginal from UNIFORM defender for JSD (shift_metric).
           Pass result['q_marginal'] from UNIFORM run as q_ref to downstream calls.
           When q_ref=None, shift_metric falls back to KL vs flat uniform.
    """
    if isinstance(board_size, int):
        H = W = board_size
    else:
        H, W = board_size
    rng = np.random.default_rng(seed)
    grids = []
    for _ in range(n_samples):
        grid = defender.sample_layout((H, W), ships, rng)
        grids.append((grid >= 0).astype(np.float32))
    return compute_metrics(grids, q_ref=q_ref)


def format_metrics_row(name: str, metrics: dict, is_reference: bool = False) -> str:
    shift = metrics.get("shift_metric", metrics.get("kl_from_uniform", float("nan")))
    if is_reference:
        flag = "  (reference)"
    elif shift < 0.005:
        flag = "  ⚠ weak_shift"
    else:
        flag = ""
    return (
        f"| {name:<10} "
        f"| {metrics.get('edge_occupancy', float('nan')):>6.3f} "
        f"| {metrics.get('mean_dist_to_edge', float('nan')):>6.3f} "
        f"| {metrics.get('occupancy_entropy', float('nan')):>6.3f} "
        f"| {shift:>8.5f}{flag} |"
    )
