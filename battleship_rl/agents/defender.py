"""
battleship_rl/agents/defender.py
==================================
Defender strategy classes for the Battleship adversarial RL benchmark.

All biased defenders use the K=200 vectorized proposal-sampler approach:
  1. Generate K random legal placements using stride-tricks validity masks (O(K) not O(H*W*L))
  2. Score each proposal by the defender's criterion
  3. Weighted sample (softmax or direct probability)

This avoids the _enumerate_candidates O(H*W*L) Python loop that caused multi-minute hangs.

Defender ladder:
  UNIFORM   UniformRandomDefender  -- benchmark baseline
  EDGE      EdgeBiasedDefender     -- ships prefer boundary cells
  CLUSTER   ClusteredDefender      -- ships huddle near each other
  SPREAD    SpreadDefender         -- ships maximise inter-ship distance
  PARITY    ParityDefender         -- largest ship locked to top-left quadrant
  ADVERSARIAL AdversarialDefender  -- RL-trained (requires model_path)
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from battleship_rl.envs.placement import (
    _normalize_board_size,
    _normalize_ships,
    decode_placement_action,
    sample_placement,
)


# ---------------------------------------------------------------------------
# Shared fast placement utility (stride tricks, zero Python loops per cell)
# ---------------------------------------------------------------------------

def _valid_mask_h(occupied: np.ndarray, length: int) -> np.ndarray:
    """(H, W-L+1) bool — True where horizontal ship of `length` is legal."""
    H, W = occupied.shape
    if W < length:
        return np.zeros((H, 0), dtype=bool)
    windows = np.lib.stride_tricks.sliding_window_view(occupied, length, axis=1)
    return ~windows.any(axis=2)


def _valid_mask_v(occupied: np.ndarray, length: int) -> np.ndarray:
    """(H-L+1, W) bool — True where vertical ship of `length` is legal."""
    H, W = occupied.shape
    if H < length:
        return np.zeros((0, W), dtype=bool)
    windows = np.lib.stride_tricks.sliding_window_view(occupied, length, axis=0)
    return ~windows.any(axis=2)


def _propose(
    rng: np.random.Generator,
    occupied: np.ndarray,
    length: int,
    K: int = 200,
    score_fn=None,
    greedy: bool = False,
) -> tuple[int, int, int]:
    """Sample one placement (r, c, orient) via K-proposal scoring.

    score_fn(rs, cs, ors, occupied, H, W, length) -> float array shape (N,)
    If None, samples uniformly.
    If greedy=True and score_fn is provided, returns the argmax proposal.
    Returns (r, c, orient)  where orient 0=horizontal, 1=vertical.
    """
    H, W = occupied.shape
    vh = _valid_mask_h(occupied, length)  # (H, W-L+1)
    vv = _valid_mask_v(occupied, length)  # (H-L+1, W)

    rsh, csh = np.nonzero(vh)
    rsv, csv = np.nonzero(vv)
    rs_all = np.concatenate([rsh, rsv])
    cs_all = np.concatenate([csh, csv])
    ors_all = np.concatenate([
        np.zeros(len(rsh), dtype=np.int8),
        np.ones(len(rsv), dtype=np.int8),
    ])

    n_total = len(rs_all)
    if n_total == 0:
        raise ValueError("No legal placements available.")

    # Sub-sample K proposals (or all if fewer)
    if n_total <= K:
        idx_pool = np.arange(n_total)
    else:
        idx_pool = rng.choice(n_total, size=K, replace=False)

    rs = rs_all[idx_pool]
    cs = cs_all[idx_pool]
    ors = ors_all[idx_pool]

    if score_fn is not None:
        w = score_fn(rs, cs, ors, occupied, H, W, length)
        if greedy:
            chosen = int(np.argmax(w))
            return int(rs[chosen]), int(cs[chosen]), int(ors[chosen])

        w = np.maximum(w, 0.0)
        total = w.sum()
        if total > 0:
            p = w / total
        else:
            p = None
    else:
        p = None

    chosen = rng.choice(len(rs), p=p)
    return int(rs[chosen]), int(cs[chosen]), int(ors[chosen])


def _apply(ship_id_grid: np.ndarray, r: int, c: int, orient: int, length: int, ship_id: int) -> None:
    """Place ship_id onto ship_id_grid (in-place)."""
    if orient == 0:
        ship_id_grid[r, c:c + length] = ship_id
    else:
        ship_id_grid[r:r + length, c] = ship_id


def _occupied_from_grid(ship_id_grid: np.ndarray) -> np.ndarray:
    return ship_id_grid >= 0


# ---------------------------------------------------------------------------
# Vectorized weight helpers
# ---------------------------------------------------------------------------

def _edge_weights(H: int, W: int) -> np.ndarray:
    """(H, W) grid: weight 1/(dist_to_edge+1)."""
    rs, cs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    dist = np.minimum(np.minimum(rs, H - 1 - rs), np.minimum(cs, W - 1 - cs))
    return (1.0 / (dist + 1)).astype(np.float32)


def _placement_center(rs: np.ndarray, cs: np.ndarray, ors: np.ndarray, length: int
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Fractional center (r_c, c_c) for each proposal."""
    half = (length - 1) / 2.0
    r_c = np.where(ors == 0, rs.astype(float), rs + half)
    c_c = np.where(ors == 0, cs + half, cs.astype(float))
    return r_c, c_c


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseDefender:
    def sample_layout(
        self,
        board_size: int | Sequence[int],
        ships: Sequence[int] | dict,
        rng: np.random.Generator,
    ) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# UNIFORM
# ---------------------------------------------------------------------------

class UniformRandomDefender(BaseDefender):
    """UNIFORM: ships placed uniformly at random."""

    def sample_layout(self, board_size, ships, rng):
        return sample_placement(board_size, ships, rng)


# ---------------------------------------------------------------------------
# EDGE  (formerly BiasedDefender — kept as alias)
# ---------------------------------------------------------------------------

class EdgeBiasedDefender(BaseDefender):
    """EDGE: ships prefer boundary cells (1/(dist+1) weighting)."""
    _WEIGHT_CACHE: dict[tuple, np.ndarray] = {}

    def _weight_grid(self, H: int, W: int) -> np.ndarray:
        key = (H, W)
        if key not in self._WEIGHT_CACHE:
            self._WEIGHT_CACHE[key] = _edge_weights(H, W)
        return self._WEIGHT_CACHE[key]

    def sample_layout(self, board_size, ships, rng):
        H, W = _normalize_board_size(board_size)
        ship_lengths = _normalize_ships(ships)
        ship_id_grid = -np.ones((H, W), dtype=np.int32)
        wgrid = self._weight_grid(H, W)

        for ship_id, length in enumerate(ship_lengths):
            def score(rs, cs, ors, occ, H, W, L, _wg=wgrid, _L=length):
                # Mean edge-weight of cells in each proposal
                r_c, c_c = _placement_center(rs, cs, ors, _L)
                # Sample-center weight (use floor coords as proxy)
                r_i = np.clip(r_c.astype(int), 0, H - 1)
                c_i = np.clip(c_c.astype(int), 0, W - 1)
                return _wg[r_i, c_i]

            r, c, o = _propose(rng, _occupied_from_grid(ship_id_grid), length, score_fn=score)
            _apply(ship_id_grid, r, c, o, length, ship_id)

        return ship_id_grid


# Keep old name as alias for backward compatibility
BiasedDefender = EdgeBiasedDefender


# ---------------------------------------------------------------------------
# CLUSTER
# ---------------------------------------------------------------------------

class ClusteredDefender(BaseDefender):
    """CLUSTER: ships prefer to be placed near already-placed ships.
    Scores proposals by proximity to centroid of occupied cells using exponential weight.
    """

    def sample_layout(self, board_size, ships, rng):
        H, W = _normalize_board_size(board_size)
        ship_lengths = _normalize_ships(ships)
        ship_id_grid = -np.ones((H, W), dtype=np.int32)
        
        center_r, center_c = H / 2.0, W / 2.0

        for ship_id, length in enumerate(ship_lengths):
            occupied = _occupied_from_grid(ship_id_grid)
            placed = np.argwhere(occupied)
            if len(placed) == 0:
                # First ship: Bias towards board center to anchor the cluster away from edges
                def score_center(rs, cs, ors, occ, H, W, L, cr=center_r, cc=center_c):
                    r_c, c_c = _placement_center(rs, cs, ors, L)
                    dist = np.hypot(r_c - cr, c_c - cc)
                    return np.exp(-1.0 * dist)
                r, c, o = _propose(rng, occupied, length, score_fn=score_center)
            else:
                centroid_r = placed[:, 0].mean()
                centroid_c = placed[:, 1].mean()

                def score_cluster(rs, cs, ors, occ, H, W, L, cr=centroid_r, cc=centroid_c):
                    r_c, c_c = _placement_center(rs, cs, ors, L)
                    dist = np.hypot(r_c - cr, c_c - cc)
                    return np.exp(-2.0 * dist)

                r, c, o = _propose(rng, occupied, length, score_fn=score_cluster)

            _apply(ship_id_grid, r, c, o, length, ship_id)

        return ship_id_grid


# ---------------------------------------------------------------------------
# SPREAD
# ---------------------------------------------------------------------------

class SpreadDefender(BaseDefender):
    """SPREAD: ships maximise minimum distance from already-placed ships.
    Scores proposals greedily by argmax min cell-to-placed-ship distance.
    """

    def sample_layout(self, board_size, ships, rng):
        H, W = _normalize_board_size(board_size)
        ship_lengths = _normalize_ships(ships)
        ship_id_grid = -np.ones((H, W), dtype=np.int32)

        # Precompute row/col index grids for distance calculations
        rows_idx, cols_idx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        for ship_id, length in enumerate(ship_lengths):
            occupied = _occupied_from_grid(ship_id_grid)
            placed = np.argwhere(occupied)  # (M, 2)
            if len(placed) == 0:
                r, c, o = _propose(rng, occupied, length, score_fn=None)
            else:
                # Distance from each board cell to nearest placed cell
                # Vectorized: (M, H, W) differences, then min over M
                dr = (rows_idx[None, :, :] - placed[:, 0:1, None]).astype(np.float32)  # (M, H, W)
                dc = (cols_idx[None, :, :] - placed[:, 1:2, None]).astype(np.float32)
                dist_map = np.hypot(dr, dc).min(axis=0)  # (H, W) min dist to any placed cell

                def score_spread(rs, cs, ors, occ, H, W, L, dm=dist_map):
                    r_c, c_c = _placement_center(rs, cs, ors, L)
                    r_i = np.clip(r_c.astype(int), 0, H - 1)
                    c_i = np.clip(c_c.astype(int), 0, W - 1)
                    return dm[r_i, c_i]

                # Use greedy=True to explicitly separate ships as much as possible
                r, c, o = _propose(rng, occupied, length, score_fn=score_spread, greedy=True)

            _apply(ship_id_grid, r, c, o, length, ship_id)

        return ship_id_grid


# ---------------------------------------------------------------------------
# PARITY  (symmetry-broken / quadrant-constrained)
# ---------------------------------------------------------------------------

class ParityDefender(BaseDefender):
    """PARITY: largest ship (index 0) is constrained to top-left quadrant.
    Remaining ships are uniform within legal positions.
    """

    def sample_layout(self, board_size, ships, rng):
        H, W = _normalize_board_size(board_size)
        ship_lengths = _normalize_ships(ships)
        ship_id_grid = -np.ones((H, W), dtype=np.int32)
        qH, qW = H // 2, W // 2  # quadrant boundary (exclusive)

        for ship_id, length in enumerate(ship_lengths):
            occupied = _occupied_from_grid(ship_id_grid)

            if ship_id == 0:
                # Restrict to top-left quadrant
                def score(rs, cs, ors, occ, H, W, L, qH=qH, qW=qW):
                    # Proposals fully inside quadrant get weight 1, others 0
                    r_c, c_c = _placement_center(rs, cs, ors, L)
                    in_quad = (r_c < qH - 0.5) & (c_c < qW - 0.5)
                    return in_quad.astype(float)

                r, c, o = _propose(rng, occupied, length, score_fn=score)
            else:
                r, c, o = _propose(rng, occupied, length, score_fn=None)

            _apply(ship_id_grid, r, c, o, length, ship_id)

        return ship_id_grid


# ---------------------------------------------------------------------------
# ADVERSARIAL  (RL-trained, requires pre-trained model)
# ---------------------------------------------------------------------------

class AdversarialDefender(BaseDefender):
    """Defender that uses an RL policy to place ships.
    Requires a pre-trained SB3/MaskablePPO model (trained on BattleshipPlacementEnv).
    Falls back to UniformRandomDefender if model is unavailable.
    """

    def __init__(self, model_path: str | None = None, deterministic: bool = False):
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

    def sample_layout(self, board_size, ships, rng):
        if self.model is None:
            return sample_placement(board_size, ships, rng)

        H, W = _normalize_board_size(board_size)
        ship_lengths = _normalize_ships(ships)
        board = np.full((H, W), -1, dtype=np.int32)

        for ship_idx, length in enumerate(ship_lengths):
            obs = np.zeros((3, H, W), dtype=np.float32)
            obs[0] = (board != -1).astype(np.float32)
            obs[1] = float(length) / max(ship_lengths)
            obs[2] = float(ship_idx) / len(ship_lengths)

            mask = np.zeros(H * W * 2, dtype=bool)
            for r in range(H):
                for c in range(W):
                    if c + length <= W and np.all(board[r, c:c + length] == -1):
                        mask[r * W + c] = True
                    if r + length <= H and np.all(board[r:r + length, c] == -1):
                        mask[H * W + r * W + c] = True

            if not np.any(mask):
                return sample_placement(board_size, ships, rng)

            action, _ = self.model.predict(obs, action_masks=mask, deterministic=self.deterministic)
            r, c, orientation = decode_placement_action(int(action), H, W)
            if orientation == 0:
                board[r, c:c + length] = ship_idx
            else:
                board[r:r + length, c] = ship_idx

        return board
