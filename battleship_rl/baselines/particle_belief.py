"""
battleship_rl/baselines/particle_belief.py
============================================
Particle Belief Baseline for Battleship.

This is the mathematically principled upgrade over HeuristicProbMapAgent.
The ProbMap agent samples hits/misses to create a heat map. This baseline
maintains an explicit particle filter (set of board hypotheses), applies strict
constraint filtering at every observation, and acts by marginalizing over the
particles to get the probability of each cell being occupied.

Constraints enforced per particle:
  • MISS cells          → no ship cell may overlap a miss
  • HIT cells (unsunk)  → at least one ship must cover each hit; ship not fully-hit
  • SUNK cells          → exactly the set of hit cells for that ship-id; fully hit

Unlike the heuristic, which re-samples particles from scratch each step,
this baseline:
  1. Maintains a pool of P particles across steps.
  2. At each new observation, FILTERS the pool by rejecting particles
     inconsistent with the new constraint.
  3. Re-samples (bootstraps) from the surviving pool to maintain P particles.
  4. If the pool collapses to zero, falls back to fresh sampling.

This is a Sequential Monte Carlo (SMC) / Particle Filter approach.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


class ParticleBeliefAgent:
    """Particle filter attacker for Battleship.

    Parameters
    ----------
    board_size      : int or (H, W)
    ships           : list of ship lengths, e.g. [5, 4, 3, 3, 2]
    rng             : numpy Generator (default: default_rng())
    n_particles     : number of particles in the pool (default 300)
    max_resample    : max backtrack steps per layout sample (default 500)
    fallback_budget : if pool collapses and fresh sampling fails this many times,
                      use Hunt/Target fallback
    """

    def __init__(
        self,
        board_size: int | Sequence[int],
        ships: Sequence[int],
        rng: Optional[np.random.Generator] = None,
        n_particles: int = 300,
        max_resample: int = 500,
        fallback_budget: int = 50,
    ) -> None:
        if isinstance(board_size, int):
            self.height = self.width = board_size
        else:
            self.height, self.width = int(board_size[0]), int(board_size[1])
        self.ships = list(ships)
        self.num_ships = len(self.ships)
        self.rng = rng or np.random.default_rng()
        self.n_particles = n_particles
        self.max_resample = max_resample
        self.fallback_budget = fallback_budget

        # State
        self.hit_grid: np.ndarray = np.zeros((self.height, self.width), dtype=bool)
        self.miss_grid: np.ndarray = np.zeros((self.height, self.width), dtype=bool)
        self.sunk_set: Set[int] = set()
        self.hit_cells_by_ship: Dict[int, List[Tuple[int, int]]] = {}
        self.fallback_used: bool = False
        self.fallback_reason: str = ""
        # Particle pool: list of (H,W) bool arrays
        self._pool: List[np.ndarray] = []
        self._initialized: bool = False

    def reset(self) -> None:
        self.hit_grid = np.zeros((self.height, self.width), dtype=bool)
        self.miss_grid = np.zeros((self.height, self.width), dtype=bool)
        self.sunk_set = set()
        self.hit_cells_by_ship = {i: [] for i in range(self.num_ships)}
        self.fallback_used = False
        self.fallback_reason = ""
        self._pool = []
        self._initialized = False

    # ------------------------------------------------------------------
    # Sampling a single layout via backtracking
    # ------------------------------------------------------------------
    def _is_valid(self, r: int, c: int, length: int, orient: int, occupied: np.ndarray) -> bool:
        if orient == 0:
            if c + length > self.width:
                return False
            if np.any(occupied[r, c:c + length]) or np.any(self.miss_grid[r, c:c + length]):
                return False
        else:
            if r + length > self.height:
                return False
            if np.any(occupied[r:r + length, c]) or np.any(self.miss_grid[r:r + length, c]):
                return False
        return True

    def _local_ok(self, r: int, c: int, length: int, orient: int, known_hits: List[Tuple[int, int]]) -> bool:
        if orient == 0:
            cells = {(r, c + i) for i in range(length)}
        else:
            cells = {(r + i, c) for i in range(length)}
        for hr, hc in known_hits:
            if (hr, hc) not in cells:
                return False
        if all(self.hit_grid[rr, cc] for rr, cc in cells):
            return False  # not sunk → cannot be fully hit
        return True

    def _backtrack(self, idx: int, ship_order: List[int], occupied: np.ndarray, steps: List[int]) -> bool:
        steps[0] += 1
        if steps[0] > self.max_resample:
            return False
        if idx == len(ship_order):
            return True
        ship_id = ship_order[idx]
        length = self.ships[ship_id]
        known_hits = self.hit_cells_by_ship[ship_id]
        candidates: List[Tuple[int, int, int]] = []
        for ro in range(self.height):
            for co in range(self.width - length + 1):
                if self._is_valid(ro, co, length, 0, occupied) and self._local_ok(ro, co, length, 0, known_hits):
                    candidates.append((ro, co, 0))
        for co in range(self.width):
            for ro in range(self.height - length + 1):
                if self._is_valid(ro, co, length, 1, occupied) and self._local_ok(ro, co, length, 1, known_hits):
                    candidates.append((ro, co, 1))
        if not candidates:
            return False
        self.rng.shuffle(candidates)
        for ro, co, orient in candidates:
            if orient == 0:
                cells = [(ro, co + i) for i in range(length)]
            else:
                cells = [(ro + i, co) for i in range(length)]
            for rr, cc in cells:
                occupied[rr, cc] = True
            if self._backtrack(idx + 1, ship_order, occupied, steps):
                return True
            for rr, cc in cells:
                occupied[rr, cc] = False
        return False

    def _sample_layout(self) -> Optional[np.ndarray]:
        occupied = np.zeros((self.height, self.width), dtype=bool)
        # Anchor sunk ships
        for ship_id in self.sunk_set:
            for hr, hc in self.hit_cells_by_ship[ship_id]:
                occupied[hr, hc] = True
        active = [i for i in range(self.num_ships) if i not in self.sunk_set]
        order = list(self.rng.permutation(active))
        steps = [0]
        if self._backtrack(0, order, occupied, steps):
            if np.all(self.hit_grid <= occupied):
                return occupied
        return None

    # ------------------------------------------------------------------
    # Particle pool management
    # ------------------------------------------------------------------
    def _build_pool(self) -> None:
        """Fill the particle pool from scratch."""
        self._pool = []
        budget = self.fallback_budget * self.n_particles
        attempts = 0
        while len(self._pool) < self.n_particles and attempts < budget:
            attempts += 1
            layout = self._sample_layout()
            if layout is not None:
                self._pool.append(layout.copy())

    def _filter_pool(self) -> None:
        """Remove particles that violate current constraints."""
        valid = []
        for particle in self._pool:
            if np.all(self.hit_grid <= particle) and not np.any(particle & self.miss_grid):
                valid.append(particle)
        self._pool = valid

    def _resample(self) -> None:
        """Bootstrap to n_particles from current (potentially collapsed) pool."""
        if len(self._pool) == 0:
            self._build_pool()
            return
        # Bootstrap with replacement
        indices = self.rng.integers(0, len(self._pool), size=self.n_particles)
        self._pool = [self._pool[i].copy() for i in indices]

    # ------------------------------------------------------------------
    def _update_from_obs(self, obs: np.ndarray, info: Optional[dict]) -> None:
        new_hits = (obs[0] > 0.5) & (~self.hit_grid)
        self.hit_grid = obs[0] > 0.5
        self.miss_grid = obs[1] > 0.5
        if info is not None:
            o_type = info.get("outcome_type")
            o_ship = info.get("outcome_ship_id")
            if o_type in ("HIT", "SUNK") and o_ship is not None:
                for r, c in np.argwhere(new_hits):
                    self.hit_cells_by_ship[int(o_ship)].append((int(r), int(c)))
            if o_type == "SUNK" and o_ship is not None:
                self.sunk_set.add(int(o_ship))

    def _compute_prob_map(self) -> np.ndarray:
        if len(self._pool) == 0:
            return np.zeros((self.height, self.width), dtype=np.float32)
        prob = np.mean(np.stack(self._pool, axis=0), axis=0)
        return prob.astype(np.float32)

    def _fallback(self, mask: np.ndarray) -> int:
        """Hunt/Target fallback when particle pool collapses."""
        # Target: any adjacent valid cell of a hit
        hit_cells = np.argwhere(self.hit_grid & mask)
        for r, c in self.rng.permutation(hit_cells):
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < self.height and 0 <= nc < self.width and mask[nr, nc]:
                    return nr * self.width + nc
        # Hunt: pick random valid cell
        valid = np.argwhere(mask)
        if len(valid) == 0:
            return 0
        r, c = valid[int(self.rng.integers(0, len(valid)))]
        return int(r) * self.width + int(c)

    # ------------------------------------------------------------------
    def act(self, obs: np.ndarray, info: Optional[dict] = None) -> dict:
        self.fallback_used = False
        self.fallback_reason = ""
        self._update_from_obs(obs, info)

        # Initialize or filter the pool
        if not self._initialized:
            self._build_pool()
            self._initialized = True
        else:
            self._filter_pool()
            if len(self._pool) < self.n_particles // 2:
                self._resample()

        mask_2d = ~(self.hit_grid | self.miss_grid)
        mask_flat = mask_2d.flatten()

        prob_map = self._compute_prob_map()
        prob_masked = prob_map * mask_2d

        if prob_masked.max() > 0:
            best = np.argwhere(prob_masked == prob_masked.max())
            r, c = best[self.rng.integers(0, len(best))]
            return {"action": int(r) * self.width + int(c), "fallback_used": False, "fallback_reason": ""}

        self.fallback_used = True
        self.fallback_reason = "pool_collapsed_or_zero_prob"
        return {"action": self._fallback(mask_2d), "fallback_used": True, "fallback_reason": self.fallback_reason}
