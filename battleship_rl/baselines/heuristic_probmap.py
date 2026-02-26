from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple

import numpy as np


class HeuristicProbMapAgent:
    """
    Sequential Monte Carlo (SMC) agent for Battleship.
    
    Maintains joint consistency:
    - Misses must be empty.
    - Hits must be occupied.
    - Sunk ships must be fully fully covered by hits.
    - UNSUNK ships must NOT be fully covered by hits.
    - All hits must be explained by the union of ships.
    
    Algorithm:
    1. Update internal state (hits, misses, sunk_set) from observation.
    2. Sample N consistent full-board layouts (particles) using randomized backtracking.
    3. Aggregate ship occupancy counts from consistent layouts.
    4. Normalize to get probability map.
    5. Fallback to 'Hunt/Target' heuristic logic if sampling fails.
    """

    def __init__(
        self,
        board_size: int | Sequence[int],
        ships: Sequence[int],
        rng: np.random.Generator | None = None,
        num_particles: int = 2000,
        max_samples: int = 100,
        max_backtrack_steps: int = 2000,
    ) -> None:
        if isinstance(board_size, int):
            self.height = board_size
            self.width = board_size
        else:
            self.height = int(board_size[0])
            self.width = int(board_size[1])
        
        self.ships = list(ships) # List of lengths
        self.num_ships = len(self.ships)
        self.rng = rng or np.random.default_rng()
        self.num_particles = num_particles
        self.max_samples = max_samples
        self.max_backtrack_steps = max_backtrack_steps
        
        # Pyre static type initializations
        self.hit_grid = np.zeros((self.height, self.width), dtype=bool)
        self.miss_grid = np.zeros((self.height, self.width), dtype=bool)
        self.sunk_set: Set[int] = set()
        self.fallback_used = False
        self.fallback_reason = ""
        self._backtrack_steps = 0
        self.rejection_counts = {
            "bounds_violation": 0,
            "overlap_violation": 0,
            "miss_violation": 0,
            "hit_violation": 0,
            "not_sunk_fully_hit_violation": 0,
        }
        from typing import Dict, Tuple, List
        self.hit_cells_by_ship_id: Dict[int, List[Tuple[int, int]]] = {}
        
        self.reset()

    def reset(self) -> None:
        self.hit_grid = np.zeros((self.height, self.width), dtype=bool)
        self.miss_grid = np.zeros((self.height, self.width), dtype=bool)
        self.sunk_set: Set[int] = set()
        self.fallback_used = False
        self.fallback_reason = ""
        self._backtrack_steps = 0
        from typing import Dict
        self.hit_cells_by_ship_id: Dict[int, List[Tuple[int, int]]] = {
            i: [] for i in range(self.num_ships)
        }

    def _update_from_obs(self, obs: np.ndarray, info: dict | None) -> None:
        # Obs is (3, H, W): 0=Hit, 1=Miss, 2=Unknown
        new_hits = (obs[0] > 0.5) & (~self.hit_grid)
        self.hit_grid = obs[0] > 0.5
        self.miss_grid = obs[1] > 0.5

        if info is not None:
            o_type = info.get("outcome_type")
            o_ship = info.get("outcome_ship_id")
            if o_type in ("HIT", "SUNK") and o_ship is not None:
                hit_coords = np.argwhere(new_hits)
                for r, c in hit_coords:
                    self.hit_cells_by_ship_id[int(o_ship)].append((int(r), int(c)))
            if o_type == "SUNK" and o_ship is not None:
                self.sunk_set.add(int(o_ship))

    def _is_valid_placement(
        self, 
        r: int, 
        c: int, 
        length: int, 
        orientation: int, # 0=Horz, 1=Vert
        occupied: np.ndarray
    ) -> bool:
        if orientation == 0: # Horizontal
            if c + length > self.width:
                self.rejection_counts["bounds_violation"] += 1
                return False
            # Check overlap with other ships
            if np.any(occupied[r, c : c + length]):
                self.rejection_counts["overlap_violation"] += 1
                return False
            # Check Misses (obstacles)
            if np.any(self.miss_grid[r, c : c + length]):
                self.rejection_counts["miss_violation"] += 1
                return False
        else: # Vertical
            if r + length > self.height:
                self.rejection_counts["bounds_violation"] += 1
                return False
            if np.any(occupied[r : r + length, c]):
                self.rejection_counts["overlap_violation"] += 1
                return False
            if np.any(self.miss_grid[r : r + length, c]):
                self.rejection_counts["miss_violation"] += 1
                return False
        return True

    def _satisfies_local_constraints(
        self, 
        r: int, 
        c: int, 
        length: int, 
        orientation: int, 
        known_hits: List[Tuple[int, int]]
    ) -> bool:
        """
        Check if an active (unsunk) ship placement includes all its known hits
        and is NOT fully covered by hits.
        """
        if orientation == 0:
            cells = {(r, c + i) for i in range(length)}
        else:
            cells = {(r + i, c) for i in range(length)}
            
        for hr, hc in known_hits:
            if (hr, hc) not in cells:
                self.rejection_counts["hit_violation"] += 1
                return False
                
        # Cannot be fully hit (as it's not sunk)
        if all(self.hit_grid[rr, cc] for rr, cc in cells):
            self.rejection_counts["not_sunk_fully_hit_violation"] += 1
            return False
            
        return True

    def _sample_layout(self) -> Optional[np.ndarray]:
        """
        Randomized backtracking to find ONE consistent layout.
        Returns: grid of 1.0s where ships are, or None if failed.
        """
        occupied = np.zeros((self.height, self.width), dtype=bool)
        
        # Lock in sunk ships permanently
        for ship_id in self.sunk_set:
            for hr, hc in self.hit_cells_by_ship_id[ship_id]:
                occupied[hr, hc] = True
                
        # Only search over ships that are NOT sunk
        active_ships = [i for i in range(self.num_ships) if i not in self.sunk_set]
        ship_order = self.rng.permutation(active_ships)
        
        self._backtrack_steps = 0
        
        if self._backtrack(0, ship_order, occupied):
            # Final Check: All global hits must be covered by ~something~
            if np.all(self.hit_grid <= occupied):
                return occupied.astype(np.float32)
        return None

    def _backtrack(self, idx: int, ship_order: np.ndarray, occupied: np.ndarray) -> bool:
        self._backtrack_steps += 1
        if self._backtrack_steps > self.max_backtrack_steps:
            return False
            
        if idx == len(ship_order):
            return True

        ship_id = ship_order[idx]
        length = self.ships[ship_id]
        known_hits = self.hit_cells_by_ship_id[ship_id]
        
        # Enumerate candidates for this ship
        candidates = []
        
        # Horizontal
        for r in range(self.height):
            for c in range(self.width - length + 1):
                if self._is_valid_placement(r, c, length, 0, occupied):
                    if self._satisfies_local_constraints(r, c, length, 0, known_hits):
                        candidates.append((r, c, 0))
        
        # Vertical
        for c in range(self.width):
            for r in range(self.height - length + 1):
                if self._is_valid_placement(r, c, length, 1, occupied):
                    if self._satisfies_local_constraints(r, c, length, 1, known_hits):
                        candidates.append((r, c, 1))

        if not candidates:
            return False

        self.rng.shuffle(candidates)
        
        # Try a subset of candidates to save time? Or strictly all?
        # For small boards, all is fine. For 10x10, maybe limit branching factor?
        # We'll try all but rely on early returns.
        
        for r, c, ori in candidates:
            # Place
            if ori == 0:
                occupied[r, c : c + length] = True
            else:
                occupied[r : r + length, c] = True
                
            if self._backtrack(idx + 1, ship_order, occupied):
                return True
                
            # Unplace (Backtrack)
            if ori == 0:
                occupied[r, c : c + length] = False
            else:
                occupied[r : r + length, c] = False
                
        return False

    def _compute_prob_map(self) -> np.ndarray:
        prob_map = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Try to collect N valid samples
        valid_samples = 0
        
        # Optimization: hit_grid check.
        # If total hits > total ship area, impossible (bug or weird state).
        
        for _ in range(self.max_samples): # Budget for attempts
            layout = self._sample_layout()
            if layout is not None:
                prob_map += layout
                valid_samples += 1
                if valid_samples >= self.num_particles:
                    break
        
        return prob_map

    def _fallback_action(self, mask: np.ndarray) -> int:
        """Hunt/Target logic if SMC fails."""
        # Target Mode: Adjacent to active hits only
        hit_cells = np.argwhere(self.hit_grid)
        candidate_cells = []
        for r, c in hit_cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.height and 0 <= cc < self.width and mask[rr, cc]:
                    candidate_cells.append((rr, cc))
        
        if candidate_cells:
            rr, cc = candidate_cells[self.rng.integers(0, len(candidate_cells))]
            return int(rr * self.width + cc)
            
        # Hunt Mode: Random valid parity (checkerboard) or pure random
        # Pure random on valid mask
        valid = np.argwhere(mask)
        if valid.size == 0:
            return 0 # Should be impossible if not truncated
            
        # Checkboard heuristic (parity)
        parity_candidates = []
        for r, c in valid:
            if (r + c) % 2 == 0: # minimal parity for smallest ship=2
                parity_candidates.append((r, c))
                
        if parity_candidates:
            rr, cc = parity_candidates[self.rng.integers(0, len(parity_candidates))]
            return int(rr * self.width + cc)
            
        rr, cc = valid[self.rng.integers(0, len(valid))]
        return int(rr * self.width + cc)

    def act(self, obs: np.ndarray, info: dict | None = None) -> dict:
        self.fallback_used = False
        self.fallback_reason = ""
        self._update_from_obs(obs, info)
        
        # Mask: 0 for Hit/Miss cells (already fired), 1 for Unknown
        mask = np.logical_not(np.logical_or(self.hit_grid, self.miss_grid))
        
        prob_map = self._compute_prob_map()
        
        # Mask prob_map (don't fire at known cells)
        valid_mask = mask.astype(bool)
        prob_map_masked = prob_map * valid_mask
        
        if prob_map_masked.max() > 0:
            # Greedy max probability
            best = np.argwhere(prob_map_masked == prob_map_masked.max())
            rr, cc = best[self.rng.integers(0, len(best))]
            action = int(rr * self.width + cc)
            return {"action": action, "fallback_used": False, "fallback_reason": ""}
            
        self.fallback_used = True
        self.fallback_reason = "sampling_budget_exhausted_or_zero_prob"
        action = self._fallback_action(mask)
        return {"action": action, "fallback_used": True, "fallback_reason": self.fallback_reason}
