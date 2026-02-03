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
        
        self.reset()

    def reset(self) -> None:
        self.hit_grid = np.zeros((self.height, self.width), dtype=bool)
        self.miss_grid = np.zeros((self.height, self.width), dtype=bool)
        self.sunk_set: Set[int] = set()

    def _update_from_obs(self, obs: np.ndarray, info: dict | None) -> None:
        # Obs is (3, H, W): 0=Hit, 1=Miss, 2=Unknown
        # We rely on channel 0 and 1.
        self.hit_grid = obs[0] > 0.5
        self.miss_grid = obs[1] > 0.5
        
        if info is not None and info.get("outcome_type") == "SUNK":
            ship_id = info.get("outcome_ship_id")
            if ship_id is not None:
                self.sunk_set.add(int(ship_id))

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
                return False
            # Check Misses (obstacles)
            if np.any(self.miss_grid[r, c : c + length]):
                return False
            # Check overlap with other ships
            if np.any(occupied[r, c : c + length]):
                return False
        else: # Vertical
            if r + length > self.height:
                return False
            if np.any(self.miss_grid[r : r + length, c]):
                return False
            if np.any(occupied[r : r + length, c]):
                return False
        return True

    def _satisfies_constraints(
        self, 
        r: int, 
        c: int, 
        length: int, 
        orientation: int, 
        is_sunk: bool
    ) -> bool:
        """
        Check if a placement satisfies Sunk/Not-Sunk constraints wrt Hits.
        """
        # Get cells covered by this ship
        if orientation == 0:
            ship_cells = self.hit_grid[r, c : c + length]
        else:
            ship_cells = self.hit_grid[r : r + length, c]
            
        # Sunk Constraint: MUST be fully hit
        if is_sunk:
            if not np.all(ship_cells):
                return False
        # Not Sunk Constraint: CANNOT be fully hit
        else:
            if np.all(ship_cells):
                return False
        return True

    def _sample_layout(self) -> Optional[np.ndarray]:
        """
        Randomized backtracking to find ONE consistent layout.
        Returns: grid of 1.0s where ships are, or None if failed.
        """
        # Order ships randomly to diversify samples
        ship_order = self.rng.permutation(self.num_ships)
        
        # Grid to track occupancy in this sample: 0=Empty, 1=Occupied
        occupied = np.zeros((self.height, self.width), dtype=bool)
        
        if self._backtrack(0, ship_order, occupied):
            # Final Check: All global hits must be explained
            # (Backtracking ensures no ship overlaps miss, and sunk constraints met)
            # But we must ensure every TRUE hit on board is covered by SOME ship.
            if np.all(self.hit_grid <= occupied):
                return occupied.astype(np.float32)
        return None

    def _backtrack(self, idx: int, ship_order: np.ndarray, occupied: np.ndarray) -> bool:
        if idx == self.num_ships:
            return True

        ship_id = ship_order[idx]
        length = self.ships[ship_id]
        is_sunk = ship_id in self.sunk_set
        
        # Enumerate candidates for this ship
        candidates = []
        
        # Horizontal
        for r in range(self.height):
            for c in range(self.width - length + 1):
                if self._is_valid_placement(r, c, length, 0, occupied):
                    if self._satisfies_constraints(r, c, length, 0, is_sunk):
                        candidates.append((r, c, 0))
        
        # Vertical
        for c in range(self.width):
            for r in range(self.height - length + 1):
                if self._is_valid_placement(r, c, length, 1, occupied):
                    if self._satisfies_constraints(r, c, length, 1, is_sunk):
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
        # Target Mode: Adjacent to hits
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

    def act(self, obs: np.ndarray, info: dict | None = None) -> int:
        self._update_from_obs(obs, info)
        
        # Mask: 0 for Hit/Miss cells (already fired), 1 for Unknown
        mask = np.logical_not(np.logical_or(self.hit_grid, self.miss_grid))
        
        prob_map = self._compute_prob_map()
        
        # Mask prob_map (don't fire at known cells)
        prob_map = prob_map * mask
        
        if prob_map.max() > 0:
            # Greedy max probability
            best = np.argwhere(prob_map == prob_map.max())
            rr, cc = best[self.rng.integers(0, len(best))]
            return int(rr * self.width + cc)
            
        return self._fallback_action(mask)
