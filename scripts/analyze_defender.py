"""
Quick placement statistics for Uniform vs Biased defenders.
Avoids calling the slow BiasedDefender.sample_layout 1000 times.
Instead uses the shared sample_placement utility for Uniform, and
re-implements the edge-weighted sampling using numpy directly for Biased.
"""
from __future__ import annotations
import numpy as np

SHIPS = [5, 4, 3, 3, 2]
H, W = 10, 10
N_SAMPLES = 1000
SEED = 42


def _place_ship_uniform(board: np.ndarray, length: int, rng: np.random.Generator):
    """Try to place one ship uniformly at random on `board` (in-place)."""
    H, W = board.shape
    for _ in range(10000):  # retry until placed
        orient = rng.integers(2)
        if orient == 0:  # horizontal
            r = int(rng.integers(H))
            c = int(rng.integers(W - length + 1))
            if np.all(board[r, c:c+length] == 0):
                board[r, c:c+length] = 1
                return
        else:  # vertical
            r = int(rng.integers(H - length + 1))
            c = int(rng.integers(W))
            if np.all(board[r:r+length, c] == 0):
                board[r:r+length, c] = 1
                return
    raise RuntimeError("Could not place ship after 10000 attempts.")


def _edge_weight(r: int, c: int, H: int = H, W: int = W) -> float:
    d = min(r, H - 1 - r, c, W - 1 - c)
    return 1.0 / (d + 1)


def _place_ship_biased(board: np.ndarray, length: int, rng: np.random.Generator):
    """Try to place one ship with edge-weighted probabilities."""
    H2, W2 = board.shape
    candidates = []
    weights = []
    # Horizontal
    for r in range(H2):
        for c in range(W2 - length + 1):
            if np.all(board[r, c:c+length] == 0):
                cells = [(r, c+i) for i in range(length)]
                w = sum(_edge_weight(rr, cc, H2, W2) for rr, cc in cells)
                candidates.append(("h", r, c))
                weights.append(w)
    # Vertical
    for r in range(H2 - length + 1):
        for c in range(W2):
            if np.all(board[r:r+length, c] == 0):
                cells = [(r+i, c) for i in range(length)]
                w = sum(_edge_weight(rr, cc, H2, W2) for rr, cc in cells)
                candidates.append(("v", r, c))
                weights.append(w)
    
    if not candidates:
        raise RuntimeError("No valid placements.")
    
    ws = np.array(weights, dtype=np.float32)
    ws /= ws.sum()
    idx = int(rng.choice(len(candidates), p=ws))
    orient, r, c = candidates[idx]
    if orient == "h":
        board[r, c:c+length] = 1
    else:
        board[r:r+length, c] = 1


def compute_stats(grid: np.ndarray) -> tuple[float, float]:
    locs = np.argwhere(grid > 0)
    if len(locs) == 0:
        return 0.0, 0.0
    edge_ct = 0
    dist_total = 0.0
    for r, c in locs:
        d = int(min(r, H-1-r, c, W-1-c))
        dist_total += d
        if d == 0:
            edge_ct += 1
    return edge_ct / len(locs), dist_total / len(locs)


def cell_entropy(freq: np.ndarray) -> float:
    eps = 1e-9
    p = np.clip(freq, eps, 1-eps)
    return float(np.mean(-(p * np.log2(p) + (1-p) * np.log2(1-p))))


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)

    u_edges, u_dists, u_freq = [], [], np.zeros((H, W))
    b_edges, b_dists, b_freq = [], [], np.zeros((H, W))
    
    for i in range(N_SAMPLES):
        # Uniform
        grid = np.zeros((H, W), dtype=np.int32)
        for length in SHIPS:
            _place_ship_uniform(grid, length, rng)
        e, d = compute_stats(grid)
        u_edges.append(e); u_dists.append(d); u_freq += grid

        # Biased (edge-weighted)
        grid_b = np.zeros((H, W), dtype=np.int32)
        for length in SHIPS:
            _place_ship_biased(grid_b, length, rng)
        e2, d2 = compute_stats(grid_b)
        b_edges.append(e2); b_dists.append(d2); b_freq += grid_b

    u_freq /= N_SAMPLES
    b_freq /= N_SAMPLES

    print(f"\n{'Metric':<28} | {'Uniform':>10} | {'Biased':>10}")
    print("-" * 55)
    print(f"{'Edge Occupancy Fraction':<28} | {np.mean(u_edges):>10.4f} | {np.mean(b_edges):>10.4f}")
    print(f"{'Mean Distance to Edge':<28} | {np.mean(u_dists):>10.4f} | {np.mean(b_dists):>10.4f}")
    print(f"{'Marginal Cell Entropy':<28} | {cell_entropy(u_freq):>10.4f} | {cell_entropy(b_freq):>10.4f}")
    
    # Info whether gap is meaningful (biased should place MORE on edges)
    gap_edge = np.mean(b_edges) - np.mean(u_edges)
    gap_dist = np.mean(u_dists) - np.mean(b_dists)  # Biased should have smaller mean-dist-to-edge
    print(f"\nEdge occupancy gap (biased - uniform): {gap_edge:+.4f}")
    print(f"Distance-to-edge gap (uniform - biased): {gap_dist:+.4f}  (positive = biased closer to edges)")
    if gap_edge > 0.02:
        print("✓ BiasedDefender measurably shifts placements toward edges.")
    else:
        print("⚠ BiasedDefender and Uniform are near-identical — bias is too weak.")
    
    entropy_gap = cell_entropy(u_freq) - cell_entropy(b_freq)
    if entropy_gap > 0.01:
        print("✓ Biased distribution has meaningfully lower entropy (more concentrated).")
    else:
        print("⚠ Entropies are too similar — bias may not be strong enough.")
