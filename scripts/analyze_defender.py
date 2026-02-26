import numpy as np
from battleship_rl.agents.defender import UniformRandomDefender, BiasedDefender

def compute_metrics(grid):
    # grid is 10x10. Ship locations are > 0.
    locs = np.argwhere(grid > 0)
    if len(locs) == 0:
        return 0.0, 0.0
        
    edge_count = 0
    total_dist = 0.0
    for r, c in locs:
        # Edge distance: min(r, 9-r, c, 9-c)
        dist = min(r, 9 - r, c, 9 - c)
        total_dist += dist
        if dist == 0:
            edge_count += 1
            
    edge_occupancy = edge_count / len(locs)
    mean_dist = total_dist / len(locs)
    return edge_occupancy, mean_dist

def analyze():
    uniform = UniformRandomDefender()
    biased = BiasedDefender() # assuming defaults, edge-biased typically
    
    n_samples = 1000
    rng = np.random.default_rng(42)
    board_shape = (10, 10)
    ships = [5, 4, 3, 3, 2]
    
    u_edges = []
    u_dists = []
    u_freq = np.zeros(board_shape)
    
    b_edges = []
    b_dists = []
    b_freq = np.zeros(board_shape)
    
    for _ in range(n_samples):
        # Uniform
        ug = uniform.sample_layout(board_shape, ships, rng)
        ue, ud = compute_metrics(ug)
        u_edges.append(ue)
        u_dists.append(ud)
        u_freq += (ug > 0)
        
        # Biased
        bg = biased.sample_layout(board_shape, ships, rng)
        be, bd = compute_metrics(bg)
        b_edges.append(be)
        b_dists.append(bd)
        b_freq += (bg > 0)
        
    u_freq /= n_samples
    b_freq /= n_samples
    
    # Simple placement entropy proxy over the 10x10 board
    # Entropy of the marginal probability of each cell being occupied
    # H = - \sum (p log p + (1-p) log(1-p))
    def cell_entropy(prob_grid):
        eps = 1e-9
        p = np.clip(prob_grid, eps, 1-eps)
        ent = - (p * np.log2(p) + (1-p) * np.log2(1-p))
        return np.mean(ent)
        
    u_ent = cell_entropy(u_freq)
    b_ent = cell_entropy(b_freq)
    
    print(f"{'Metric':<25} | {'Uniform':<10} | {'Biased':<10}")
    print("-" * 50)
    print(f"{'Edge Occupancy Fraction':<25} | {np.mean(u_edges):.4f}     | {np.mean(b_edges):.4f}")
    print(f"{'Mean Distance to Edge':<25} | {np.mean(u_dists):.4f}     | {np.mean(b_dists):.4f}")
    print(f"{'Marginal Cell Entropy':<25} | {u_ent:.4f}     | {b_ent:.4f}")

if __name__ == "__main__":
    analyze()
