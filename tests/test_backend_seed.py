"""
test_backend_seed.py — Fixed seed produces identical placement and outcome stream.
Enforces the "single RNG per episode" invariant: backend seed is derived from
self.np_random (not the raw seed arg), so identical episode seeds are reproducible.
"""
import numpy as np
import pytest
from battleship_rl.envs.battleship_env import BattleshipEnv


def test_same_seed_same_placement():
    """Identical seed → identical ship_id_grid."""
    env1 = BattleshipEnv()
    env2 = BattleshipEnv()
    for seed in [0, 1, 42, 9999]:
        env1.reset(seed=seed)
        env2.reset(seed=seed)
        np.testing.assert_array_equal(
            env1.ship_id_grid, env2.ship_id_grid,
            err_msg=f"Placement mismatch at seed={seed}"
        )
    env1.close()
    env2.close()


def test_same_seed_same_outcome_stream():
    """Identical seed + identical action sequence → identical outcome stream."""
    seed = 77
    actions = [5, 12, 45, 99, 0, 77, 33, 50]

    env1 = BattleshipEnv()
    env2 = BattleshipEnv()
    env1.reset(seed=seed)
    env2.reset(seed=seed)

    for a in actions:
        _, r1, t1, u1, i1 = env1.step(a)
        _, r2, t2, u2, i2 = env2.step(a)
        assert r1 == r2,                      f"reward mismatch at action={a}"
        assert t1 == t2,                      f"terminated mismatch at action={a}"
        assert i1["outcome_type"] == i2["outcome_type"]
    env1.close()
    env2.close()


def test_different_seeds_different_placements():
    """Different seeds should (almost certainly) produce different placements."""
    env = BattleshipEnv()
    placements = []
    for seed in range(10):
        env.reset(seed=seed)
        placements.append(env.ship_id_grid.copy())
    # At least some placements should differ
    unique = len({arr.tobytes() for arr in placements})
    assert unique > 1, "All seeds produced identical placements (seeding is broken)"
    env.close()
