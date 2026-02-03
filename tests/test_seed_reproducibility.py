import numpy as np

from battleship_rl.envs.battleship_env import BattleshipEnv


def test_seed_reproducibility():
    env1 = BattleshipEnv(board_size=5, ships=[2, 3])
    env2 = BattleshipEnv(board_size=5, ships=[2, 3])

    env1.reset(seed=123)
    env2.reset(seed=123)

    assert np.array_equal(env1.ship_id_grid, env2.ship_id_grid)

    for action in range(env1.action_space.n):
        _, _, term1, trunc1, info1 = env1.step(action)
        _, _, term2, trunc2, info2 = env2.step(action)
        assert info1["outcome_type"] == info2["outcome_type"]
        assert info1["outcome_ship_id"] == info2["outcome_ship_id"]
        assert term1 == term2
        assert trunc1 == trunc2
        if term1 or trunc1:
            break
