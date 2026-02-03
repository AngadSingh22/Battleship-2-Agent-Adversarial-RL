import numpy as np

from battleship_rl.envs.battleship_env import BattleshipEnv


class FixedDefender:
    def __init__(self, layout: np.ndarray) -> None:
        self.layout = layout.astype(np.int32)

    def sample_layout(self, board_size, ships, rng):
        return self.layout.copy()


def test_hit_and_sunk_flow():
    layout = np.array([[0, 0], [-1, -1]], dtype=np.int32)
    env = BattleshipEnv(
        board_size=(2, 2),
        ships=[2],
        defender=FixedDefender(layout),
        debug=True,
    )
    env.reset(seed=123)

    _, _, terminated, truncated, info = env.step(0)
    assert info["outcome_type"] == "HIT"
    assert info["outcome_ship_id"] == 0
    assert info["last_outcome"] == ("HIT", 0)
    assert not terminated
    assert not truncated

    _, _, terminated, truncated, info = env.step(1)
    assert info["outcome_type"] == "SUNK"
    assert info["outcome_ship_id"] == 0
    assert info["last_outcome"] == ("SUNK", 0)
    assert terminated
    assert not truncated


def test_invalid_action_truncates_without_state_change():
    layout = np.array([[0, 0], [-1, -1]], dtype=np.int32)
    env = BattleshipEnv(
        board_size=(2, 2),
        ships=[2],
        defender=FixedDefender(layout),
        debug=False,
    )
    env.reset(seed=123)

    env.step(0)
    hits_before = env.hits_grid.copy()
    misses_before = env.miss_grid.copy()

    _, reward, terminated, truncated, info = env.step(0)
    assert info["outcome_type"] == "INVALID"
    assert reward == -100.0
    assert truncated
    assert not terminated
    assert np.array_equal(hits_before, env.hits_grid)
    assert np.array_equal(misses_before, env.miss_grid)
