"""
test_regime_c_switching.py — verify that IBR weight switching via env_method
changes the active defender distribution on subsequent resets.
"""
import numpy as np
import pytest
from battleship_rl.agents.defender import SpreadDefender, UniformRandomDefender
from battleship_rl.envs.battleship_env import BattleshipEnv


def test_set_defender_weights_changes_distribution():
    """set_defender_weights + reset must change the sampled defender."""
    env = BattleshipEnv(
        defenders=[UniformRandomDefender(), SpreadDefender()],
        defender_weights=[1.0, 0.0],
    )

    # With weights [1,0], all resets should choose UniformRandomDefender
    names_before = set()
    for seed in range(20):
        obs, info = env.reset(seed=seed)
        names_before.add(info["defender_mode"])
    assert names_before == {"UniformRandomDefender"}, \
        f"Expected only UniformRandomDefender, got {names_before}"

    # Switch to [0,1] → all resets should choose SpreadDefender
    env.set_defender_weights([0.0, 1.0])
    names_after = set()
    for seed in range(20):
        obs, info = env.reset(seed=seed)
        names_after.add(info["defender_mode"])
    assert names_after == {"SpreadDefender"}, \
        f"Expected only SpreadDefender, got {names_after}"

    env.close()


def test_defender_mode_in_info_on_step():
    """info['defender_mode'] must be present and consistent on every step."""
    env = BattleshipEnv(
        defenders=[UniformRandomDefender(), SpreadDefender()],
        defender_weights=[0.5, 0.5],
    )
    obs, info = env.reset(seed=42)
    episode_mode = info["defender_mode"]

    mask = env.get_action_mask()
    valid = np.where(mask)[0]
    obs, r, term, trunc, info2 = env.step(int(valid[0]))
    assert "defender_mode" in info2
    assert info2["defender_mode"] == episode_mode, \
        "defender_mode must remain constant within an episode"
    env.close()
