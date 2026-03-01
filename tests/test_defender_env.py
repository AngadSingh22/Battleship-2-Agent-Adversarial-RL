"""
test_defender_env.py — DefenderEnv must terminate in exactly one step,
and reward should be positive (shots > 0 for any attacker).
"""
import numpy as np
import pytest
from battleship_rl.envs.defender_env import DefenderEnv, build_layout_pool


class _DummyAttacker:
    """Trivial attacker: always picks the first valid action."""
    def __init__(self):
        self.observation_space = None

    def predict(self, obs, action_masks=None, deterministic=True):
        if action_masks is not None:
            mask = action_masks[0] if action_masks.ndim > 1 else action_masks
            valid = np.where(mask)[0]
            action = int(valid[0]) if len(valid) > 0 else 0
        else:
            action = 0
        return np.array([action]), None


def test_defender_env_terminates_in_one_step():
    """DefenderEnv must always terminate after exactly one step."""
    pool = build_layout_pool(pool_size=200, board_size=10, seed=0)
    env = DefenderEnv(
        layout_pool=pool,
        attacker_policy=_DummyAttacker(),
        k_eval_episodes=1,
        generation=0,
        max_generations=5,
    )
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape, "obs shape mismatch"

    obs2, reward, terminated, truncated, info = env.step(0)
    assert terminated is True,  "DefenderEnv must terminate after one step"
    assert truncated  is False, "DefenderEnv must not truncate"
    assert reward > 0,          f"Reward (shots) must be positive, got {reward}"
    env.close()


def test_defender_env_reward_equals_shots():
    """Reward should equal mean attacker shots-to-win (sanity check)."""
    pool = build_layout_pool(pool_size=100, board_size=10, seed=7)
    env = DefenderEnv(
        layout_pool=pool,
        attacker_policy=_DummyAttacker(),
        k_eval_episodes=2,
        generation=1,
        max_generations=5,
    )
    env.reset(seed=0)
    _, reward, terminated, _, info = env.step(5)
    # Reward should roughly match mean shots returned in info
    assert abs(reward - info["mean_shots"]) < 1e-6
    assert terminated is True
    env.close()


def test_defender_env_obs_shape_and_range():
    """Observation must be in [0,1] and match observation_space."""
    pool = build_layout_pool(pool_size=50, board_size=10, seed=1)
    env = DefenderEnv(
        layout_pool=pool,
        attacker_policy=_DummyAttacker(),
        k_eval_episodes=1,
        history_len=8,
    )
    obs, _ = env.reset(seed=0)
    assert obs.shape == (9,), f"Expected (9,), got {obs.shape}"   # 8 history + 1 gen
    assert obs.min() >= 0.0 and obs.max() <= 1.0, \
        f"Observation out of [0,1]: {obs}"
    env.close()
