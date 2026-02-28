"""Smoke test for DiagnosisEnv — no external deps needed."""
import numpy as np
import pytest
from battleship_rl.envs.diagnosis_env import DiagnosisEnv


def test_reset_obs_shape():
    env = DiagnosisEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (env.n_tests,), f"Bad obs shape: {obs.shape}"
    assert np.all(obs == -1.0), "Initial obs should be all -1 (untested)"
    assert "faulty_component" in info
    assert 0 <= info["faulty_component"] < env.n_components


def test_run_test_action():
    env = DiagnosisEnv()
    env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(0)  # run test 0
    assert not terminated
    assert not truncated
    assert obs[0] in (0.0, 1.0), "Tested slot must be 0 or 1"
    assert reward == env.step_penalty
    assert "test_idx" in info


def test_declaration_correct():
    env = DiagnosisEnv(fp_rate=0.0, fn_rate=0.0)  # noiseless
    obs, info = env.reset(seed=7)
    faulty = info["faulty_component"]
    # Run all tests to identify the fault exactly (noiseless)
    for t in range(env.n_tests):
        env.step(t)
    # Declare the actual faulty component
    obs, reward, terminated, truncated, info = env.step(env.n_tests + faulty)
    assert terminated
    assert reward == 1.0, f"Expected reward 1.0 for correct declaration, got {reward}"


def test_declaration_wrong():
    env = DiagnosisEnv()
    env.reset(seed=12)
    wrong = (env._faulty + 1) % env.n_components
    obs, reward, terminated, truncated, info = env.step(env.n_tests + wrong)
    assert terminated
    assert reward == -1.0


def test_action_mask_removes_tested():
    env = DiagnosisEnv()
    env.reset(seed=1)
    mask_before = env.get_action_mask()
    assert mask_before[0], "Test 0 should be available before running it"
    env.step(0)
    mask_after = env.get_action_mask()
    assert not mask_after[0], "Test 0 should be masked after running it"


def test_max_steps_truncation():
    env = DiagnosisEnv(max_steps=3)
    env.reset(seed=0)
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            break
    assert truncated or terminated


def test_gymnasium_compliance():
    """gymnasium.utils.check_env should pass (we do a manual equivalent)."""
    env = DiagnosisEnv()
    obs, info = env.reset(seed=99)
    assert env.observation_space.contains(obs)
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            break
