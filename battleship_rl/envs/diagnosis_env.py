"""
battleship_rl/envs/diagnosis_env.py
=====================================
Fault Diagnosis POMDP — a second POMDP benchmark alongside Battleship.

Problem formulation
--------------------
  • A hidden system has exactly one faulty component chosen uniformly at random
    from N_COMPONENTS possible components.
  • The attacker runs sequential diagnostic tests. Each test targets one subset
    of components (a "test channel").
  • Each test returns a NOISY binary result:
      result = 1  (fault detected)  if the faulty component is in the channel AND rng > fp_rate
      result = 0  (fault absent)    otherwise (or with miss_rate probability even if present)
  • The attacker's goal is to identify (isolate) the single faulty component in
    as few tests as possible.
  • Termination:  agent declares a component identity via action >= N_TESTS.
                  Correct → success (terminated=True, reward=+1).
                  Wrong   → failure (terminated=True, reward=-1).
  • A step penalty of -0.05 applies on every test (not on the final declaration).

Observation space
------------------
  Box(shape=(N_TESTS,), dtype=float32)
  Each entry is one of {-1=untested, 0=negative, 1=positive}.

Action space
-------------
  Discrete(N_TESTS + N_COMPONENTS)
  Actions 0..N_TESTS-1     : run test i
  Actions N_TESTS..end     : declare component (action - N_TESTS) as faulty

Default config (easily overridden via constructor kwargs):
  n_components = 8
  n_tests      = 6        (6 binary tests can distinguish up to 2^6=64 components)
  fp_rate      = 0.05     (false positive: test fires when component not in channel)
  fn_rate      = 0.10     (false negative: test misses when component IS in channel)
  max_steps    = 20
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# Default test-channel matrix (N_TESTS x N_COMPONENTS), binary.
# Each row i is the set of components covered by test i.
# With 6 tests and 8 components this is a standard binary Gray-code-like matrix.
_DEFAULT_CHANNELS = np.array(
    [
        [1, 1, 1, 1, 0, 0, 0, 0],  # test 0 covers components 0-3
        [1, 1, 0, 0, 1, 1, 0, 0],  # test 1 covers 0,1,4,5
        [1, 0, 1, 0, 1, 0, 1, 0],  # test 2 covers 0,2,4,6
        [0, 1, 0, 1, 0, 1, 0, 1],  # test 3 covers 1,3,5,7
        [1, 1, 1, 1, 1, 1, 0, 0],  # test 4 covers 0-5
        [0, 0, 1, 1, 1, 1, 1, 1],  # test 5 covers 2-7
    ],
    dtype=np.float32,
)


class DiagnosisEnv(gym.Env):
    """Noisy sequential fault-diagnosis POMDP.

    Parameters
    ----------
    n_components : int
    n_tests      : int
    channels     : np.ndarray, shape (n_tests, n_components) — test coverage matrix
    fp_rate      : float — false positive rate per test
    fn_rate      : float — false negative (miss) rate per test
    step_penalty : float — reward per diagnostic test (should be negative)
    max_steps    : int   — maximum number of steps before truncation
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        n_components: int = 8,
        n_tests: int = 6,
        channels: Optional[np.ndarray] = None,
        fp_rate: float = 0.05,
        fn_rate: float = 0.10,
        step_penalty: float = -0.05,
        max_steps: int = 20,
    ) -> None:
        super().__init__()
        self.n_components = n_components
        self.n_tests = n_tests
        self.channels = (
            channels if channels is not None else _DEFAULT_CHANNELS[:n_tests, :n_components]
        )
        assert self.channels.shape == (
            n_tests,
            n_components,
        ), f"channels must be ({n_tests}, {n_components}), got {self.channels.shape}"
        self.fp_rate = fp_rate
        self.fn_rate = fn_rate
        self.step_penalty = step_penalty
        self.max_steps = max_steps

        # action: 0..n_tests-1 → run test; n_tests..n_tests+n_components-1 → declare component
        self.action_space = spaces.Discrete(n_tests + n_components)
        # observation: test results, one slot per test: -1=untested, 0=negative, 1=positive
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_tests,), dtype=np.float32
        )

        # Runtime state (initialised in reset)
        self._faulty: int = 0
        self._obs: np.ndarray = np.full(n_tests, -1.0, dtype=np.float32)
        self._steps: int = 0

    # ------------------------------------------------------------------
    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._faulty = int(self.np_random.integers(0, self.n_components))
        self._obs = np.full(self.n_tests, -1.0, dtype=np.float32)
        self._steps = 0
        return self._obs.copy(), {"faulty_component": self._faulty}

    # ------------------------------------------------------------------
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = int(action)

        # ---- Declaration action ----------------------------------------
        if action >= self.n_tests:
            declared = action - self.n_tests
            correct = declared == self._faulty
            reward = 1.0 if correct else -1.0
            info = {
                "outcome": "correct" if correct else "wrong",
                "declared": declared,
                "faulty": self._faulty,
            }
            return self._obs.copy(), reward, True, False, info

        # ---- Diagnostic test action ------------------------------------
        test_idx = action
        self._steps += 1

        # Determine noisy result
        component_in_channel = bool(self.channels[test_idx, self._faulty] > 0.5)
        if component_in_channel:
            # True signal but may miss (false negative)
            result = 0 if self.np_random.random() < self.fn_rate else 1
        else:
            # No signal but may false-alarm (false positive)
            result = 1 if self.np_random.random() < self.fp_rate else 0

        self._obs[test_idx] = float(result)

        truncated = self._steps >= self.max_steps
        info = {
            "test_idx": test_idx,
            "result": result,
            "true_in_channel": component_in_channel,
        }
        return self._obs.copy(), self.step_penalty, False, truncated, info

    # ------------------------------------------------------------------
    def render(self) -> str:
        lines = [f"Faulty component: ??? (hidden)   Steps: {self._steps}/{self.max_steps}"]
        for i, v in enumerate(self._obs):
            status = {-1.0: "untested", 0.0: "negative", 1.0: "positive"}.get(v, "?")
            lines.append(f"  Test {i}: {status}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def get_action_mask(self) -> np.ndarray:
        """Boolean mask over actions.
        - Tests already run are masked out (can't repeat).
        - Declaration actions are always valid.
        """
        mask = np.ones(self.n_tests + self.n_components, dtype=bool)
        for i in range(self.n_tests):
            if self._obs[i] != -1.0:
                mask[i] = False  # already ran this test
        return mask
