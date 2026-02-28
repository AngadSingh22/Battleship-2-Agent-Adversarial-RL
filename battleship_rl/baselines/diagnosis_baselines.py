"""
battleship_rl/baselines/diagnosis_baselines.py
================================================
Baselines for the DiagnosisEnv POMDP.

RandomTester
  Picks a random untested test at each step. Declares randomly if all tests run.

GreedySplitTester
  Greedy information-theoretic policy:
    - Maintain belief set B = set of components not yet ruled out by observations
    - At each step, pick the untested test t that maximises:
        score(t) = min( |{c∈B : channels[t,c]=1}| , |{c∈B : channels[t,c]=0}| )
      (this is the balanced-split criterion, equivalent to minimising worst-case
       remaining candidates under a deterministic oracle, and an approximation to
       greedy entropy minimisation for noisy channels)
    - After all tests run, declare the unique surviving component (if multiple
      survive due to noise, pick the one most consistent with observations)
    - Noisy channel handling: instead of hard rule-out, maintain soft 'posterior'
      weights updated by likelihood P(obs | comp in channel, noise rates)
"""
from __future__ import annotations

from typing import Optional
import numpy as np


class RandomTester:
    """Picks a random untested test each step.
    Declares the argmax-belief component after exhausting all tests
    (using a uniform prior — i.e., random among survivors).
    """

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self.rng = rng or np.random.default_rng()
        self._belief: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._belief = None

    def act(self, obs: np.ndarray, info: dict, env) -> int:
        n_tests = env.n_tests
        n_comp = env.n_components
        untested = [i for i in range(n_tests) if obs[i] == -1.0]
        # Valid actions: any untested test, or any declaration
        valid_actions = untested + list(range(n_tests, n_tests + n_comp))
        return int(self.rng.choice(valid_actions))


class GreedySplitTester:
    """Greedy minimum-worst-case split tester with soft Bayesian belief update.

    Belief b[c] = unnormalized posterior P(fault=c | observations so far).
    Updated at each test using the noisy channel likelihoods.

    Declares early when posterior is sufficiently concentrated
    (max_belief > confidence_threshold), producing variable episode lengths.
    """

    def __init__(
        self,
        rng: Optional[np.random.Generator] = None,
        confidence_threshold: float = 0.92,
    ) -> None:
        self.rng = rng or np.random.default_rng()
        self.confidence_threshold = confidence_threshold
        self._belief: Optional[np.ndarray] = None
        self._n_components: int = 0
        self._n_tests: int = 0
        self._channels: Optional[np.ndarray] = None
        self._fp_rate: float = 0.05
        self._fn_rate: float = 0.10

    def reset(self) -> None:
        self._belief = None

    def _init_from_env(self, env) -> None:
        self._n_components = env.n_components
        self._n_tests = env.n_tests
        self._channels = env.channels  # (n_tests, n_components)
        self._fp_rate = env.fp_rate
        self._fn_rate = env.fn_rate
        self._belief = np.ones(self._n_components, dtype=np.float64)  # uniform prior

    def _update_belief(self, test_idx: int, result: int) -> None:
        """Bayesian update of belief given noisy test result."""
        channels = self._channels
        fp, fn = self._fp_rate, self._fn_rate
        # P(result=1 | fault=c, test=t)
        #   = (1-fn) if c in channel, fp otherwise
        # P(result=0 | fault=c, test=t)
        #   = fn      if c in channel, (1-fp) otherwise
        in_channel = channels[test_idx].astype(np.float64)  # 1.0 or 0.0 per component
        if result == 1:
            likelihood = in_channel * (1 - fn) + (1 - in_channel) * fp
        else:
            likelihood = in_channel * fn + (1 - in_channel) * (1 - fp)
        self._belief *= likelihood
        # Renormalize to prevent underflow
        total = self._belief.sum()
        if total > 0:
            self._belief /= total
        else:
            # All hypotheses collapsed (shouldn't happen with nonzero noise)
            self._belief[:] = 1.0 / self._n_components

    def _choose_best_test(self, tested: set[int]) -> int:
        """Return the untested test with the highest balanced-split score on soft belief."""
        channels = self._channels
        belief = self._belief
        best_score = -1.0
        best_t = -1

        for t in range(self._n_tests):
            if t in tested:
                continue
            # Expected mass in channel vs out of channel
            in_mass = float(np.dot(channels[t], belief))
            out_mass = 1.0 - in_mass  # belief is normalized
            score = min(in_mass, out_mass)  # balanced-split score
            if score > best_score:
                best_score = score
                best_t = t

        return best_t if best_t >= 0 else next(t for t in range(self._n_tests) if t not in tested)

    def act(self, obs: np.ndarray, info: dict, env) -> int:
        if self._belief is None:
            self._init_from_env(env)

        n_tests = env.n_tests
        tested = {i for i in range(n_tests) if obs[i] != -1.0}

        # Rebuild belief from all observations (correct and stable under noise)
        self._belief = np.ones(self._n_components, dtype=np.float64)
        for i in tested:
            self._update_belief(i, int(obs[i]))

        # Early declaration if confidence exceeds threshold (only after >= 1 test)
        if len(tested) > 0 and self._belief.max() >= self.confidence_threshold:
            best_comp = int(np.argmax(self._belief))
            return n_tests + best_comp

        if len(tested) < n_tests:
            return self._choose_best_test(tested)

        # All tests exhausted: declare argmax of belief
        best_comp = int(np.argmax(self._belief))
        return n_tests + best_comp
