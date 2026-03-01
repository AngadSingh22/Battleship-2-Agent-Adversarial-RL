"""
DefenderEnv — Gymnasium environment for training a defender placement policy.

One episode = one placement decision + one attacker rollout evaluation.

Action space: Discrete(pool_size)
  The defender picks an index into a precomputed pool of legal ship layouts.

Observation: rolling statistics over the last `history_len` attacker episodes
  (mean shots-to-win, last shot count, min shots-to-win, max shots-to-win,
   current generation number normalised to [0,1]).

Reward: mean shots-to-win over K attacker episodes (defender wants to maximise).

Termination: always after one step (episode length = 1).

Design rule: attacker evaluation is CPU-batched in the MAIN LEARNER PROCESS.
Never spawn GPU-using subprocesses from inside this env.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from battleship_rl.agents.defender import UniformRandomDefender
from battleship_rl.envs.battleship_env import BattleshipEnv


# ---------------------------------------------------------------------------
# Layout pool generator
# ---------------------------------------------------------------------------

def build_layout_pool(
    pool_size: int,
    board_size: int = 10,
    ships: Optional[List[int]] = None,
    seed: int = 0,
) -> np.ndarray:
    """Pre-generate a pool of legal ship layouts.

    Returns a (pool_size, H, W) int32 array where each slice is a legal
    ship_id_grid (0 = empty, 1..n = ship IDs).
    """
    if ships is None:
        ships = [5, 4, 3, 3, 2]
    rng = np.random.default_rng(seed)
    defender = UniformRandomDefender()
    H = board_size
    layouts = np.zeros((pool_size, H, H), dtype=np.int32)
    for i in range(pool_size):
        layout = defender.sample_layout((H, H), ships, rng)
        layouts[i] = layout.astype(np.int32)
    return layouts


# ---------------------------------------------------------------------------
# CPU-only attacker evaluation (called in main learner process, not subprocess)
# ---------------------------------------------------------------------------

def evaluate_attacker_on_layout(
    layout: np.ndarray,
    attacker_policy,
    k_episodes: int = 1,
    board_size: int = 10,
    ships: Optional[List[int]] = None,
    seed: int = 0,
) -> Tuple[float, List[int]]:
    """Run K episodes of the attacker against a fixed layout.

    Returns (mean_shots, list_of_shot_counts).

    The attacker policy must implement:
        actions, _ = attacker_policy.predict(obs, action_masks=mask, deterministic=True)

    This function must ONLY be called from the main learner process.
    Do not call from SubprocVecEnv workers.
    """
    if ships is None:
        ships = [5, 4, 3, 3, 2]

    shot_counts: List[int] = []
    env = BattleshipEnv(board_size=board_size, ships=ships, debug=False)
    max_steps = board_size * board_size  # hard cap: can never take more than HW shots

    try:
        for ep in range(k_episodes):
            obs, info = env.reset(seed=seed + ep)
            # Override the layout with the defender's chosen layout
            env.ship_id_grid = layout.astype(np.int32)
            env.backend.set_board(env.ship_id_grid)
            # Re-bind views after set_board (backend memory may have moved)
            env.hits_grid = env.backend.hits
            env.miss_grid = env.backend.misses

            steps = 0
            terminated = truncated = False
            while not (terminated or truncated):
                mask = env.get_action_mask()
                # predict expects a batch dimension: (1, *obs_shape)
                action, _ = attacker_policy.predict(
                    obs[np.newaxis], action_masks=mask[np.newaxis], deterministic=True
                )
                obs, _, terminated, truncated, info = env.step(int(action[0]))
                steps += 1
                if steps >= max_steps:
                    break  # safety cap — should never happen with valid masks
            shot_counts.append(steps)
    finally:
        env.close()  # always release backend resources, even on exception

    return float(np.mean(shot_counts)), shot_counts


# ---------------------------------------------------------------------------
# DefenderEnv
# ---------------------------------------------------------------------------

class DefenderEnv(gym.Env):
    """Gymnasium env for training a defender placement policy.

    Parameters
    ----------
    layout_pool:
        (N, H, W) int32 array of pre-generated legal layouts.
    attacker_policy:
        A callable implementing .predict(obs, action_masks, deterministic)
        Compatible with MaskablePPO. Set on each IBR generation.
    k_eval_episodes:
        Number of attacker episodes to average per defender step.
    history_len:
        How many past episode results to include in observation.
    generation:
        Current IBR generation index (used as normalised obs feature).
    max_generations:
        Total number of IBR generations planned (for normalisation).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        layout_pool: np.ndarray,
        attacker_policy=None,
        k_eval_episodes: int = 2,
        history_len: int = 8,
        generation: int = 0,
        max_generations: int = 10,
        board_size: int = 10,
        ships: Optional[List[int]] = None,
        seed: int = 0,
    ) -> None:
        super().__init__()

        self.layout_pool       = layout_pool
        self.attacker_policy   = attacker_policy
        self.k_eval_episodes   = k_eval_episodes
        self.history_len       = history_len
        self.generation        = generation
        self.max_generations   = max_generations
        self.board_size        = board_size
        self.ships             = ships or [5, 4, 3, 3, 2]
        self._base_seed        = seed

        pool_size = layout_pool.shape[0]
        self.action_space = spaces.Discrete(pool_size)

        # Observation: [mean_shots/100, last_shots/100, min/100, max/100, gen/max_gen]
        # history_len past (mean_shots) values + 1 gen feature
        obs_dim = history_len + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Rolling history of mean shots-to-win
        self._history: List[float] = [0.0] * history_len
        self._episode_seed = seed

    def set_attacker(self, attacker_policy) -> None:
        """Swap the frozen attacker after each IBR generation."""
        self.attacker_policy = attacker_policy

    def _get_obs(self) -> np.ndarray:
        obs = np.array(
            [v / 100.0 for v in self._history]
            + [self.generation / max(self.max_generations, 1)],
            dtype=np.float32,
        )
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Don't reset history — it carries context across episodes (that's the design)
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        assert self.attacker_policy is not None, \
            "Set attacker_policy via env.set_attacker() before stepping DefenderEnv"

        action = int(action)
        layout = self.layout_pool[action]

        mean_shots, _ = evaluate_attacker_on_layout(
            layout=layout,
            attacker_policy=self.attacker_policy,
            k_episodes=self.k_eval_episodes,
            board_size=self.board_size,
            ships=self.ships,
            seed=self._episode_seed,
        )
        self._episode_seed += 1

        # Reward = shots-to-win (defender maximises difficulty)
        reward = float(mean_shots)

        # Shift history
        self._history.pop(0)
        self._history.append(mean_shots)

        obs = self._get_obs()

        # One step per episode
        return obs, reward, True, False, {"mean_shots": mean_shots, "layout_idx": action}

    def close(self):
        pass
