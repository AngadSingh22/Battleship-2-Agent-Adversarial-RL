from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from battleship_rl.agents.defender import BiasedDefender, UniformRandomDefender
from battleship_rl.baselines.heuristic_probmap import HeuristicProbMapAgent
from battleship_rl.baselines.random_agent import RandomAgent
from battleship_rl.envs.battleship_env import BattleshipEnv
from battleship_rl.eval.metrics import generalization_gap, summarize


class PolicyAdapter:
    def reset(self) -> None:
        return None

    def act(self, obs: np.ndarray, info: dict, env: BattleshipEnv) -> int:
        raise NotImplementedError


class RandomPolicyAdapter(PolicyAdapter):
    def __init__(self, rng: np.random.Generator) -> None:
        self.agent = RandomAgent(rng=rng)

    def reset(self) -> None:
        self.agent.reset()

    def act(self, obs: np.ndarray, info: dict, env: BattleshipEnv) -> int:
        return self.agent.act(obs, info)


class HeuristicPolicyAdapter(PolicyAdapter):
    def __init__(self, env: BattleshipEnv, rng: np.random.Generator) -> None:
        self.agent = HeuristicProbMapAgent(
            board_size=(env.height, env.width),
            ships=env.ship_lengths,
            rng=rng,
        )

    def reset(self) -> None:
        self.agent.reset()

    def act(self, obs: np.ndarray, info: dict, env: BattleshipEnv) -> int:
        return self.agent.act(obs, info)


class SB3PolicyAdapter(PolicyAdapter):
    def __init__(self, model_path: str, deterministic: bool = True) -> None:
        try:
            from sb3_contrib import MaskablePPO
        except ImportError as exc:
            raise ImportError("sb3-contrib is required for SB3 policy evaluation.") from exc
        self.model = MaskablePPO.load(model_path)
        self.deterministic = deterministic

    def act(self, obs: np.ndarray, info: dict, env: BattleshipEnv) -> int:
        action, _ = self.model.predict(
            obs, action_masks=env.get_action_mask(), deterministic=self.deterministic
        )
        return int(action)


def _run_episode(
    env: BattleshipEnv,
    policy: PolicyAdapter,
    seed: Optional[int] = None,
    record_steps: bool = False,
) -> tuple[int, bool, list]:
    obs, info = env.reset(seed=seed)
    policy.reset()
    steps: List[dict] = []
    t = 0
    while True:
        action = policy.act(obs, info, env)
        obs, _, terminated, truncated, info = env.step(action)
        if record_steps:
            steps.append(
                {
                    "r": int(action // env.width),
                    "c": int(action % env.width),
                    "type": info.get("outcome_type"),
                }
            )
        t += 1
        if terminated or truncated:
            break
    return t, truncated, steps


def _make_policy(policy_type: str, env: BattleshipEnv, rng: np.random.Generator, model_path: str | None):
    if policy_type == "random":
        return RandomPolicyAdapter(rng)
    if policy_type == "heuristic":
        return HeuristicPolicyAdapter(env, rng)
    if policy_type == "sb3":
        if model_path is None:
            raise ValueError("model_path must be provided for sb3 policy.")
        return SB3PolicyAdapter(model_path=model_path)
    raise ValueError(f"Unknown policy_type: {policy_type}")


def evaluate_policy(
    policy_type: str,
    model_path: str | None = None,
    env_config: dict | None = None,
    n_episodes: int = 100,
    seed: int = 0,
    capture_replays: int = 0,
) -> dict:
    env_config = env_config or {}
    results: Dict[str, dict] = {}

    replays: List[dict] = []

    for defender_name, defender in [
        ("uniform", UniformRandomDefender()),
        ("biased", BiasedDefender()),
    ]:
        env = BattleshipEnv(config=env_config, defender=defender)
        rng = np.random.default_rng(seed)
        policy = _make_policy(policy_type, env, rng, model_path)
        lengths: List[int] = []
        truncated_flags: List[bool] = []
        for idx in range(n_episodes):
            record_steps = defender_name == "uniform" and idx < capture_replays
            episode_seed = seed + idx
            length, truncated, steps = _run_episode(
                env,
                policy,
                seed=episode_seed,
                record_steps=record_steps,
            )
            lengths.append(length)
            truncated_flags.append(truncated)
            if record_steps:
                replays.append({"seed": episode_seed, "steps": steps})
        results[defender_name] = summarize(lengths, truncated_flags)

    gap = generalization_gap(results["biased"]["mean"], results["uniform"]["mean"])
    results["gap"] = gap
    if replays:
        results["replays"] = replays

    return results


def _format_table(uniform: dict, biased: dict, gap: float) -> str:
    header = "Mode | Mean | Std | 90th% | Fail Rate | Gap"
    line = "--- | --- | --- | --- | --- | ---"
    rows = [
        "Uniform | {mean:.2f} | {std:.2f} | {p90:.2f} | {fail:.2f} | {gap:.2f}".format(
            mean=uniform["mean"],
            std=uniform["std"],
            p90=uniform["p90"],
            fail=uniform["fail_rate"],
            gap=0.0,
        ),
        "Biased | {mean:.2f} | {std:.2f} | {p90:.2f} | {fail:.2f} | {gap:.2f}".format(
            mean=biased["mean"],
            std=biased["std"],
            p90=biased["p90"],
            fail=biased["fail_rate"],
            gap=gap,
        ),
    ]
    return "\n".join([header, line, *rows])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Battleship policies.")
    parser.add_argument("--policy", choices=["random", "heuristic", "sb3"], default="random")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = evaluate_policy(
        policy_type=args.policy,
        model_path=args.model_path,
        n_episodes=args.episodes,
        seed=args.seed,
        capture_replays=0,
    )
    table = _format_table(results["uniform"], results["biased"], results["gap"])
    print(table)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
