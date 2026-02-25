from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from battleship_rl.agents.defender import (
    AdversarialDefender,
    BiasedDefender,
    UniformRandomDefender,
)
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
    adversarial_defender_path: str | None = None,
    env_config: dict | None = None,
    n_episodes: int = 100,
    seed: int = 0,
    capture_replays: int = 0,
) -> dict:
    """Evaluate a policy across multiple placement distributions.

    Gap 8: Three distributions are now evaluated as mandated by the solution
    formulation §5:
        (i)  uniform  — UniformRandomDefender
        (ii) biased   — BiasedDefender (edge-biased scripted heuristic)
        (iii) adversarial — AdversarialDefender (learned RL defender)
                           only when adversarial_defender_path is provided.

    Gap 11: The primary generalisation gap is computed as
        Δ_gen = E[τ]_adversarial − E[τ]_uniform   (Eq. 7 of solution doc)
    with a fallback to biased − uniform when no adversarial model is provided.
    """
    env_config = env_config or {}
    results: Dict[str, dict] = {}
    replays: List[dict] = []

    # ---- Defender distributions to evaluate --------------------------------
    defenders: list[tuple[str, object]] = [
        ("uniform", UniformRandomDefender()),
        ("biased", BiasedDefender()),
    ]
    if adversarial_defender_path:
        defenders.append(
            ("adversarial", AdversarialDefender(model_path=adversarial_defender_path))
        )

    for defender_name, defender in defenders:
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

    # ---- Generalisation gap ------------------------------------------------
    # Use adversarial distribution as the challenge if available; else biased.
    challenge_key = "adversarial" if "adversarial" in results else "biased"
    results["gap"] = generalization_gap(
        mean_challenge=results[challenge_key]["mean"],
        mean_uniform=results["uniform"]["mean"],
    )
    results["gap_source"] = challenge_key  # document which was used

    if replays:
        results["replays"] = replays

    return results


def _format_table(results: dict) -> str:
    rows = ["Mode | Mean | Std | 90th% | Fail Rate", "--- | --- | --- | --- | ---"]
    for key in ("uniform", "biased", "adversarial"):
        if key not in results:
            continue
        d = results[key]
        rows.append(
            f"{key.capitalize()} | {d['mean']:.2f} | {d['std']:.2f} | "
            f"{d['p90']:.2f} | {d['fail_rate']:.2f}"
        )
    gap_src = results.get("gap_source", "biased")
    rows.append(f"\nΔ_gen ({gap_src} − uniform) = {results.get('gap', float('nan')):.2f}")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Battleship policies.")
    parser.add_argument("--policy", choices=["random", "heuristic", "sb3"], default="random")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--adversarial-defender", default=None,
                        help="Path to trained AdversarialDefender model (.zip)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = evaluate_policy(
        policy_type=args.policy,
        model_path=args.model_path,
        adversarial_defender_path=args.adversarial_defender,
        n_episodes=args.episodes,
        seed=args.seed,
        capture_replays=0,
    )
    print(_format_table(results))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
