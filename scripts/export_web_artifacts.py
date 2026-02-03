from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import yaml

from battleship_rl.eval.evaluate import evaluate_policy


def _load_yaml(path: str | None) -> dict:
    if path is None:
        return {}
    data = Path(path)
    if not data.exists():
        return {}
    with data.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _extract_board_meta(env_config: dict) -> tuple[int, int, list[int]]:
    board_size = env_config.get("board_size", 10)
    ship_config = env_config.get("ship_config", None)
    ships = ship_config if ship_config is not None else env_config.get("ships", [5, 4, 3, 3, 2])
    if isinstance(board_size, int):
        height, width = board_size, board_size
    else:
        height, width = int(board_size[0]), int(board_size[1])
    ship_lengths = list(ships.values()) if isinstance(ships, dict) else list(ships)
    return height, width, [int(length) for length in ship_lengths]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export web artifacts for GitHub Pages.")
    parser.add_argument("--policy", choices=["random", "heuristic", "sb3"], default="random")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--replays", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-config", default="configs/env.yaml")
    parser.add_argument("--output-dir", default="site/data")
    args = parser.parse_args()

    env_config = _load_yaml(args.env_config)
    results = evaluate_policy(
        policy_type=args.policy,
        model_path=args.model_path,
        env_config=env_config,
        n_episodes=args.episodes,
        seed=args.seed,
        capture_replays=args.replays,
    )

    height, width, ship_lengths = _extract_board_meta(env_config)
    output_dir = Path(args.output_dir)
    replays_dir = output_dir / "replays"
    replays_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {"H": height, "W": width, "ships": ship_lengths}
    metrics_payload = {
        "mean": results["uniform"]["mean"],
        "std": results["uniform"]["std"],
        "p90": results["uniform"]["p90"],
        "gap": results["gap"],
    }

    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    for idx, replay in enumerate(results.get("replays", [])):
        replay_path = replays_dir / f"replay_{idx}.json"
        with replay_path.open("w", encoding="utf-8") as handle:
            json.dump(replay, handle, indent=2)


if __name__ == "__main__":
    main()
