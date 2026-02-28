#!/usr/bin/env python3
"""
scripts/evaluate_suite.py
===========================
Unified evaluation suite across all policies and defender distributions.

Runs:
  1. Random baseline (Uniform + Biased defenders)
  2. Heuristic ProbMap baseline (Uniform + Biased defenders)
  3. Particle Belief baseline (Uniform + Biased defenders)

For each, computes:
  - Core metrics: mean/std/p90 shots, fail rate, fallback rate
  - Δ_gen generalisation gap
  - Behavioural diagnostics: time-to-first-hit, hit-rate, hunt-efficiency,
    revisit-rate, shots-per-ship-sunk

Saves results to:
  results/final_evaluation.md   (human-readable markdown summary)
  results/final_evaluation.json (raw results for downstream processing)

Usage:
  PYTHONPATH=. .venv/bin/python3 scripts/evaluate_suite.py [--episodes N] [--seed S]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

from battleship_rl.eval.evaluate import evaluate_policy, _run_episode, _make_policy
from battleship_rl.eval.diagnostics import summarize_diagnostics, format_diagnostics
from battleship_rl.agents.defender import UniformRandomDefender, BiasedDefender
from battleship_rl.envs.battleship_env import BattleshipEnv


# ---------------------------------------------------------------------------
def _collect_step_data(policy_type: str, n_episodes: int, seed: int) -> list[list[dict]]:
    """Run N episodes with Uniform defender and collect step logs."""
    env = BattleshipEnv(defender=UniformRandomDefender())
    rng = np.random.default_rng(seed)
    policy = _make_policy(policy_type, env, rng, model_path=None)
    all_steps = []
    for idx in range(n_episodes):
        _, _, steps, _, _ = _run_episode(env, policy, seed=seed + idx, record_steps=True)
        policy.reset()
        all_steps.append(steps)
    return all_steps


# ---------------------------------------------------------------------------
def run_suite(n_episodes: int = 100, seed: int = 0) -> dict:
    policies = {
        "random":    "Random baseline — fires at random valid cells",
        "heuristic": "ProbMap heuristic — probability sampling with constraint backtracking",
        "particle":  "Particle Belief SMC — full particle filter over board hypotheses",
    }

    suite_results = {}
    for policy_type, description in policies.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {policy_type.upper()}  ({description})")
        print(f"{'='*60}")

        eval_result = evaluate_policy(
            policy_type=policy_type,
            n_episodes=n_episodes,
            seed=seed,
            capture_replays=0,
        )

        # Collect diagnostics from step logs (uniform defender)
        diag_episodes = _collect_step_data(policy_type, min(n_episodes, 30), seed)
        diag = summarize_diagnostics(diag_episodes)

        suite_results[policy_type] = {
            "description": description,
            "eval": eval_result,
            "diagnostics": diag,
        }
        print(f"  Uniform: {eval_result['uniform']['mean']:.1f} ± {eval_result['uniform']['std']:.1f}")
        print(f"  Biased:  {eval_result['biased']['mean']:.1f} ± {eval_result['biased']['std']:.1f}")
        print(f"  Δ_gen:   {eval_result['gap']:.2f}")
        print(f"  Hit Rate: {diag['hit_rate']:.3f}  |  Time-to-first-hit: {diag['time_to_first_hit']:.1f}")

    return suite_results


# ---------------------------------------------------------------------------
def _policy_interpretation(policy: str, results: dict) -> str:
    """Generate a concise interpretation of what the results mean for this policy."""
    e = results["eval"]
    d = results["diagnostics"]
    u_mean = e["uniform"]["mean"]
    b_mean = e["biased"]["mean"]
    gap = e["gap"]
    tth = d["time_to_first_hit"]
    hr = d["hit_rate"]

    if policy == "random":
        return (
            f"**Specific:** The random agent requires **{u_mean:.1f}** shots on Uniform and **{b_mean:.1f}** on Biased, "
            f"with Δ_gen = {gap:.2f}. Hit rate = {hr:.3f}, time-to-first-hit = {tth:.1f} shots. "
            f"This serves as the absolute lower bound; any learned or heuristic agent should do better.\n\n"
            f"**General:** Random performance establishes the baseline expectation. "
            f"The hit rate ({hr:.3f}) is consistent with the geometric probability of hitting a ship cell "
            f"on a 10×10 board with ships covering ~17 cells out of 100."
        )
    elif policy == "heuristic":
        return (
            f"**Specific:** The ProbMap heuristic requires **{u_mean:.1f}** shots on Uniform and **{b_mean:.1f}** on Biased, "
            f"with Δ_gen = {gap:.2f}. Hit rate = {hr:.3f}, time-to-first-hit = {tth:.1f} shots. "
            f"Fallback rate = {e['uniform'].get('fallback_rate', 0):.3f} (fraction of steps "
            f"where the sampler exhausted its budget).\n\n"
            f"**General:** The heuristic substantially outperforms random. The small Δ_gen indicates that the "
            f"BiasedDefender does not provide a meaningfully harder challenge for a probability-map agent. "
            f"A gap close to zero or negative would indicate the bias accidentally helps the heuristic "
            f"(e.g. placing ships on edges where probability mass concentrates naturally)."
        )
    elif policy == "particle":
        return (
            f"**Specific:** The Particle Belief SMC requires **{u_mean:.1f}** shots on Uniform and **{b_mean:.1f}** on Biased, "
            f"with Δ_gen = {gap:.2f}. Hit rate = {hr:.3f}, time-to-first-hit = {tth:.1f} shots.\n\n"
            f"**General:** The particle filter is the strongest scripted baseline. It maintains a posterior over "
            f"full board layouts and marginalizes to select the highest-probability untried cell. "
            f"A better performance vs heuristic indicates that the explicit constraint propagation "
            f"(filtering invalid particles vs. re-sampling particles each step) extracts more signal from observations."
        )
    return ""


# ---------------------------------------------------------------------------
def format_markdown(suite_results: dict, n_episodes: int, seed: int) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Final Evaluation Suite Results",
        f"",
        f"**Generated:** {now}  |  **Episodes per policy:** {n_episodes}  |  **Seed:** {seed}",
        f"",
        f"---",
        f"",
    ]

    # Summary comparison table
    lines += [
        "## Summary Table",
        "",
        "| Policy | Uniform Mean ± Std | Biased Mean ± Std | Δ_gen | Hit Rate | Time-to-First-Hit |",
        "|--------|-------------------|------------------|-------|----------|-------------------|",
    ]
    for policy, data in suite_results.items():
        e = data["eval"]
        d = data["diagnostics"]
        u = e["uniform"]
        b = e["biased"]
        lines.append(
            f"| {policy.capitalize()} | {u['mean']:.1f} ± {u['std']:.1f} | {b['mean']:.1f} ± {b['std']:.1f} | "
            f"{e['gap']:+.2f} | {d['hit_rate']:.3f} | {d['time_to_first_hit']:.1f} |"
        )

    lines += ["", "---", ""]

    # Per-policy detail sections
    for policy, data in suite_results.items():
        e = data["eval"]
        d = data["diagnostics"]
        u = e["uniform"]
        b = e["biased"]

        lines += [
            f"## Policy: {policy.capitalize()}",
            f"> {data['description']}",
            f"",
            f"### Performance Metrics",
            f"",
            f"| Mode | Mean | Std | 90th% | Fail Rate | Fallback Rate |",
            f"|------|------|-----|-------|-----------|---------------|",
            f"| Uniform | {u['mean']:.2f} | {u['std']:.2f} | {u['p90']:.2f} | {u['fail_rate']:.3f} | {u.get('fallback_rate', 0):.3f} |",
            f"| Biased  | {b['mean']:.2f} | {b['std']:.2f} | {b['p90']:.2f} | {b['fail_rate']:.3f} | {b.get('fallback_rate', 0):.3f} |",
            f"",
            f"**Δ_gen ({e.get('gap_source', 'biased')} − uniform) = {e['gap']:+.2f}**",
            f"",
            f"### Behavioural Diagnostics (Uniform Defender, first 30 episodes)",
            f"",
            format_diagnostics(d),
            f"",
            f"### Interpretation",
            f"",
            _policy_interpretation(policy, data),
            f"",
            f"---",
            f"",
        ]

    # Project-wide interpretation
    lines += [
        "## Overall Project Interpretation",
        "",
        "The results above establish a performance ladder for Battleship attackers:",
        "",
    ]
    policy_list = list(suite_results.items())
    for i, (policy, data) in enumerate(policy_list):
        rank = i + 1
        u_mean = data["eval"]["uniform"]["mean"]
        lines.append(f"{rank}. **{policy.capitalize()}** — {u_mean:.1f} shots on Uniform")

    lines += [
        "",
        "The generalisation gap (Δ_gen) measures how much harder the BiasedDefender is vs. Uniform. ",
        "A gap close to 0 means the scripted bias does not present a meaningful challenge; ",
        "a positive gap signals the attacker genuinely struggles against the biased placement strategy.",
        "",
        "The next step to produce a larger, more reliable Δ_gen is to train a full `AdversarialDefender` "
        "via RL (already implemented in `defender.py` via `AdversarialDefender`), "
        "then pass `--adversarial-defender <path>` to the evaluator.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Full evaluation suite for Battleship policies.")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per policy per defender")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results", help="Directory to save outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suite_results = run_suite(n_episodes=args.episodes, seed=args.seed)

    # Save JSON
    json_path = output_dir / "final_evaluation.json"
    serializable = {}
    for policy, data in suite_results.items():
        serializable[policy] = {
            "description": data["description"],
            "eval": {k: v for k, v in data["eval"].items() if k != "replays"},
            "diagnostics": data["diagnostics"],
        }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, default=float)
    print(f"\nRaw JSON saved to: {json_path}")

    # Save Markdown
    md_content = format_markdown(suite_results, n_episodes=args.episodes, seed=args.seed)
    md_path = output_dir / "final_evaluation.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Markdown report saved to: {md_path}")

    # Print summary to stdout
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for policy, data in suite_results.items():
        e = data["eval"]
        print(f"{policy:12s} | Uniform: {e['uniform']['mean']:.1f} | Biased: {e['biased']['mean']:.1f} | Δ_gen: {e['gap']:+.2f}")


if __name__ == "__main__":
    main()
