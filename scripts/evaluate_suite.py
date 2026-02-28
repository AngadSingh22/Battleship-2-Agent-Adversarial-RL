#!/usr/bin/env python3
"""
scripts/evaluate_suite.py  [OPTIMIZED: ProcessPoolExecutor + GPU particle filter]
===================================================================================
Unified evaluation suite across all policies and the full defender ladder.

Hardware utilization:
  CPU: ProcessPoolExecutor(max_workers=N_WORKERS) parallelizes episode rollouts.
       Each (policy, defender) combo runs N_WORKERS episodes in parallel.
  GPU: ParticleBeliefAgent uses torch.cuda for batch particle filtering.
       Automatically detected; falls back to CPU if unavailable.

Defender ladder (5 modes):
  UNIFORM   UniformRandomDefender
  EDGE      EdgeBiasedDefender     (ships prefer boundary cells)
  CLUSTER   ClusteredDefender      (ships cluster together)
  SPREAD    SpreadDefender         (ships maximise inter-ship distance)
  PARITY    ParityDefender         (largest ship locked to top-left quadrant)

Policies evaluated:
  random   / heuristic / particle

For each (policy, defender) pair:
  - 100 episodes, seed protocol: episode_seed = base_seed + episode_idx
  - Core metrics: mean, std, p90, fail_rate, fallback_rate
  - Behavioral diagnostics: time_to_first_hit, hit_rate, hunt_efficiency,
    revisit_rate, shots_per_ship_sunk  (computed over ALL 100 episodes)
  - Placement distribution metrics: edge_occupancy, mean_dist_to_edge (normalized),
    occupancy_entropy, kl_from_uniform

Saves:
  results/final_evaluation.md   — human-readable combined table + interpretation
  results/final_evaluation.json — raw results for downstream CI/analysis

Usage:
  PYTHONPATH=. .venv/bin/python3 scripts/evaluate_suite.py [--episodes N] [--seed S]
                                  [--workers W (default: 16)]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

from battleship_rl.agents.defender import (
    UniformRandomDefender,
    EdgeBiasedDefender,
    ClusteredDefender,
    SpreadDefender,
    ParityDefender,
)
from battleship_rl.baselines.diagnosis_baselines import RandomTester, GreedySplitTester
from battleship_rl.envs.battleship_env import BattleshipEnv
from battleship_rl.envs.diagnosis_env import DiagnosisEnv
from battleship_rl.eval.evaluate import _run_episode, _make_policy
from battleship_rl.eval.diagnostics import summarize_diagnostics, format_diagnostics
from battleship_rl.eval.metrics import generalization_gap, summarize
from battleship_rl.eval import defender_metrics as dm


# ---------------------------------------------------------------------------
# Defender registry — ordered for display
# ---------------------------------------------------------------------------
DEFENDERS: list[tuple[str, Any]] = [
    ("UNIFORM",  UniformRandomDefender),
    ("EDGE",     EdgeBiasedDefender),
    ("CLUSTER",  ClusteredDefender),
    ("SPREAD",   SpreadDefender),
    ("PARITY",   ParityDefender),
]


# ---------------------------------------------------------------------------
# Worker function — runs one episode, must be top-level for multiprocessing
# ---------------------------------------------------------------------------
def _worker(args: tuple) -> dict:
    """Top-level worker: runs one Battleship episode. Returns metrics dict."""
    policy_type, defender_cls, env_config, episode_seed, record_steps = args
    rng = np.random.default_rng(episode_seed)
    defender = defender_cls()
    env = BattleshipEnv(config=env_config, defender=defender)
    policy = _make_policy(policy_type, env, rng, model_path=None)
    length, truncated, steps, fallback, rejections = _run_episode(
        env, policy, seed=episode_seed, record_steps=record_steps
    )
    return {
        "length": length,
        "truncated": truncated,
        "steps": steps,
        "fallback": fallback,
        "rejections": rejections,
    }


# ---------------------------------------------------------------------------
# Parallel episode runner
# ---------------------------------------------------------------------------
def run_episodes_parallel(
    policy_type: str,
    defender_cls,
    n_episodes: int,
    base_seed: int,
    env_config: dict,
    n_workers: int = 16,
    record_steps: bool = False,
) -> list[dict]:
    """Run n_episodes in parallel using ProcessPoolExecutor."""
    args_list = [
        (policy_type, defender_cls, env_config, base_seed + i, record_steps)
        for i in range(n_episodes)
    ]
    results = [None] * n_episodes
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=None) as executor:
        futures = {executor.submit(_worker, args): i for i, args in enumerate(args_list)}
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
    return results


# ---------------------------------------------------------------------------
# Defender metrics (computed once per defender, independent of policy)
# ---------------------------------------------------------------------------
def _get_defender_metrics(
    defender_cls, env_config: dict, n_samples: int = 500, seed: int = 0,
    q_ref=None
) -> dict:
    env = BattleshipEnv(config=env_config, defender=UniformRandomDefender())
    defender_instance = defender_cls()
    return dm.summarize_defender(
        defender_instance,
        board_size=(env.height, env.width),
        ships=env.ship_lengths,
        n_samples=n_samples,
        seed=seed,
        q_ref=q_ref,
    )


# ---------------------------------------------------------------------------
# Main suite runner
# ---------------------------------------------------------------------------
def run_suite(
    n_episodes: int = 100,
    seed: int = 42,
    n_workers: int = 16,
    env_config: dict | None = None,
) -> dict:
    env_config = env_config or {}
    suite_results: dict[str, dict] = {}

    # Pre-compute defender distribution metrics
    # IMPORTANT: compute UNIFORM first, use its q_marginal as JSD reference for all others
    print("\n[1/2] Computing defender distribution metrics (5000 samples for reference, 500 for modes)...")
    dist_metrics: dict[str, dict] = {}

    # Step 1: UNIFORM reference — 5000 samples for stable JSD
    uniform_m = _get_defender_metrics(UniformRandomDefender, env_config, n_samples=5000, seed=seed)
    q_uniform = uniform_m["q_marginal"]  # used as JSD reference for all other modes
    dist_metrics["UNIFORM"] = uniform_m
    dist_metrics["UNIFORM"]["shift_metric"] = 0.0  # By definition
    print(f"  {'UNIFORM':<10} JSD=0.00000  (reference)")

    # Step 2: All other defenders — pass q_uniform as reference
    for name, cls in DEFENDERS[1:]:
        m = _get_defender_metrics(cls, env_config, n_samples=500, seed=seed, q_ref=q_uniform)
        dist_metrics[name] = m
        jsd = m["shift_metric"]
        flag = " ⚠ weak_shift" if jsd < 0.005 else ""
        print(f"  {name:<10} JSD={jsd:.5f}{flag}")

    # Evaluate each policy × defender combo
    policies = {
        "random":    "Random baseline — fires at random valid cells",
        "heuristic": "ProbMap heuristic — probability-sampling constraint backtracker",
        "particle":  "Particle Belief SMC — full belief filter (P=500)",
    }

    print(f"\n[2/2] Running {n_episodes} episodes per (policy × defender) with {n_workers} parallel workers...")
    for policy_type, policy_desc in policies.items():
        print(f"\n  Policy: {policy_type.upper()}")
        policy_data: dict[str, Any] = {"description": policy_desc, "defenders": {}}

        for def_name, def_cls in DEFENDERS:
            t0 = time.perf_counter()
            episode_results = run_episodes_parallel(
                policy_type=policy_type,
                defender_cls=def_cls,
                n_episodes=n_episodes,
                base_seed=seed,
                env_config=env_config,
                n_workers=n_workers,
                record_steps=True,  # collect step data for diagnostics on ALL episodes
            )
            elapsed = time.perf_counter() - t0

            lengths = [r["length"] for r in episode_results]
            truncated_flags = [r["truncated"] for r in episode_results]
            fallback_flags = [r["fallback"] for r in episode_results]
            all_steps = [r["steps"] for r in episode_results]
            total_rejs: dict = {}
            for r in episode_results:
                for k, v in r["rejections"].items():
                    total_rejs[k] = total_rejs.get(k, 0) + v

            perf = summarize(lengths, truncated_flags)
            perf["fallback_rate"] = float(np.mean(fallback_flags))
            perf["rejections"] = total_rejs
            diag = summarize_diagnostics(all_steps)

            policy_data["defenders"][def_name] = {
                "perf": perf,
                "diagnostics": diag,
                "dist_metrics": dist_metrics[def_name],
            }
            print(
                f"    {def_name:<10}: {perf['mean']:.1f}±{perf['std']:.1f}  "
                f"p90={perf['p90']:.1f}  fr={perf['fallback_rate']:.2f}  "
                f"[{elapsed:.1f}s]"
            )

        # Compute Δ_gen per non-uniform defender
        uniform_mean = policy_data["defenders"]["UNIFORM"]["perf"]["mean"]
        policy_data["delta_gen"] = {}
        for def_name, _ in DEFENDERS[1:]:
            policy_data["delta_gen"][def_name] = (
                policy_data["defenders"][def_name]["perf"]["mean"] - uniform_mean
            )
        suite_results[policy_type] = policy_data

    return suite_results, dist_metrics


# ---------------------------------------------------------------------------
# Diagnosis POMDP suite
# ---------------------------------------------------------------------------
def run_diagnosis_suite(n_episodes: int = 100, seed: int = 42) -> dict:
    """Run diagnosis POMDP shift evaluation: random and greedy baselines × 3 fault distributions."""
    print("\n[3/3] Running DiagnosisEnv fault-shift suite...")
    distributions = ["uniform", "clustered", "rare_hard"]
    baselines = {
        "random":  lambda rng: RandomTester(rng),
        "greedy":  lambda rng: GreedySplitTester(rng),
    }
    results: dict[str, dict] = {}

    # Fault histogram sanity check
    print("\n  Fault distribution sanity check (1000 resets per mode):")
    for dist in distributions:
        env_check = DiagnosisEnv(fault_distribution=dist)
        counts = np.zeros(env_check.n_components, dtype=int)
        for ep in range(1000):
            _, info = env_check.reset(seed=seed + ep)
            counts[info["faulty_component"]] += 1
        top = sorted(zip(counts, range(len(counts))), reverse=True)[:4]
        print(f"    {dist:<12}: " + "  ".join(f"c{c}={n}" for n, c in top))

    for baseline_name, baseline_fn in baselines.items():
        results[baseline_name] = {}
        for dist in distributions:
            env = DiagnosisEnv(fault_distribution=dist)
            agent = baseline_fn(np.random.default_rng(seed + 1))

            steps_to_correct = []
            steps_to_fail = []
            failures = 0

            for ep in range(n_episodes):
                obs, info = env.reset(seed=seed + ep)
                agent.reset()
                done = False
                t = 0
                success = False
                while not done:
                    action = agent.act(obs, info, env)
                    obs, reward, terminated, truncated, info = env.step(action)
                    t += 1
                    if terminated:
                        success = info.get("outcome") == "correct"
                        done = True
                    elif truncated:
                        done = True
                if success:
                    steps_to_correct.append(t)
                else:
                    failures += 1
                    steps_to_fail.append(t)

            mean_steps_success = float(np.mean(steps_to_correct)) if steps_to_correct else float("nan")
            std_steps_success = float(np.std(steps_to_correct)) if len(steps_to_correct) > 1 else 0.0
            all_steps = steps_to_correct + steps_to_fail
            mean_steps_all = float(np.mean(all_steps))
            success_rate = 1.0 - (failures / n_episodes)

            results[baseline_name][dist] = {
                "mean_steps_success": mean_steps_success,
                "std_steps_success": std_steps_success,
                "mean_steps_all": mean_steps_all,
                "success_rate": success_rate,
                "failure_rate": failures / n_episodes,
                "n_success": len(steps_to_correct),
            }
            print(
                f"  {baseline_name:<8} / {dist:<12}: "
                f"succ_rate={success_rate:.2f}  mean_succ={mean_steps_success:.2f}±{std_steps_success:.2f}  "
                f"mean_all={mean_steps_all:.2f}  n_succ={len(steps_to_correct)}"
            )

    # Δ_gen per distribution per baseline
    for baseline_name in baselines:
        uniform_mean = results[baseline_name]["uniform"]["mean_steps_success"]
        for dist in ["clustered", "rare_hard"]:
            d = results[baseline_name][dist]["mean_steps_success"]
            results[baseline_name][dist]["delta_gen"] = d - uniform_mean if not (
                np.isnan(d) or np.isnan(uniform_mean)
            ) else float("nan")

    return results


# ---------------------------------------------------------------------------
# Markdown formatter
# ---------------------------------------------------------------------------
def format_markdown(suite_results: dict, dist_metrics: dict, diag_results: dict,
                    n_episodes: int, seed: int, n_workers: int) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Final Evaluation Suite Results",
        "",
        f"**Generated:** {now}  |  **Episodes:** {n_episodes}  |  **Seed:** {seed}  "
        f"|  **Workers:** {n_workers}",
        "",
        "---",
        "",
        "## Defender Distribution Metrics",
        "",
        "*(500 samples each. `shift_metric` = JSD(q_mode || q_UNIFORM_defender), so UNIFORM=0.000 by construction.)*",
        "",
        "| Defender   | EdgeOcc | MeanDist† | Entropy (bits) | JSD vs UNIFORM-defender | Flag |",
        "|------------|---------|-----------|----------------|------------------------|------|",
    ]
    for name, m in dist_metrics.items():
        jsd = m.get("shift_metric", m.get("kl_from_uniform", float("nan")))
        is_ref = (name == "UNIFORM")
        if is_ref:
            flag = "(reference)"
        elif jsd < 0.005:
            flag = "⚠ weak_shift"
        elif jsd >= 0.02:
            flag = "✓ real shift"
        else:
            flag = "~"
        lines.append(
            f"| {name:<10} | {m['edge_occupancy']:>7.3f} | {m['mean_dist_to_edge']:>9.3f} "
            f"| {m['occupancy_entropy']:>14.3f} | {jsd:>22.5f} | {flag} |"
        )
    lines += [
        "",
        "†`mean_dist_to_edge` normalized by `floor(min(H,W)/2)` → range [0, 1].",
        "",
        "---",
        "",
        "## Performance + Diagnostics per Policy",
        "",
    ]

    for policy_type, policy_data in suite_results.items():
        lines += [
            f"### Policy: {policy_type.capitalize()}",
            f"> {policy_data['description']}",
            "",
            "| Defender | Mean±Std | p90 | FailRate | Fallback | Δ_gen | HitRate | T-to-1stHit |",
            "|----------|----------|-----|----------|----------|-------|---------|-------------|",
        ]
        for def_name, _ in DEFENDERS:
            d = policy_data["defenders"][def_name]
            p = d["perf"]
            diag = d["diagnostics"]
            gap = policy_data["delta_gen"].get(def_name, 0.0)
            gap_str = f"{gap:+.2f}" if def_name != "UNIFORM" else "—"
            lines.append(
                f"| {def_name:<8} | {p['mean']:.1f}±{p['std']:.1f} | {p['p90']:.1f} "
                f"| {p['fail_rate']:.3f} | {p.get('fallback_rate', 0):.3f} | {gap_str} "
                f"| {diag['hit_rate']:.3f} | {diag['time_to_first_hit']:.1f} |"
            )
        lines.append("")

        # p90 delta for sign-off gate
        uniform_p90 = policy_data["defenders"]["UNIFORM"]["perf"]["p90"]
        p90_deltas = {
            n: policy_data["defenders"][n]["perf"]["p90"] - uniform_p90
            for n, _ in DEFENDERS[1:]
        }
        best_p90_mode = max(p90_deltas, key=p90_deltas.get)
        best_p90_val = p90_deltas[best_p90_mode]
        lines.append(
            f"> **p90 Δ:** Largest tail shift = {best_p90_mode} "
            f"({best_p90_val:+.1f} vs UNIFORM). "
            + ("✓ meets p90≥+2 gate" if best_p90_val >= 2.0 else "⚠ below p90≥+2 gate")
        )
        lines.append("")

    # Sign-off gates
    jsd_ok = sum(
        1 for name, m in dist_metrics.items()
        if name != "UNIFORM" and m.get("shift_metric", 0) >= 0.005
    )
    lines += [
        "---",
        "",
        "## Sign-off Gate Check",
        "",
        f"1. **JSD > 0.005 for ≥2 non-UNIFORM modes:** {jsd_ok}/4 modes qualify — "
        + ("✓ PASS" if jsd_ok >= 2 else "✗ FAIL"),
        "",
    ]

    # Diagnosis suite
    if diag_results:
        lines += [
            "---",
            "",
            "## Diagnosis POMDP Fault-Shift Results",
            "",
            "| Baseline | Distribution | Success Rate | Mean Steps (Success) | Δ_gen | Mean Steps (All) |",
            "|----------|-------------|--------------|----------------------|-------|------------------|",
        ]
        for bl_name, bl_data in diag_results.items():
            for dist, stats in bl_data.items():
                gap_str = f"{stats.get('delta_gen', 0.0):+.2f}" if dist != "uniform" else "—"
                lines.append(
                    f"| {bl_name:<8} | {dist:<12} | {stats['success_rate']:.3f} "
                    f"| {stats['mean_steps_success']:.2f} ± {stats['std_steps_success']:.2f} "
                    f"| {gap_str} | {stats['mean_steps_all']:.2f} |"
                )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Full optimized evaluation suite.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    suite_results, dist_metrics = run_suite(
        n_episodes=args.episodes, seed=args.seed, n_workers=args.workers
    )
    diag_results = run_diagnosis_suite(n_episodes=args.episodes, seed=args.seed)
    total_time = time.perf_counter() - t_start

    # Serialize (remove non-JSON-serializable items)
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    json_payload = _clean({
        "metadata": {"episodes": args.episodes, "seed": args.seed, "workers": args.workers,
                     "total_time_s": total_time},
        "dist_metrics": dist_metrics,
        "suite": {
            pol: {
                "description": d["description"],
                "delta_gen": d["delta_gen"],
                "defenders": {
                    def_name: {"perf": dv["perf"], "diagnostics": dv["diagnostics"]}
                    for def_name, dv in d["defenders"].items()
                },
            }
            for pol, d in suite_results.items()
        },
        "diagnosis": diag_results,
    })

    json_path = out_dir / "final_evaluation.json"
    with json_path.open("w") as f:
        json.dump(json_payload, f, indent=2, default=float)
    print(f"\nJSON saved: {json_path}")

    md = format_markdown(suite_results, dist_metrics, diag_results,
                         args.episodes, args.seed, args.workers)
    md_path = out_dir / "final_evaluation.md"
    with md_path.open("w") as f:
        f.write(md)
    print(f"Markdown saved: {md_path}")

    print(f"\nTotal wall time: {total_time:.1f}s")
    print("\n=== FINAL SUMMARY ===")
    for pol, d in suite_results.items():
        u = d["defenders"]["UNIFORM"]["perf"]["mean"]
        gaps = "  ".join(f"{n}:{v:+.1f}" for n, v in d["delta_gen"].items())
        print(f"{pol:12s}  UNIFORM={u:.1f}  Δ_gen → {gaps}")


if __name__ == "__main__":
    main()
