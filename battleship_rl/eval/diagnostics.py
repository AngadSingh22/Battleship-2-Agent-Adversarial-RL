"""
battleship_rl/eval/diagnostics.py
====================================
Behavioural diagnostic metrics for Battleship attacker policies.

All functions operate on a list of episode step-dicts produced by `_run_episode`
with `record_steps=True`. Each step-dict has:
  { "r": int, "c": int, "type": str }   # type is outcome_type from info

Metrics
-------
  time_to_first_hit      : mean number of shots before the first hit
  hit_rate               : fraction of shots that result in a "HIT" or "SUNK"
  hunt_efficiency        : ratio of hunt-phase shots to total shots
                           hunt phase = shots before the first hit
  revisit_rate           : fraction of shots that revisit an already-fired cell
                           (should always be 0 for a valid policy, validates env)
  shots_per_ship_sunk    : mean shots required to sink each ship
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np


def time_to_first_hit(episodes: List[List[dict]]) -> float:
    """Mean number of shots before first HIT/SUNK, excluding episodes with no hits."""
    counts = []
    for ep in episodes:
        for i, step in enumerate(ep):
            if step.get("type") in ("HIT", "SUNK"):
                counts.append(i)  # 0-indexed: i shots before this one
                break
    return float(np.mean(counts)) if counts else float("nan")


def hit_rate(episodes: List[List[dict]]) -> float:
    """Fraction of shots that resulted in a HIT or SUNK outcome."""
    total = sum(len(ep) for ep in episodes)
    hits = sum(
        1 for ep in episodes for step in ep if step.get("type") in ("HIT", "SUNK")
    )
    return hits / total if total > 0 else float("nan")


def hunt_efficiency(episodes: List[List[dict]]) -> float:
    """Fraction of shots that occurred BEFORE the first hit (hunt-phase shots)."""
    hunt_shots = 0
    total_shots = 0
    for ep in episodes:
        total_shots += len(ep)
        for i, step in enumerate(ep):
            if step.get("type") in ("HIT", "SUNK"):
                hunt_shots += i
                break
        else:
            hunt_shots += len(ep)  # no hit at all → all hunt
    return hunt_shots / total_shots if total_shots > 0 else float("nan")


def revisit_rate(episodes: List[List[dict]]) -> float:
    """Fraction of shots fired at a cell that was already targeted (should be 0)."""
    revisits = 0
    total = 0
    for ep in episodes:
        seen = set()
        for step in ep:
            cell = (step["r"], step["c"])
            if cell in seen:
                revisits += 1
            seen.add(cell)
            total += 1
    return revisits / total if total > 0 else 0.0


def shots_per_ship_sunk(episodes: List[List[dict]]) -> float:
    """Mean number of shots between consecutive sunk events."""
    intervals = []
    for ep in episodes:
        last_sunk = 0
        sunk_count = 0
        for i, step in enumerate(ep):
            if step.get("type") == "SUNK":
                intervals.append(i + 1 - last_sunk)
                last_sunk = i + 1
                sunk_count += 1
    return float(np.mean(intervals)) if intervals else float("nan")


def summarize_diagnostics(episodes: List[List[dict]]) -> Dict[str, float]:
    return {
        "time_to_first_hit":    time_to_first_hit(episodes),
        "hit_rate":             hit_rate(episodes),
        "hunt_efficiency":      hunt_efficiency(episodes),
        "revisit_rate":         revisit_rate(episodes),
        "shots_per_ship_sunk":  shots_per_ship_sunk(episodes),
    }


def format_diagnostics(d: Dict[str, float]) -> str:
    lines = ["| Metric | Value |", "|--------|-------|"]
    labels = {
        "time_to_first_hit":    "Time to First Hit (shots)",
        "hit_rate":             "Hit Rate",
        "hunt_efficiency":      "Hunt Phase Fraction",
        "revisit_rate":         "Revisit Rate (should be 0)",
        "shots_per_ship_sunk":  "Shots Per Ship Sunk",
    }
    for k, label in labels.items():
        v = d.get(k, float("nan"))
        lines.append(f"| {label} | {v:.4f} |")
    return "\n".join(lines)
