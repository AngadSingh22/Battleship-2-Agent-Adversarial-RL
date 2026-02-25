from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def mean_shots_to_win(lengths: Sequence[int]) -> float:
    if not lengths:
        return float("nan")
    return float(np.mean(lengths))


def shots_90th_percentile(lengths: Sequence[int]) -> float:
    if not lengths:
        return float("nan")
    return float(np.percentile(lengths, 90))


def failure_rate(truncated_flags: Sequence[bool]) -> float:
    if not truncated_flags:
        return 0.0
    return float(np.mean(truncated_flags))


def generalization_gap(mean_challenge: float, mean_uniform: float) -> float:
    """Δ_gen = E[τ]_challenge − E[τ]_uniform  (formulation solution Eq. 7).

    A positive gap means the agent takes more shots against the challenge
    distribution than under uniform random placement, indicating fragility.

    Args:
        mean_challenge: mean shots-to-win under adversarial / biased placements.
        mean_uniform:   mean shots-to-win under uniform random placements.
    """
    return float(mean_challenge - mean_uniform)


def calibration_error(
    pred_probs: Sequence[float],
    realized_hits: Sequence[int],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) over per-cell hit predictions.

    Args:
        pred_probs:    Predicted occupancy probabilities p_t(c) for each cell.
        realized_hits: Ground-truth binary hit indicators for the same cells.
        n_bins:        Number of equal-width bins for calibration.

    Returns:
        ECE in [0, 1] — lower is better.
    """
    pred_probs = np.asarray(pred_probs, dtype=np.float64)
    realized_hits = np.asarray(realized_hits, dtype=np.float64)
    if pred_probs.size == 0:
        return float("nan")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (pred_probs >= lo) & (pred_probs < hi)
        if not np.any(mask):
            continue
        bin_acc = float(np.mean(realized_hits[mask]))
        bin_conf = float(np.mean(pred_probs[mask]))
        bin_frac = float(np.sum(mask)) / float(pred_probs.size)
        ece += bin_frac * abs(bin_acc - bin_conf)
    return ece


def summarize(lengths: Sequence[int], truncated_flags: Sequence[bool]) -> dict:
    lengths_arr = np.array(lengths, dtype=np.float32)
    return {
        "mean": float(np.mean(lengths_arr)) if lengths else float("nan"),
        "std": float(np.std(lengths_arr)) if lengths else float("nan"),
        "p90": shots_90th_percentile(lengths),
        "fail_rate": failure_rate(truncated_flags),
    }
