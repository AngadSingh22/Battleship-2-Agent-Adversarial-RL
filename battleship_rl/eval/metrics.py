from __future__ import annotations

from typing import Iterable, Sequence

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


def generalization_gap(mean_biased: float, mean_uniform: float) -> float:
    return float(mean_biased - mean_uniform)


def summarize(lengths: Sequence[int], truncated_flags: Sequence[bool]) -> dict:
    lengths_arr = np.array(lengths, dtype=np.float32)
    return {
        "mean": float(np.mean(lengths_arr)) if lengths else float("nan"),
        "std": float(np.std(lengths_arr)) if lengths else float("nan"),
        "p90": shots_90th_percentile(lengths),
        "fail_rate": failure_rate(truncated_flags),
    }
