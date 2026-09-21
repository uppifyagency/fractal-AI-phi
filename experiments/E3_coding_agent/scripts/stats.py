#!/usr/bin/env python3
"""Bootstrap confidence intervals. Every mean E3 reports carries one."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

BOOT = 10000


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = BOOT,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float, int]:
    """Return (mean, lo, hi, n) with a percentile bootstrap CI."""
    v = np.asarray(list(values), dtype=np.float64)
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    if n == 1:
        return float(v[0]), float(v[0]), float(v[0]), 1
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi), n


def bootstrap_paired_diff(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = BOOT,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Paired a - b: mean difference, CI95, and the sign test."""
    a = np.asarray(list(a), dtype=np.float64)
    b = np.asarray(list(b), dtype=np.float64)
    assert a.shape == b.shape, "paired arms must have equal n"
    d = a - b
    mean, lo, hi, n = bootstrap_ci(d, n_boot=n_boot, alpha=alpha, seed=seed)
    return {
        "mean_diff": mean, "ci95": [lo, hi], "n": n,
        "frac_negative": float((d < 0).mean()),
        "frac_positive": float((d > 0).mean()),
        "frac_zero": float((d == 0).mean()),
        "significant": bool(hi < 0 or lo > 0),
    }


def summarize(values: Sequence[float], seed: int = 0) -> dict:
    mean, lo, hi, n = bootstrap_ci(values, seed=seed)
    return {"mean": mean, "ci95": [lo, hi], "n": n}


def rate_ci(successes: int, n: int, n_boot: int = BOOT, seed: int = 0) -> dict:
    """Bootstrap CI on a proportion, reported the same way as any other mean."""
    v = np.zeros(n)
    v[:successes] = 1.0
    mean, lo, hi, _ = bootstrap_ci(v, n_boot=n_boot, seed=seed)
    return {"mean": mean, "ci95": [lo, hi], "n": n}


def fmt(d: dict, digits: int = 3) -> str:
    return (f"{d['mean']:.{digits}f} "
            f"[{d['ci95'][0]:.{digits}f}, {d['ci95'][1]:.{digits}f}]")
