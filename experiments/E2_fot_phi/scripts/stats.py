"""Bootstrap statistics for E2. Every mean reported carries a CI95 from here."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

B_DEFAULT = 10000


def boot_mean(x: Sequence[float], n_boot: int = B_DEFAULT, seed: int = 0) -> Dict:
    """Mean of x with a percentile bootstrap CI95."""
    a = np.asarray(list(x), dtype=np.float64)
    n = len(a)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    if n == 1:
        return {"mean": float(a[0]), "lo": float(a[0]), "hi": float(a[0]), "n": 1}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = a[idx].mean(axis=1)
    return {
        "mean": float(a.mean()),
        "lo": float(np.percentile(means, 2.5)),
        "hi": float(np.percentile(means, 97.5)),
        "n": int(n),
    }


def boot_diff(
    x: Sequence[float], y: Sequence[float], n_boot: int = B_DEFAULT, seed: int = 0
) -> Dict:
    """mean(x) - mean(y) with a percentile bootstrap CI95 and a two-sided p.

    Unpaired resampling: the two arms are independent runs, and once budgets are
    matched they are not paired by episode either.
    """
    a = np.asarray(list(x), dtype=np.float64)
    b = np.asarray(list(y), dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_x": len(a), "n_y": len(b), "p_two_sided": float("nan")}
    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    db = b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(axis=1)
    d = da - db
    frac_le0 = float((d <= 0).mean())
    p = 2.0 * min(frac_le0, 1.0 - frac_le0)
    return {
        "diff": float(a.mean() - b.mean()),
        "lo": float(np.percentile(d, 2.5)),
        "hi": float(np.percentile(d, 97.5)),
        "n_x": int(len(a)),
        "n_y": int(len(b)),
        "p_two_sided": float(min(1.0, p)),
    }


def fmt(d: Dict, pct: bool = False, digits: int = 3) -> str:
    """Render a boot_mean dict as 'mean [lo, hi]'."""
    if pct:
        return (f"{100 * d['mean']:.1f}% [{100 * d['lo']:.1f}, "
                f"{100 * d['hi']:.1f}]")
    return f"{d['mean']:.{digits}f} [{d['lo']:.{digits}f}, {d['hi']:.{digits}f}]"


def fmt_diff(d: Dict, pct: bool = False, digits: int = 3) -> str:
    if pct:
        return (f"{100 * d['diff']:+.1f} pp [{100 * d['lo']:+.1f}, "
                f"{100 * d['hi']:+.1f}]")
    return f"{d['diff']:+.{digits}f} [{d['lo']:+.{digits}f}, {d['hi']:+.{digits}f}]"


def boot_did(xa, ya, xb, yb, n_boot: int = B_DEFAULT, seed: int = 0) -> Dict:
    """Difference in differences: (mean xa - mean ya) - (mean xb - mean yb).

    Used for P2: xa/ya are arm/baseline on the split where the baseline commits
    early, xb/yb the same pair on the other split. A prediction that "the gain
    concentrates" on split A is a claim that this quantity is positive, and it
    needs its own interval: two separately significant effects are not evidence
    of a difference between them, and neither are two separately non-significant
    ones evidence of no difference.
    """
    rng = np.random.default_rng(seed)
    arrs = [np.asarray(list(v), dtype=np.float64) for v in (xa, ya, xb, yb)]
    if any(len(a) == 0 for a in arrs):
        return {"did": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p_two_sided": float("nan")}
    ms = [a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
          for a in arrs]
    d = (ms[0] - ms[1]) - (ms[2] - ms[3])
    frac_le0 = float((d <= 0).mean())
    return {
        "did": float((arrs[0].mean() - arrs[1].mean())
                     - (arrs[2].mean() - arrs[3].mean())),
        "lo": float(np.percentile(d, 2.5)),
        "hi": float(np.percentile(d, 97.5)),
        "n": [int(len(a)) for a in arrs],
        "p_two_sided": float(min(1.0, 2.0 * min(frac_le0, 1.0 - frac_le0))),
    }
