"""Review-response statistics for E5.

Prints exactly the numbers quoted in REPORT.md, so no figure in the report is
typed by hand:

  * paired contrasts of every arm against `real` and against `mask`,
  * Clopper-Pearson exact intervals for the degenerate 0/n and n/n arms,
  * the Phi-distribution comparison between `real` and `noise`, annotated with
    the number of decisions each mean is averaged over (they are not the same
    state distribution),
  * measured vs nominal simulator cost per decision.

Usage: uv run python experiments/E5_ablation/scripts/review_stats.py [tag]
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e5_lib import boot_diff_ci  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))

ORDER = ["off", "off_matched", "off_true_matched", "real", "shuffled",
         "constant", "noise", "shuffled_keepmask", "noise_keepmask", "nosurv",
         "cheap", "cheap_matched", "cheap_true_matched", "mask",
         "mask_matched", "mask_true_matched"]


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial interval without scipy: bisection on the Beta quantile."""
    from math import comb

    def cdf_le(p, k, n):            # P(X <= k)
        return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))

    lo, hi = 0.0, 1.0
    if k > 0:
        a, b = 0.0, 1.0
        for _ in range(200):
            mid = (a + b) / 2
            if 1 - cdf_le(mid, k - 1, n) > alpha / 2:
                b = mid
            else:
                a = mid
        lo = a
    if k < n:
        a, b = 0.0, 1.0
        for _ in range(200):
            mid = (a + b) / 2
            if cdf_le(mid, k, n) < alpha / 2:
                b = mid
            else:
                a = mid
        hi = a
    return lo, hi


def rates(eps, outcome):
    return [1.0 if e["outcome"] == outcome else 0.0 for e in eps]


def main(tag=""):
    arms = {}
    for k in ORDER:
        p = os.path.join(RESULTS, f"E5_{k}{tag}.json")
        if os.path.exists(p):
            arms[k] = json.load(open(p))
    n = arms["real"]["n"]
    print(f"=== tag={tag or '(primary)'}  n={n} ===\n")

    print("-- measured vs nominal simulator cost per decision --")
    for k, a in arms.items():
        g = a["aggregate"]
        print(f"{k:>20}  nominal {g['sim_steps_per_decision']['mean']:7.0f}  "
              f"measured {g['actual_steps_per_decision']['mean']:7.0f}  "
              f"ratio {g['actual_over_nominal']['mean']:.3f}")

    print("\n-- exact (Clopper-Pearson) intervals on goal and trap counts --")
    for k, a in arms.items():
        for o in ("goal", "trap"):
            kk = int(sum(rates(a["episodes"], o)))
            lo, hi = clopper_pearson(kk, a["n"])
            print(f"{k:>20} {o:>4}: {kk}/{a['n']}  exact [{lo:.3f}, {hi:.3f}]")

    for ref in ("real", "mask_true_matched"):
        print(f"\n-- paired contrasts vs `{ref}` (positive = {ref} higher) --")
        base = arms[ref]["episodes"]
        for k, a in arms.items():
            if k == ref:
                continue
            row = []
            for o in ("goal", "trap", "fuel"):
                d, lo, hi = boot_diff_ci(rates(base, o), rates(a["episodes"], o))
                row.append(f"{o} {d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
            print(f"{k:>20}  " + "   ".join(row))

    print("\n-- Phi distributions are measured over different state "
          "distributions --")
    for k in ("real", "noise", "shuffled", "constant", "mask"):
        if k not in arms:
            continue
        a = arms[k]
        eps = a["episodes"]
        mp = [e["mean_phi"] for e in eps if e["mean_phi"] == e["mean_phi"]]
        st = [e["steps"] for e in eps]
        print(f"{k:>20}  mean Phi {np.mean(mp):.3f} averaged over "
              f"{np.mean(st):.2f} decisions/episode "
              f"(total {int(np.sum(st))} decisions across {len(eps)} episodes)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
