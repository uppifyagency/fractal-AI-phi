"""Render the E5 result JSONs as the markdown tables used in REPORT.md."""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))

ORDER = ["off", "off_matched", "off_true_matched", "real", "shuffled",
         "constant", "noise", "shuffled_keepmask", "noise_keepmask", "nosurv",
         "cheap", "cheap_matched", "cheap_true_matched", "mask",
         "mask_matched", "mask_true_matched"]


def ci(d):
    return f"{d['mean']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}]"


def main(tag=""):
    s = json.load(open(os.path.join(RESULTS, f"E5_summary{tag}.json")))
    n = s["n_seeds"]
    print(f"### Outcome rates, n={n} paired seeds, mean [CI95 bootstrap, 10k]\n")
    print("| arm | mode | N | n | nominal steps/decision | measured steps/decision | "
          "measured/nominal | goal | trap | fuel |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k in ORDER:
        a = s["arms"][k]
        g = a["aggregate"]
        print(f"| `{k}` | {a['mode']} | {a['N']} | {a['n']} | "
              f"{g['sim_steps_per_decision']['mean']:.0f} | "
              f"{g['actual_steps_per_decision']['mean']:.0f} | "
              f"{g['actual_over_nominal']['mean']:.3f} | {ci(g['goal_rate'])} | "
              f"{ci(g['trap_rate'])} | {ci(g['fuel_rate'])} |")

    print(f"\n### Episode diagnostics, n={n}\n")
    print("| arm | steps | mean b_eff | mean Phi | mean ESS |")
    print("|---|---|---|---|---|")
    for k in ORDER:
        g = s["arms"][k]["aggregate"]
        mp = g["mean_phi"]
        mps = "n/a" if mp["mean"] != mp["mean"] else ci(mp)
        print(f"| `{k}` | {ci(g['steps'])} | {ci(g['mean_b_eff'])} | {mps} | "
              f"{ci(g['mean_ess'])} |")

    print(f"\n### Paired contrasts against `real`, n={n} "
          "(positive = real is higher)\n")
    print("| arm | d goal rate [CI95] | d trap rate [CI95] |")
    print("|---|---|---|")
    for k in ORDER:
        if k == "real":
            continue
        c = s["contrasts_vs_real"][k]
        g, t = c["goal_rate_real_minus_arm"], c["trap_rate_real_minus_arm"]
        print(f"| `{k}` | {g['diff']:+.2f} [{g['lo']:+.2f}, {g['hi']:+.2f}] | "
              f"{t['diff']:+.2f} [{t['lo']:+.2f}, {t['hi']:+.2f}] |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
