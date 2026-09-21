"""E5 review response: gamma-matched comparison.

The BRIEF pre-registered "at the best gamma from E1". E5 was run at gamma = 1.0,
fixed a priori because E1 had not reported yet. E1's REPORT.md now puts the
goal-rate optimum at gamma = 0.25-0.5 (0.667 [0.500, 0.833]) and states that
above gamma ~ 1 the layer is a brake. gamma = 1.0 is therefore on the brake side
of phi_cone's own optimum, which matters for the comparison against the cheap
proxy and the viability mask: those were never tuned either.

This script runs `real`, `cheap`, `mask` and their budget-matched variants at
gamma in {0.25, 0.5, 1.0} on the same 30 seeds, so each estimator can be read at
its own best gamma.

Usage: uv run python experiments/E5_ablation/scripts/gamma_check.py [--seeds 30]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fmcphi.envs.trapgrid import TrapGrid  # noqa: E402
from e5_lib import StepCounter, boot_ci, boot_diff_ci, run_episode_e5  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))

FUEL = 24.0
MAX_STEPS = 60
BASE = dict(M=10, alpha=1.0, beta=1.0, phi_m=4, phi_h=3, phi_every=1,
            budget_attr=None, budget_max=1.0)
SEED0 = 1000
GAMMAS = (0.25, 0.5, 1.0)
# name -> (mode, N)
ESTIMATORS = [
    ("real", "real", 32),
    ("cheap", "cheap", 32),
    ("cheap_matched", "cheap", 69),
    ("mask", "mask", 32),
    ("mask_matched", "mask", 416),
]


def _seed(*parts) -> int:
    """Deterministic bootstrap seed (str hash() is salted per process)."""
    return int(zlib.crc32("|".join(map(str, parts)).encode())) % (2 ** 31)


def run(mode, gamma, n_walkers, seeds):
    recs = []
    for s in seeds:
        env = StepCounter(TrapGrid(fuel=FUEL))
        r = run_episode_e5(env, env.reset(), max_steps=MAX_STEPS, seed=s,
                           mode=mode, gamma=gamma, N=n_walkers, **BASE)
        d = asdict(r)
        d["seed"] = s
        recs.append(d)
    return recs


def rates(recs, outcome):
    return [1.0 if r["outcome"] == outcome else 0.0 for r in recs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    args = ap.parse_args()
    seeds = [SEED0 + i for i in range(args.seeds)]

    t0 = time.time()
    cells = {}
    episodes = {}
    for name, mode, n_walkers in ESTIMATORS:
        for g in GAMMAS:
            recs = run(mode, g, n_walkers, seeds)
            episodes[(name, g)] = recs
            agg = {}
            for o in ("goal", "trap", "fuel", "timeout"):
                m, lo, hi = boot_ci(rates(recs, o), seed=_seed(name, g, o))
                agg[f"{o}_rate"] = dict(mean=m, lo=lo, hi=hi)
            for field in ("steps", "sim_steps", "actual_steps"):
                vals = [r[field] for r in recs]
                m, lo, hi = boot_ci(vals, seed=_seed(name, g, field))
                agg[field] = dict(mean=m, lo=lo, hi=hi)
            per_dec = [r["actual_steps"] / max(1, r["steps"]) for r in recs]
            agg["actual_steps_per_decision"] = dict(mean=float(np.mean(per_dec)))
            cells[f"{name}@{g}"] = dict(arm=name, mode=mode, N=n_walkers,
                                        gamma=g, n=len(recs), aggregate=agg)
            a = agg
            print(f"{name:>14} gamma={g:<5} goal={a['goal_rate']['mean']:.2f} "
                  f"[{a['goal_rate']['lo']:.2f},{a['goal_rate']['hi']:.2f}] "
                  f"trap={a['trap_rate']['mean']:.2f} fuel={a['fuel_rate']['mean']:.2f} "
                  f"act/dec={a['actual_steps_per_decision']['mean']:.0f}")

    # best gamma per estimator on goal rate, then each estimator at its own best
    best = {}
    for name, _mode, _n in ESTIMATORS:
        g_best = max(GAMMAS, key=lambda g: cells[f"{name}@{g}"]["aggregate"]["goal_rate"]["mean"])
        best[name] = g_best
    contrasts = {}
    real_best = episodes[("real", best["real"])]
    for name, _mode, _n in ESTIMATORS:
        if name == "real":
            continue
        arm_best = episodes[(name, best[name])]
        c = {}
        for o in ("goal", "trap"):
            d, lo, hi = boot_diff_ci(rates(real_best, o), rates(arm_best, o))
            c[f"{o}_rate_real_minus_arm"] = dict(diff=d, lo=lo, hi=hi)
        c["real_gamma"] = best["real"]
        c["arm_gamma"] = best[name]
        contrasts[name] = c

    out = dict(experiment="E5_ablation/gamma_check", gammas=list(GAMMAS),
               n_seeds=len(seeds), seeds=seeds,
               env=dict(kind="TrapGrid", fuel=FUEL, max_steps=MAX_STEPS),
               config=dict(**BASE), cells=cells,
               best_gamma_on_goal_rate=best,
               contrasts_at_own_best_gamma=contrasts,
               wall_seconds=time.time() - t0)
    with open(os.path.join(RESULTS, "E5_gamma_check.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("best gamma per estimator (goal rate):", best)
    for k, v in contrasts.items():
        g = v["goal_rate_real_minus_arm"]
        print(f"  real@{v['real_gamma']} minus {k}@{v['arm_gamma']}: "
              f"{g['diff']:+.2f} [{g['lo']:+.2f}, {g['hi']:+.2f}]")
    print(f"wall {out['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
