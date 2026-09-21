"""E5 — ablations on the Phi layer.

Question: is the E1/E4 effect caused by Phi *as defined*, or by anything that
perturbs the virtual reward?

Sixteen arms, all on TrapGrid(fuel=24), alpha=beta=1, gamma=1.0 by default
(NOT inherited from E1; see gamma_check.py for the gamma-matched comparison).
Seeds are paired across arms.

Budget note (SPEC section 4, invariant 2): the Phi arms pay a *nominal*
M*N*(1+m*h) sim steps per decision, the gamma=0 arm pays M*N. That nominal
count is a worst case: phi_cone breaks its rollout early on death and returns 0
with zero steps on a non-viable state, and phi_viable_actions likewise. Every
arm is therefore run inside `StepCounter`, which measures the true number of
`env.step` calls, and both numbers are reported.

Arms `off_matched`, `mask_matched` (N x13) and `cheap_matched` (N x~2.17)
equalise the nominal per-decision budget; `cheap_true_matched` (N=60) equalises
the *measured* one.

Usage:  uv run python experiments/E5_ablation/scripts/run_e5.py [--seeds 30]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import zlib
import time
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fmcphi.envs.trapgrid import TrapGrid  # noqa: E402
from e5_lib import (  # noqa: E402
    StepCounter, boot_ci, boot_diff_ci, parity_check, run_episode_e5,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))

FUEL = 24.0
MAX_STEPS = 60
BASE = dict(M=10, alpha=1.0, beta=1.0, phi_m=4, phi_h=3, phi_every=1,
            budget_attr=None, budget_max=1.0)
GAMMA = 1.0          # fixed a priori, not inherited from E1
SEED0 = 1000

# name -> (mode, gamma, N, note)
ARMS = [
    ("off",           "off",      0.0, 32,  "canonical FMC, gamma=0, equal N"),
    ("off_matched",   "off",      0.0, 416, "canonical FMC at equal nominal budget (N x13)"),
    ("off_true_matched", "off",   0.0, 329, "canonical FMC at equal *measured* budget"),
    ("real",          "real",     1.0, 32,  "phi_composite, weight_by_survival=True"),
    ("shuffled",      "shuffled", 1.0, 32,  "arm 1: Phi permuted across walkers (mask destroyed too)"),
    ("constant",      "constant", 1.0, 32,  "arm 2: Phi replaced by swarm mean (mask destroyed too)"),
    ("nosurv",        "nosurv",   1.0, 32,  "arm 3: weight_by_survival=False"),
    ("cheap",         "cheap",    1.0, 32,  "arm 4: phi_viable_actions, equal N"),
    ("cheap_matched", "cheap",    1.0, 69,  "arm 4 at equal nominal budget (N x~2.17)"),
    ("cheap_true_matched", "cheap", 1.0, 60, "arm 4 at equal *measured* budget"),
    ("noise",         "noise",    1.0, 32,  "arm 5: uniform noise on the same support (mask destroyed too)"),
    ("shuffled_keepmask", "shuffled_keepmask", 1.0, 32,
     "arm A: Phi permuted in the VR, kill_dead on the true Phi (isolates the graded factor)"),
    ("noise_keepmask", "noise_keepmask", 1.0, 32,
     "arm A: Phi noised in the VR, kill_dead on the true Phi (isolates the graded factor)"),
    ("mask",          "mask",     1.0, 32,  "arm B: Phi = 1{viable}, zero rollouts (isolates the death mask)"),
    ("mask_matched",  "mask",     1.0, 416, "arm B at equal nominal budget (N x13)"),
    ("mask_true_matched", "mask", 1.0, 329, "arm B at equal *measured* budget (real spends ~3291 env.step/decision)"),
]

OUTCOMES = ("goal", "trap", "fuel", "timeout", "dead")


def _seed(*parts) -> int:
    """Deterministic bootstrap seed. `hash()` of a str is salted per process,
    so the previous version produced slightly different CI endpoints on every
    run; crc32 does not."""
    return int(zlib.crc32("|".join(map(str, parts)).encode())) % (2 ** 31)


def run_arm(name, mode, gamma, n_walkers, note, seeds):
    recs = []
    t0 = time.time()
    for s in seeds:
        env = StepCounter(TrapGrid(fuel=FUEL))
        r = run_episode_e5(env, env.reset(), max_steps=MAX_STEPS, seed=s,
                           mode=mode, gamma=gamma, N=n_walkers, **BASE)
        d = asdict(r)
        d["seed"] = s
        recs.append(d)
    wall = time.time() - t0

    agg = {}
    for o in OUTCOMES:
        ind = [1.0 if r["outcome"] == o else 0.0 for r in recs]
        m, lo, hi = boot_ci(ind, seed=_seed(name, o))
        agg[f"{o}_rate"] = dict(mean=m, lo=lo, hi=hi)
    for r in recs:
        # per-decision budgets, nominal and measured
        r["sim_steps_per_decision"] = r["sim_steps"] / max(1, r["steps"])
        r["actual_steps_per_decision"] = r["actual_steps"] / max(1, r["steps"])
        r["actual_over_nominal"] = (r["actual_steps"] / r["sim_steps"]
                                    if r["sim_steps"] else float("nan"))
    for field in ("steps", "sim_steps", "actual_steps", "mean_b_eff",
                  "mean_phi", "mean_ess", "total_reward",
                  "sim_steps_per_decision", "actual_steps_per_decision",
                  "actual_over_nominal"):
        vals = [r[field] for r in recs]
        vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        if not vals:
            agg[field] = dict(mean=float("nan"), lo=float("nan"), hi=float("nan"))
            continue
        m, lo, hi = boot_ci(vals, seed=_seed(name, field))
        agg[f"{field}"] = dict(mean=m, lo=lo, hi=hi)

    return dict(arm=name, mode=mode, gamma=gamma, N=n_walkers, note=note,
                n=len(recs), seeds=list(seeds), env=dict(kind="TrapGrid", fuel=FUEL,
                width=12, height=4, max_steps=MAX_STEPS),
                config=dict(**BASE), wall_seconds=wall,
                episodes=recs, aggregate=agg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--tag", type=str, default="",
                    help="filename suffix, for secondary runs at another n")
    args = ap.parse_args()
    seeds = [SEED0 + i for i in range(args.seeds)]
    os.makedirs(RESULTS, exist_ok=True)

    par = parity_check(lambda: TrapGrid(fuel=FUEL),
                       N=32, M=10, alpha=1.0, beta=1.0, phi_m=4, phi_h=3)
    print("parity vs fmcphi.planner.plan (all diagnostics + labels + RNG state):",
          {k: v["all_equal"] for k, v in par.items()})
    assert all(v["all_equal"] for v in par.values()), "mirrored loop drifted"

    runs = {}
    t0 = time.time()
    for name, mode, gamma, n_walkers, note in ARMS:
        if args.only and name != args.only:
            continue
        res = run_arm(name, mode, gamma, n_walkers, note, seeds)
        res["parity_check"] = par
        res["python"] = platform.python_version()
        path = os.path.join(RESULTS, f"E5_{name}{args.tag}.json")
        with open(path, "w") as f:
            json.dump(res, f, indent=2)
        runs[name] = res
        a = res["aggregate"]
        print(f"{name:>14}  n={res['n']:2d}  goal={a['goal_rate']['mean']:.2f} "
              f"trap={a['trap_rate']['mean']:.2f} fuel={a['fuel_rate']['mean']:.2f} "
              f"steps={a['steps']['mean']:5.1f} "
              f"nom/dec={a['sim_steps_per_decision']['mean']:7.0f} "
              f"act/dec={a['actual_steps_per_decision']['mean']:7.0f} "
              f"[{res['wall_seconds']:.1f}s]")

    if args.only:
        return

    # paired contrasts against the real arm
    contrasts = {}
    real = runs["real"]["episodes"]
    for name, res in runs.items():
        if name == "real":
            continue
        c = {}
        for o in ("goal", "trap", "fuel"):
            a = [1.0 if r["outcome"] == o else 0.0 for r in real]
            b = [1.0 if r["outcome"] == o else 0.0 for r in res["episodes"]]
            d, lo, hi = boot_diff_ci(a, b)
            c[f"{o}_rate_real_minus_arm"] = dict(diff=d, lo=lo, hi=hi)
        contrasts[name] = c

    summary = dict(
        experiment="E5_ablation",
        gamma=GAMMA,
        gamma_source="fixed a priori; see gamma_check.py for the gamma-matched run",
        n_seeds=len(seeds),
        env=dict(kind="TrapGrid", fuel=FUEL, max_steps=MAX_STEPS),
        config=dict(**BASE),
        parity_check=par,
        wall_seconds=time.time() - t0,
        arms={k: dict(mode=v["mode"], gamma=v["gamma"], N=v["N"], n=v["n"],
                      note=v["note"], aggregate=v["aggregate"]) for k, v in runs.items()},
        contrasts_vs_real=contrasts,
    )
    with open(os.path.join(RESULTS, f"E5_summary{args.tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"total wall {summary['wall_seconds']:.1f}s -> {RESULTS}/E5_summary{args.tag}.json")


if __name__ == "__main__":
    main()
