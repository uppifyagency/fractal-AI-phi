"""E1 - gamma sweep on TrapGrid.

Runs the four sweeps of experiments/E1_trapgrid_sweep/BRIEF.md and writes one
JSON per arm into ../results/. Read-only with respect to the rest of the repo.

    uv run python experiments/E1_trapgrid_sweep/scripts/run_e1.py --all
    uv run python experiments/E1_trapgrid_sweep/scripts/run_e1.py --sweep 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from e1_lib import (check_parity, run_episode_instrumented, summarize,  # noqa: E402
                    write_json)
from fmcphi.envs.trapgrid import TrapGrid  # noqa: E402

# ---- fixed experimental constants (BRIEF section "Runs") -------------------
FUEL = 24.0
MAX_STEPS = 40
N_SEEDS_MAIN = 30
N_SEEDS_GEOM = 15
BASE = dict(N=32, M=10, alpha=1.0, beta=1.0, phi_m=4, phi_h=3, phi_every=1)


def run_arm(label, n_seeds, **plan_kwargs):
    env = TrapGrid(fuel=FUEL)
    x0 = env.reset()
    seeds = list(range(n_seeds))
    t0 = time.time()
    eps = [run_episode_instrumented(env, x0, max_steps=MAX_STEPS, seed=s,
                                    keep_trajectory=(s < 3), **plan_kwargs)
           for s in seeds]
    dt = time.time() - t0
    cfg = dict(plan_kwargs)
    cfg.update(fuel=FUEL, max_steps=MAX_STEPS)
    s = summarize(eps, seeds, cfg, label)
    s["wall_clock_s"] = dt
    print(f"  {label:<34s} n={len(eps):<3d} "
          f"goal={s['counts']['goal']:<3d} trap={s['counts']['trap']:<3d} "
          f"fuel={s['counts']['fuel']:<3d} timeout={s['counts']['timeout']:<3d} "
          f"steps={s['steps']['mean']:.1f} "
          f"sim/dec={s['sim_steps_per_decision']['mean']:.0f}  [{dt:.1f}s]")
    return s


def sweep1():
    print("Sweep 1 - gamma sweep, N=32 M=10, 30 seeds")
    out = []
    for g in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        s = run_arm(f"gamma={g}", N_SEEDS_MAIN, gamma=g, **BASE)
        write_json(f"sweep1_gamma_{g:g}.json", s)
        out.append(s)
    return out


def sweep2():
    """Equal simulator budget. gamma=1 costs M*N*(1 + phi_m*phi_h) per decision,
    i.e. 13x the gamma=0 cost at the same (N, M). Two ways to spend that 13x on
    a gamma=0 arm: more walkers, or more inner iterations."""
    print("Sweep 2 - equal simulator budget, 30 seeds")
    out = []
    kw = dict(BASE)
    kw.pop("N"); kw.pop("M")
    arms = [
        ("gamma=1 reference N=32 M=10", dict(N=32, M=10, gamma=1.0)),
        ("gamma=0 wide   N=416 M=10", dict(N=416, M=10, gamma=0.0)),
        ("gamma=0 deep   N=32  M=130", dict(N=32, M=130, gamma=0.0)),
        ("gamma=0 both   N=116 M=36", dict(N=116, M=36, gamma=0.0)),
    ]
    for label, a in arms:
        s = run_arm(label, N_SEEDS_MAIN, **a, **kw)
        write_json("sweep2_" + label.split()[0].replace("=", "") + "_"
                   + label.split()[1] + ".json", s)
        out.append(s)
    return out


def sweep3():
    print("Sweep 3 - phi_every amortisation at gamma=1, 30 seeds")
    out = []
    kw = dict(BASE)
    kw.pop("phi_every")
    for k in (1, 2, 3, 5):
        s = run_arm(f"phi_every={k}", N_SEEDS_MAIN, gamma=1.0, phi_every=k, **kw)
        write_json(f"sweep3_phi_every_{k}.json", s)
        out.append(s)
    return out


def sweep4():
    print("Sweep 4 - cone geometry at gamma=1, 15 seeds")
    out = []
    kw = dict(BASE)
    kw.pop("phi_m"); kw.pop("phi_h")
    for m in (2, 4, 8):
        for h in (1, 3, 5):
            s = run_arm(f"phi_m={m} phi_h={h}", N_SEEDS_GEOM,
                        gamma=1.0, phi_m=m, phi_h=h, **kw)
            write_json(f"sweep4_m{m}_h{h}.json", s)
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=int, action="append", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--parity", action="store_true")
    args = ap.parse_args()

    if args.parity or args.all:
        env = TrapGrid(fuel=FUEL)
        check_parity(env, env.reset(), seeds=range(8), gamma=1.0, **BASE)
        check_parity(env, env.reset(), seeds=range(8), gamma=0.0, **BASE)
        print("parity vs fmcphi.planner.run_episode: OK (8 seeds x 2 arms)")
        if args.parity and not args.all:
            return

    todo = [1, 2, 3, 4] if args.all else (args.sweep or [1])
    t0 = time.time()
    index = {}
    fns = {1: sweep1, 2: sweep2, 3: sweep3, 4: sweep4}
    for s in todo:
        index[f"sweep{s}"] = [
            {k: v for k, v in a.items() if k != "episodes"} for a in fns[s]()
        ]
    total = time.time() - t0
    print(f"total wall clock: {total/60:.2f} min")
    if args.all:
        index["wall_clock_min"] = total / 60
        write_json("index_all_sweeps.json", index)


if __name__ == "__main__":
    main()
