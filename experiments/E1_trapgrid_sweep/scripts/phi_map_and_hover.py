"""E1 addendum.

(a) Phi map over TrapGrid, to check the null-test condition of docs/SPEC.md
    section 5 (Phi must not be constant) and to quantify how option-poor the
    goal corner is relative to open ground.
(b) Where non-goal episodes actually die: the "hover next to the goal" question.
(c) Paired within-gamma contrasts needed for prediction P3 (interior optimum).
"""

from __future__ import annotations

import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_e1 import load, paired_delta  # noqa: E402
from e1_lib import bootstrap_ci, write_json  # noqa: E402
from fmcphi.envs.trapgrid import TrapGrid, State  # noqa: E402
from fmcphi.phi import phi_cone  # noqa: E402

REPS = 400
M, H = 4, 3


def phi_map(fuel=12.0):
    env = TrapGrid(fuel=24.0)
    rng = np.random.default_rng(7)
    grid = {}
    for y in range(env.height):
        for x in range(env.width):
            s = State(x=x, y=y, fuel=fuel)
            vals = [phi_cone(env, s, rng, m=M, h=H) for _ in range(REPS)]
            m, lo, hi = bootstrap_ci(vals)
            grid[f"{x},{y}"] = {"mean": m, "lo": lo, "hi": hi,
                                "trap": env.is_trap(x, y),
                                "goal": (x, y) == env.goal}
    return env, grid


def main():
    env, grid = phi_map()
    print(f"### Phi map, phi_m={M} phi_h={H}, fuel=12, {REPS} rng draws per cell\n")
    print("Phi in [0, 4]. Rows are y, columns are x. 'T' = trap (Phi = 0 by construction).\n")
    header = "| y | " + " | ".join(str(x) for x in range(env.width)) + " |"
    print(header)
    print("|" + "---|" * (env.width + 1))
    for y in reversed(range(env.height)):
        cells = []
        for x in range(env.width):
            g = grid[f"{x},{y}"]
            cells.append("T" if g["trap"] else f"{g['mean']:.2f}")
        print(f"| {y} | " + " | ".join(cells) + " |")

    open_cells = [grid[f"{x},{y}"]["mean"] for x in range(2, env.width - 2)
                  for y in (1, 2)]
    goal = grid[f"{env.goal[0]},{env.goal[1]}"]["mean"]
    above = grid[f"{env.goal[0]},1"]["mean"]
    start = grid["0,0"]["mean"]
    print(f"\nopen ground (x in 2..9, y in 1..2), mean Phi = {np.mean(open_cells):.3f}")
    print(f"goal cell (11,0)            Phi = {goal:.3f}")
    print(f"cell above the goal (11,1)  Phi = {above:.3f}")
    print(f"start cell (0,0)            Phi = {start:.3f}")
    print(f"ratio open / goal-cell      = {np.mean(open_cells)/max(goal,1e-9):.2f}x")

    # ---- (b) where the non-goal episodes end ------------------------------
    print("\n### Where non-goal episodes end (sweep 1)\n")
    print("| arm | non-goal n | in goal column x=11, y>0 | dist<=2 | dist 3-5 | dist>5 |")
    print("|---|---|---|---|---|---|")
    hover = {}
    for g in (0.25, 0.5, 1.0, 2.0, 4.0):
        a = load(f"sweep1_gamma_{g:g}.json")
        ng = [e for e in a["episodes"] if e["outcome"] != "goal"]
        col = sum(1 for e in ng if e["final_x"] == 11 and e["final_y"] > 0)
        d2 = sum(1 for e in ng if e["goal_dist"] <= 2)
        d5 = sum(1 for e in ng if 3 <= e["goal_dist"] <= 5)
        dm = sum(1 for e in ng if e["goal_dist"] > 5)
        hover[f"gamma={g}"] = {"non_goal": len(ng), "goal_column": col,
                               "le2": d2, "d3_5": d5, "gt5": dm,
                               "cells": {f"{e['final_x']},{e['final_y']}": 0 for e in ng}}
        cnt = collections.Counter((e["final_x"], e["final_y"]) for e in ng)
        hover[f"gamma={g}"]["cells"] = {f"{k[0]},{k[1]}": v for k, v in cnt.items()}
        print(f"| gamma={g} | {len(ng)} | {col} | {d2} | {d5} | {dm} |")

    # ---- (c) P3 contrasts --------------------------------------------------
    print("\n### P3 contrasts: goal rate, paired over the same 30 seeds\n")
    print("| contrast | d goal rate [CI95] |")
    print("|---|---|")
    arms = {g: load(f"sweep1_gamma_{g:g}.json")["episodes"]
            for g in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)}
    contrasts = [(0.5, 2.0), (0.5, 4.0), (0.25, 2.0), (0.25, 4.0), (1.0, 2.0),
                 (0.25, 0.5), (0.5, 1.0), (0.0, 4.0), (0.0, 2.0)]
    p3 = {}
    for a, b in contrasts:
        d = paired_delta(arms[a], arms[b], lambda e: 1.0 * (e["outcome"] == "goal"))
        p3[f"gamma{b}_minus_gamma{a}"] = d
        print(f"| gamma={b} minus gamma={a} | {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] |")

    # trap rate contrast for the non-monotone trap curve
    print("\n| contrast | d trap rate [CI95] |")
    print("|---|---|")
    for a, b in [(1.0, 2.0), (1.0, 4.0), (0.5, 1.0)]:
        d = paired_delta(arms[a], arms[b], lambda e: 1.0 * (e["outcome"] == "trap"))
        p3[f"trap_gamma{b}_minus_gamma{a}"] = d
        print(f"| gamma={b} minus gamma={a} | {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] |")

    write_json("phi_map.json", {"phi_m": M, "phi_h": H, "reps": REPS,
                                "fuel_used": 12.0, "grid": grid})
    write_json("hover_and_p3.json", {"where_non_goal_ends": hover, "p3_contrasts": p3})


if __name__ == "__main__":
    main()
