"""Where does the no-goal agent actually go? Occupancy maps for the P3 story.

Reruns a0g1 (and a1g0 for contrast) with full trajectory logging and prints an
ASCII occupancy map plus per-region occupancy. A goal-blind agent that is not
leaking reward information should pile up in the option-rich region and never
visit the goal neighbourhood.
"""

from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import AbsorbingTrapGrid, dump_json, rollout_fmc  # noqa: E402


from fmcphi.envs.e4_funnel_rooms import FunnelRooms  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))

SEEDS = 20
N, M, PHI_M, PHI_H = 32, 10, 4, 3


def ascii_map(env, occ, total):
    lines = []
    for y in range(env.height - 1, -1, -1):
        row = []
        for x in range(env.width):
            if hasattr(env, "is_wall") and env.is_wall(x, y):
                row.append("#")
                continue
            hazard = (env.is_pit(x, y) if hasattr(env, "is_pit")
                      else env.is_trap(x, y))
            if hazard:
                row.append("X")
                continue
            if (x, y) == env.goal:
                row.append("G")
                continue
            f = occ.get((x, y), 0) / max(1, total)
            row.append(" .:-=+*#@"[min(8, int(f * 40))])
        lines.append(f"y={y}  " + " ".join(row))
    return "\n".join(lines)


def main():
    out = {}
    cases = {
        "trapgrid": (AbsorbingTrapGrid(fuel=24.0), 24.0),
        "funnel": (FunnelRooms(fuel=20.0), 20.0),
    }
    arms = {
        "a0g1": dict(N=N, M=M, alpha=0.0, beta=1.0, gamma=1.0,
                     phi_m=PHI_M, phi_h=PHI_H, budget_attr="fuel"),
        "a1g0": dict(N=N, M=M, alpha=1.0, beta=1.0, gamma=0.0),
    }
    for env_name, (env, fuel) in cases.items():
        for arm, kw in arms.items():
            kw = dict(kw)
            if "budget_attr" in kw:
                kw["budget_max"] = fuel
            occ = Counter()
            total = 0
            goal_visits = 0
            for s in range(SEEDS):
                r = rollout_fmc(env, arm, s, max_steps=60, **kw)
                for p in r.path:
                    occ[p] += 1
                    total += 1
                    if p == env.goal:
                        goal_visits += 1
            print(f"\n=== {env_name} / {arm} ({SEEDS} seeds, {total} state-visits, "
                  f"{goal_visits} visits to the goal cell) ===")
            print(ascii_map(env, occ, total))
            if hasattr(env, "region"):
                reg = Counter()
                for (x, y), c in occ.items():
                    reg[env.region(x, y)] += c
                share = {k: round(v / total, 4) for k, v in sorted(reg.items())}
                print("region occupancy share:", share)
            else:
                rows = Counter()
                for (x, y), c in occ.items():
                    rows[y] += c
                share = {f"y={k}": round(v / total, 4) for k, v in sorted(rows.items())}
                print("row occupancy share:", share)
            out[f"{env_name}__{arm}"] = {
                "seeds": SEEDS,
                "state_visits": total,
                "goal_cell_visits": goal_visits,
                "occupancy_share": share,
                "occupancy": {f"{x},{y}": c for (x, y), c in sorted(occ.items())},
            }
    dump_json(os.path.join(RESULTS, "behaviour.json"), out)


if __name__ == "__main__":
    main()
