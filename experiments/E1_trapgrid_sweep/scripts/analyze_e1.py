"""E1 analysis: markdown tables with n and bootstrap CI95, plus paired deltas.

Reads ../results/*.json written by run_e1.py and prints the tables that go into
REPORT.md. Also answers the inherited open question (does the agent stall next
to the option-poor goal corner?) from the recorded final positions.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1_lib import RESULTS_DIR, bootstrap_ci, write_json  # noqa: E402


def load(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def fmt(d, p=3):
    return f"{d['mean']:.{p}f} [{d['lo']:.{p}f}, {d['hi']:.{p}f}]"


def paired_delta(a_eps, b_eps, key_fn, n_boot=10000, seed=999):
    """CI95 on mean(b) - mean(a) resampling seeds jointly (arms share seeds)."""
    a = np.array([key_fn(e) for e in a_eps], dtype=np.float64)
    b = np.array([key_fn(e) for e in b_eps], dtype=np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = b - a
    point = float(d.mean())
    if np.all(d == d[0]):
        return point, point, point
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def outcome_table(arms, title):
    print(f"\n### {title}\n")
    print("| arm | n | sim_steps/decision | goal rate [CI95] | trap rate [CI95] "
          "| fuel rate [CI95] | timeout | steps [CI95] |")
    print("|---|---|---|---|---|---|---|---|")
    for a in arms:
        print(f"| {a['label']} | {a['n']} | {a['sim_steps_per_decision']['mean']:.0f} "
              f"| {fmt(a['rates']['goal'])} | {fmt(a['rates']['trap'])} "
              f"| {fmt(a['rates']['fuel'])} | {a['counts']['timeout']} "
              f"| {fmt(a['steps'], 1)} |")


def diag_table(arms, title):
    print(f"\n### {title}\n")
    print("| arm | n | mean Phi [CI95] | b_eff [CI95] | ESS [CI95] | "
          "final dist to goal [CI95] |")
    print("|---|---|---|---|---|---|")
    for a in arms:
        phi = fmt(a["mean_phi"]) if a["mean_phi"] else "n/a (gamma=0)"
        print(f"| {a['label']} | {a['n']} | {phi} | {fmt(a['mean_b_eff'])} "
              f"| {fmt(a['mean_ess'], 2)} | {fmt(a['final_goal_dist'], 2)} |")


def main():
    gammas = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    s1 = [load(f"sweep1_gamma_{g:g}.json") for g in gammas]
    s2 = [load(os.path.basename(p)) for p in sorted(glob.glob(os.path.join(RESULTS_DIR, "sweep2_*.json")))]
    s3 = [load(f"sweep3_phi_every_{k}.json") for k in (1, 2, 3, 5)]
    s4 = [load(f"sweep4_m{m}_h{h}.json") for m in (2, 4, 8) for h in (1, 3, 5)]

    outcome_table(s1, "Sweep 1 - gamma")
    diag_table(s1, "Sweep 1 - diagnostics")
    outcome_table(s2, "Sweep 2 - equal simulator budget")
    outcome_table(s3, "Sweep 3 - phi_every")
    outcome_table(s4, "Sweep 4 - cone geometry")

    # ---- paired deltas vs gamma = 0 ---------------------------------------
    base = s1[0]["episodes"]
    print("\n### Paired deltas vs gamma = 0 (same 30 seeds)\n")
    print("| arm | d trap rate [CI95] | d goal rate [CI95] | d fuel rate [CI95] "
          "| d steps [CI95] |")
    print("|---|---|---|---|---|")
    deltas = {}
    for a in s1[1:]:
        row = {}
        for o in ("trap", "goal", "fuel"):
            row[o] = paired_delta(base, a["episodes"], lambda e, o=o: 1.0 * (e["outcome"] == o))
        row["steps"] = paired_delta(base, a["episodes"], lambda e: e["steps"])
        deltas[a["label"]] = row
        print(f"| {a['label']} | " + " | ".join(
            f"{row[k][0]:+.3f} [{row[k][1]:+.3f}, {row[k][2]:+.3f}]"
            for k in ("trap", "goal", "fuel")) +
            f" | {row['steps'][0]:+.1f} [{row['steps'][1]:+.1f}, {row['steps'][2]:+.1f}] |")

    # equal-budget deltas: gamma=1 reference vs each gamma=0 control
    ref = [a for a in s2 if a["config"]["gamma"] == 1.0][0]
    print("\n### Paired deltas, gamma = 1 minus each equal-budget gamma = 0 control\n")
    print("| gamma=0 control | sim/dec | d trap [CI95] | d goal [CI95] |")
    print("|---|---|---|---|")
    eb = {}
    for a in s2:
        if a["config"]["gamma"] == 1.0:
            continue
        dt = paired_delta(a["episodes"], ref["episodes"], lambda e: 1.0 * (e["outcome"] == "trap"))
        dg = paired_delta(a["episodes"], ref["episodes"], lambda e: 1.0 * (e["outcome"] == "goal"))
        eb[a["label"]] = {"trap": dt, "goal": dg}
        print(f"| {a['label']} | {a['sim_steps_per_decision']['mean']:.0f} "
              f"| {dt[0]:+.3f} [{dt[1]:+.3f}, {dt[2]:+.3f}] "
              f"| {dg[0]:+.3f} [{dg[1]:+.3f}, {dg[2]:+.3f}] |")

    # ---- the inherited open question --------------------------------------
    print("\n### Open question: does the agent stall next to the option-poor goal?\n")
    print("| arm | timeouts | fuel deaths | fuel deaths with dist<=2 | "
          "final dist of non-goal episodes [CI95] | modal final cell |")
    print("|---|---|---|---|---|---|")
    hover = {}
    for a in s1[1:] + s3:
        eps = a["episodes"]
        fuel_eps = [e for e in eps if e["outcome"] == "fuel"]
        near = [e for e in fuel_eps if e["goal_dist"] <= 2]
        nongoal = [e["goal_dist"] for e in eps if e["outcome"] != "goal"]
        cells = collections.Counter((e["final_x"], e["final_y"]) for e in fuel_eps)
        modal = cells.most_common(1)[0] if cells else ("-", 0)
        ci = bootstrap_ci(nongoal) if nongoal else (float("nan"),) * 3
        hover[a["label"]] = {"n": a["n"], "timeouts": a["counts"]["timeout"],
                             "fuel": len(fuel_eps), "fuel_near_goal_le2": len(near),
                             "fuel_dists": [e["goal_dist"] for e in fuel_eps]}
        print(f"| {a['label']} | {a['counts']['timeout']} | {len(fuel_eps)} | {len(near)} "
              f"| {ci[0]:.2f} [{ci[1]:.2f}, {ci[2]:.2f}] | {modal[0]} x{modal[1]} |")

    write_json("analysis_deltas.json",
               {"paired_vs_gamma0": deltas, "equal_budget": eb, "hover": hover})


if __name__ == "__main__":
    main()
