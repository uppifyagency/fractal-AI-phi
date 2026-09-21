"""E1 addendum - audits demanded by the adversarial review.

Every check here targets one claim in the first version of REPORT.md:

  A. arm / episode denominators in the "zero timeouts" sentence
  B. is a timeout reachable at all?  (stay costs 0.5 fuel, not 1.0)
  C. b_eff vs gamma at MATCHED states and matched seeds
  D. paired delta for "gamma=0 dies faster at N=416 than at N=32"
  E. Clopper-Pearson intervals for the degenerate 0/30 and 30/30 arms
  F. forensics of every trap death at gamma in {1, 2, 4}
  G. Phi map contrasts by row instead of pooled over open ground
  H. audit of sim_steps against the real number of env.step calls
  I. paired deltas + exact McNemar for the decomposition arms

Writes results/review_checks.json. Reads only, apart from that file.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from e1_lib import RESULTS_DIR, bootstrap_ci  # noqa: E402
from fmcphi.envs.trapgrid import State, TrapGrid  # noqa: E402
from fmcphi.planner import plan as lib_plan  # noqa: E402

BASE = dict(N=32, M=10, alpha=1.0, beta=1.0, phi_m=4, phi_h=3, phi_every=1)
OUT = {}


def load(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def paired(a, b, key="goal"):
    """b minus a on an outcome indicator, paired by seed index."""
    fa = np.array([1.0 if e["outcome"] == key else 0.0 for e in a["episodes"]])
    fb = np.array([1.0 if e["outcome"] == key else 0.0 for e in b["episodes"]])
    d = fb - fa
    m, lo, hi = bootstrap_ci(d)
    n01 = int(((fa == 0) & (fb == 1)).sum())
    n10 = int(((fa == 1) & (fb == 0)).sum())
    p = float(stats.binomtest(n01, n01 + n10, 0.5).pvalue) if (n01 + n10) else 1.0
    return {"delta": m, "lo": lo, "hi": hi, "n": len(d),
            "discordant_b_only": n01, "discordant_a_only": n10, "mcnemar_exact_p": p}


def cp(k, n):
    """Clopper-Pearson exact CI95 on a proportion."""
    lo = 0.0 if k == 0 else float(stats.beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(0.975, k + 1, n - k))
    return {"k": int(k), "n": int(n), "rate": k / n, "cp_lo": lo, "cp_hi": hi}


# --- A. denominators -------------------------------------------------------
sweep_files = sorted(f for f in glob.glob(os.path.join(RESULTS_DIR, "sweep*.json")))
arms = []
for f in sweep_files:
    d = load(os.path.basename(f))
    arms.append({"file": os.path.basename(f), "n": d["n"],
                 "outcomes": [e["outcome"] for e in d["episodes"]],
                 "steps": [e["steps"] for e in d["episodes"]],
                 "sim": [e["sim_steps"] for e in d["episodes"]]})
sig = {}
for a in arms:
    k = json.dumps([a["outcomes"], a["steps"], a["sim"]])
    sig.setdefault(k, []).append(a["file"])
dupes = {v[0]: v[1:] for v in sig.values() if len(v) > 1}
OUT["A_denominators"] = {
    "sweep_arm_files": len(arms),
    "sweep_episodes": sum(a["n"] for a in arms),
    "byte_identical_reruns": dupes,
    "distinct_arms": len(sig),
    "distinct_episodes": sum(load(v[0])["n"] for v in sig.values()),
    "timeouts_total": sum(a["outcomes"].count("timeout") for a in arms),
}

# --- B. is a timeout reachable? -------------------------------------------
env = TrapGrid(fuel=24.0)
s = env.reset()
alive_steps = 0
for _ in range(40):
    s = env.step(s, 4)  # 4 = stay
    if not env.viable(s):
        break
    alive_steps += 1
OUT["B_timeout_reachable"] = {
    "stay_cost": 0.5, "move_cost": 1.0, "fuel0": 24.0, "max_steps": 40,
    "steps_survived_when_always_staying": alive_steps,
    "fuel_left_after_40_stays": 24.0 - 40 * 0.5,
    "timeout_is_reachable": alive_steps >= 40,
}

# --- C. matched-state b_eff -----------------------------------------------
probe_states = {"start (0,0) fuel 24": State(0, 0, 24.0),
                "corridor (5,2) fuel 17": State(5, 2, 17.0),
                "pre-goal (11,1) fuel 12": State(11, 1, 12.0)}
matched = {}
for name, st in probe_states.items():
    row = {}
    for g in (0.0, 0.25, 1.0, 4.0):
        vals = [lib_plan(env, st, gamma=g, seed=k, **BASE).b_eff for k in range(30)]
        m, lo, hi = bootstrap_ci(vals)
        row[f"gamma={g:g}"] = {"mean": m, "lo": lo, "hi": hi, "n": len(vals)}
    matched[name] = row
OUT["C_matched_state_b_eff"] = matched

# --- D. "dies faster" ------------------------------------------------------
ref = load("sweep1_gamma_0.json")
wide = load("sweep2_gamma0_wide.json")
deep = load("sweep2_gamma0_deep.json")
both = load("sweep2_gamma0_both.json")
d_steps = {}
for nm, arm in (("wide N=416 M=10", wide), ("deep N=32 M=130", deep),
                ("both N=116 M=36", both)):
    a = np.array([e["steps"] for e in ref["episodes"]], dtype=float)
    b = np.array([e["steps"] for e in arm["episodes"]], dtype=float)
    m, lo, hi = bootstrap_ci(b - a)
    d_steps[nm] = {"delta_steps_vs_N32M10": m, "lo": lo, "hi": hi,
                   "mean_a": float(a.mean()), "mean_b": float(b.mean()),
                   "wilcoxon_p": float(stats.wilcoxon(b - a).pvalue)
                   if np.any(b - a) else 1.0}
OUT["D_dies_faster"] = d_steps

# --- E. Clopper-Pearson ----------------------------------------------------
cp_tab = {}
for f in ["sweep1_gamma_0.json", "sweep1_gamma_0.25.json", "sweep1_gamma_0.5.json",
          "sweep1_gamma_1.json", "sweep1_gamma_2.json", "sweep1_gamma_4.json",
          "sweep2_gamma0_wide.json", "sweep2_gamma0_deep.json", "sweep2_gamma0_both.json",
          "decomp_mask_only.json", "decomp_exponent_only.json", "decomp_neither.json",
          "decomp_g0_viable_base.json", "decomp_g0_viable_wide.json"]:
    d = load(f)
    n = d["n"]
    cp_tab[f] = {o: cp(sum(1 for e in d["episodes"] if e["outcome"] == o), n)
                 for o in ("goal", "trap", "fuel", "timeout")}
OUT["E_clopper_pearson"] = cp_tab

# --- F. trap-death forensics ----------------------------------------------
forensics = {}
for g in ("1", "2", "4"):
    d = load(f"sweep1_gamma_{g}.json")
    forensics[f"gamma={g}"] = [
        {"steps": e["steps"], "x": e["final_x"], "y": e["final_y"],
         "fuel": e["final_fuel"]}
        for e in d["episodes"] if e["outcome"] == "trap"]
OUT["F_trap_forensics"] = forensics

# --- G. Phi map by row -----------------------------------------------------
pm = load("phi_map.json")
grid = pm.get("phi_map") or pm.get("map") or pm
OUT["G_phi_map_rows"] = {"raw_keys": list(pm.keys())}
try:
    rows = grid if isinstance(grid, list) else grid.get("rows")
    arr = np.array(rows, dtype=float)  # index [y][x] or [row][col]
    OUT["G_phi_map_rows"]["shape"] = list(arr.shape)
except Exception as exc:  # pragma: no cover
    OUT["G_phi_map_rows"]["error"] = repr(exc)

# --- H. sim_steps audit ----------------------------------------------------
class CountingTrapGrid(TrapGrid):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.calls = 0

    def step(self, state, action):
        self.calls += 1
        return super().step(state, action)


cenv = CountingTrapGrid(fuel=24.0)
audit = {}
for g in (1.0, 0.0):
    cenv.calls = 0
    charged = 0
    for k in range(5):
        r = lib_plan(cenv, cenv.reset(), gamma=g, seed=k, **BASE)
        charged += r.sim_steps
    audit[f"gamma={g:g}"] = {"charged_sim_steps": charged,
                             "actual_env_step_calls": cenv.calls,
                             "charged_minus_actual": charged - cenv.calls,
                             "overcharge_frac": (charged - cenv.calls) / charged}
OUT["H_sim_steps_audit"] = audit

# --- I. decomposition deltas ----------------------------------------------
g0 = load("sweep1_gamma_0.json")
g1 = load("sweep1_gamma_1.json")
mask = load("decomp_mask_only.json")
mask9 = load("decomp_mask_only_1e9.json")
expo = load("decomp_exponent_only.json")
neither = load("decomp_neither.json")
vb = load("decomp_g0_viable_base.json")
vw = load("decomp_g0_viable_wide.json")
both_arm = load("decomp_both.json")

OUT["I_decomposition"] = {
    "both_reproduces_sweep1_gamma1": [e["outcome"] for e in both_arm["episodes"]]
    == [e["outcome"] for e in g1["episodes"]],
    "mask_only_matches_gamma_1e-9": [e["outcome"] for e in mask["episodes"]]
    == [e["outcome"] for e in mask9["episodes"]],
    "counts": {nm: dict(Counter(e["outcome"] for e in d["episodes"]))
               for nm, d in (("gamma=0", g0), ("both gamma=1", g1),
                             ("mask only", mask), ("exponent only", expo),
                             ("neither", neither),
                             ("g0+viable N=32 M=10", vb),
                             ("g0+viable N=416 M=10", vw))},
    "deltas_vs_gamma0": {
        nm: {o: paired(g0, d, o) for o in ("goal", "trap", "fuel")}
        for nm, d in (("both gamma=1", g1), ("mask only", mask),
                      ("exponent only", expo), ("neither", neither),
                      ("g0+viable N=32 M=10", vb), ("g0+viable N=416 M=10", vw))},
    "deltas_vs_both": {
        nm: {o: paired(g1, d, o) for o in ("goal", "trap", "fuel")}
        for nm, d in (("mask only", mask), ("exponent only", expo),
                      ("g0+viable N=32 M=10", vb), ("g0+viable N=416 M=10", vw))},
    "steps": {nm: bootstrap_ci([e["steps"] for e in d["episodes"]])
              for nm, d in (("gamma=0", g0), ("both gamma=1", g1), ("mask only", mask),
                            ("exponent only", expo), ("neither", neither),
                            ("g0+viable N=32 M=10", vb), ("g0+viable N=416 M=10", vw))},
}

# extra: P3 contrasts questioned by the review
OUT["I_decomposition"]["p3_trap_gamma2_vs_gamma1"] = paired(g1, load("sweep1_gamma_2.json"), "trap")
OUT["I_decomposition"]["p3_trap_gamma4_vs_gamma1"] = paired(g1, load("sweep1_gamma_4.json"), "trap")
OUT["I_decomposition"]["p2_goal_gamma2_vs_gamma0"] = paired(g0, load("sweep1_gamma_2.json"), "goal")

with open(os.path.join(RESULTS_DIR, "review_checks.json"), "w") as f:
    json.dump(OUT, f, indent=2)
print(json.dumps({k: v for k, v in OUT.items() if k in
                  ("A_denominators", "B_timeout_reachable", "D_dies_faster",
                   "H_sim_steps_audit", "G_phi_map_rows")}, indent=2))
print("\nC matched b_eff:")
print(json.dumps(OUT["C_matched_state_b_eff"], indent=2))
print("\nF trap forensics:")
print(json.dumps(OUT["F_trap_forensics"], indent=2))
