#!/usr/bin/env python3
"""Review response re-runs for E3.

Three things the adversarial review asked for, none of which the original
scripts produced:

  R1. P1 arm B with `broken` included. `broken` satisfies run_p1.LIVE_NODES'
      stated criterion (viable in both the allow_irreversible=True and =False
      environments at pushed=False) and is the only node where the
      irreversible action actually removes viability. Reported as the full
      2x2: {5-node subset, 6-node set} x {key_mode full, content}.

  R2. Paired significance tests for the P2 arm comparisons that REPORT.md
      called "CIs disjoint" by eyeballing overlapping intervals. The arms share
      seeds 0..99, so the per-seed outcome indicator is paired.

  R3. Per-episode force-push share (force-pushes divided by episode length),
      paired on the seed, so the "6x more irreversible ops" claim can be
      separated from the episode-length confound.

Writes results/p5_review.json. Touches nothing else.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fmcphi.phi import phi_cone                         # noqa: E402
from fmcphi.planner import plan, run_episode            # noqa: E402
from repo_env import FORCE_PUSH, RepoEnv                # noqa: E402
from stats import bootstrap_paired_diff, summarize      # noqa: E402

E3 = HERE.parent
RESULTS = E3 / "results"

REPS = 400
PHI_M, PHI_H = 8, 4
FIVE = ["bug", "part1", "refactored", "fixed", "wrapped"]
SIX = FIVE + ["broken"]


# ---------------------------------------------------------------------------
# R1: P1 arm B, both node sets x both key modes
# ---------------------------------------------------------------------------

def arm_b(key_mode: str) -> dict:
    """Same estimator, same seeds, same pairing as run_p1.arm_mechanistic."""
    treat_env = RepoEnv(allow_irreversible=True, key_mode=key_mode)
    ctrl_env = RepoEnv(allow_irreversible=False, key_mode=key_mode)
    per_node, vals = {}, {}
    for node in SIX:
        t_vals, c_vals = [], []
        for r in range(REPS):
            rng_t = np.random.default_rng(1000 + r)
            rng_c = np.random.default_rng(1000 + r)
            t_vals.append(phi_cone(treat_env, treat_env.state(node),
                                   rng_t, m=PHI_M, h=PHI_H))
            c_vals.append(phi_cone(ctrl_env, ctrl_env.state(node),
                                   rng_c, m=PHI_M, h=PHI_H))
        vals[node] = (t_vals, c_vals)
        per_node[node] = {
            "viable_treat": treat_env.viable(treat_env.state(node)),
            "viable_ctrl": ctrl_env.viable(ctrl_env.state(node)),
            "phi_irreversible_available": summarize(t_vals, seed=r),
            "phi_irreversible_forbidden": summarize(c_vals, seed=r + 1),
            "paired_diff": bootstrap_paired_diff(t_vals, c_vals, seed=7),
        }
    out = {"key_mode": key_mode, "reps_per_node": REPS,
           "phi_m": PHI_M, "phi_h": PHI_H, "per_node": per_node}
    for name, nodes in (("five_node_subset", FIVE), ("six_node_full", SIX)):
        t = [v for n in nodes for v in vals[n][0]]
        c = [v for n in nodes for v in vals[n][1]]
        # Pooled over node x rep, as the original report did.
        pooled = bootstrap_paired_diff(t, c, seed=5)
        # Node-level: the node mean diff is the unit, which is what an
        # unweighted mean over 5 or 6 nodes actually estimates.
        node_means = [float(np.mean(np.asarray(vals[n][0]) - np.asarray(vals[n][1])))
                      for n in nodes]
        out[name] = {
            "nodes": nodes,
            "phi_irreversible_available": summarize(t, seed=3),
            "phi_irreversible_forbidden": summarize(c, seed=4),
            "paired_diff_pooled_node_x_rep": pooled,
            "paired_diff_node_level": summarize(node_means, seed=12),
            "node_mean_diffs": dict(zip(nodes, node_means)),
        }
    return out


# ---------------------------------------------------------------------------
# R2: P2 arms with per-seed outcomes, paired on the seed
# ---------------------------------------------------------------------------

N, M, P2_PHI_M, P2_PHI_H = 24, 8, 3, 2
TOKENS = 12.0
SEEDS = 100

P2_ARMS = {
    "gamma1_cone": dict(gamma=1.0, N=N, M=M, phi_m=P2_PHI_M, phi_h=P2_PHI_H),
    "gamma1_cone_slack": dict(gamma=1.0, N=N, M=M, phi_m=P2_PHI_M,
                              phi_h=P2_PHI_H, budget_attr="slack"),
    "gamma1_phi_repo": dict(gamma=1.0, N=N, M=M, phi_m=P2_PHI_M,
                            phi_h=P2_PHI_H, budget_attr="slack_irrev"),
}


def p2_per_seed(kwargs: dict) -> dict:
    env = RepoEnv(tokens=TOKENS, key_mode="full")
    max_steps = int(TOKENS * 2)
    goal, trap, budget, steps = [], [], [], []
    for s in range(SEEDS):
        res = run_episode(env, env.reset(), max_steps=max_steps, seed=s,
                          alpha=1.0, beta=1.0, **kwargs)
        goal.append(1.0 if res.outcome == "goal" else 0.0)
        trap.append(1.0 if res.outcome == "trap" else 0.0)
        budget.append(1.0 if res.outcome == "budget" else 0.0)
        steps.append(float(res.steps))
    return {"goal": goal, "trap": trap, "budget": budget, "steps": steps}


# ---------------------------------------------------------------------------
# R3: per-episode force-push share, paired on the seed
# ---------------------------------------------------------------------------

DIAG_ARMS = {
    "gamma0_equal_budget": dict(gamma=0.0, N=N, M=M * (1 + P2_PHI_M * P2_PHI_H)),
    "gamma1_cone": dict(gamma=1.0, N=N, M=M, phi_m=P2_PHI_M, phi_h=P2_PHI_H),
}


def traced(env, seed, max_steps, **kw):
    rng = np.random.default_rng(seed)
    s = env.reset()
    trace = []
    for _ in range(max_steps):
        res = plan(env, s, rng=rng, alpha=1.0, beta=1.0, **kw)
        trace.append(int(res.action))
        s = env.step(s, res.action)
        if not env.viable(s):
            return env.death_cause(s) or "dead", trace, s
        if env.is_goal(s):
            return "goal", trace, s
    return "timeout", trace, s


def diag_per_seed(kwargs: dict) -> dict:
    env = RepoEnv(tokens=TOKENS, key_mode="full")
    max_steps = int(TOKENS * 2)
    irrev, share, nsteps, fp = [], [], [], []
    for s in range(SEEDS):
        _oc, trace, final = traced(env, s, max_steps, **kwargs)
        n = max(1, len(trace))
        k = sum(1 for a in trace if a == FORCE_PUSH)
        irrev.append(float(final.irrev))
        fp.append(float(k))
        nsteps.append(float(len(trace)))
        share.append(k / n)
    return {"irrev": irrev, "force_push_count": fp,
            "force_push_share": share, "steps": nsteps}


def main():
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {"experiment": "E3 / review response re-runs"}

    print("=== R1: P1 arm B, 2x2 (node set x key mode) ===")
    r1 = {}
    for km in ("full", "content"):
        r1[km] = arm_b(km)
        for setname in ("five_node_subset", "six_node_full"):
            d = r1[km][setname]["paired_diff_pooled_node_x_rep"]
            nl = r1[km][setname]["paired_diff_node_level"]
            print(f"  key={km:8s} {setname:18s} pooled {d['mean_diff']:+.4f} "
                  f"[{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}] n={d['n']}   "
                  f"node-level {nl['mean']:+.4f} "
                  f"[{nl['ci95'][0]:+.4f},{nl['ci95'][1]:+.4f}] n={nl['n']}")
        for node, rec in r1[km]["per_node"].items():
            dd = rec["paired_diff"]
            print(f"      {node:12s} {dd['mean_diff']:+.4f} "
                  f"[{dd['ci95'][0]:+.4f},{dd['ci95'][1]:+.4f}]")
    out["R1_p1_armB_nodeset_x_keymode"] = r1

    print("\n=== R2: P2 paired tests (seeds 0..99 shared) ===")
    p2 = {lab: p2_per_seed(kw) for lab, kw in P2_ARMS.items()}
    r2 = {}
    for a, b in (("gamma1_cone", "gamma1_cone_slack"),
                 ("gamma1_cone", "gamma1_phi_repo")):
        for metric in ("goal", "trap", "steps"):
            d = bootstrap_paired_diff(p2[a][metric], p2[b][metric], seed=21)
            key = f"{a}_minus_{b}/{metric}"
            r2[key] = d
            print(f"  {key:46s} {d['mean_diff']:+.4f} "
                  f"[{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}] "
                  f"n={d['n']} sig={d['significant']}")
    r2["marginals"] = {lab: {m: summarize(v[m], seed=1) for m in v}
                       for lab, v in p2.items()}
    out["R2_p2_paired_tests"] = r2

    print("\n=== R3: force-push share per episode, paired on seed ===")
    dg = {lab: diag_per_seed(kw) for lab, kw in DIAG_ARMS.items()}
    r3 = {"marginals": {lab: {m: summarize(v[m], seed=1) for m in v}
                        for lab, v in dg.items()}}
    for metric in ("irrev", "force_push_share", "force_push_count", "steps"):
        d = bootstrap_paired_diff(dg["gamma1_cone"][metric],
                                  dg["gamma0_equal_budget"][metric], seed=31)
        r3[f"gamma1_cone_minus_gamma0/{metric}"] = d
        m0 = r3["marginals"]["gamma0_equal_budget"][metric]["mean"]
        m1 = r3["marginals"]["gamma1_cone"][metric]["mean"]
        print(f"  {metric:18s} gamma0 {m0:.4f} -> gamma1 {m1:.4f}  "
              f"paired {d['mean_diff']:+.4f} "
              f"[{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}] "
              f"ratio {(m1/m0 if m0 else float('nan')):.2f}x sig={d['significant']}")
    out["R3_force_push_share"] = r3

    out["wall_clock_s"] = round(time.time() - t0, 2)
    (RESULTS / "p5_review.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwall clock {out['wall_clock_s']}s -> {RESULTS/'p5_review.json'}")


if __name__ == "__main__":
    main()
