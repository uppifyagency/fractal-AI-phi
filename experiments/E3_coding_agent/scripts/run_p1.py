#!/usr/bin/env python3
"""P1: is Phi lower at a state that precedes an irreversible action?

P1 as pre-registered: "Phi is strictly lower on a SHA immediately preceding an
irreversible action than on a comparable SHA without one."

Phi is a function of the state, so "preceding an irreversible action" has to be
turned into a property of the state. There are three non-equivalent ways to do
it, and E3 measures all three, because only one of them is a real test.

  A. STATIC / by construction.  The proposed tool call is classified and its
     weight enters component 3 directly. phi_repo is then lower at the state
     with the irreversible pending call by definition of the component. This
     arm exists to report the magnitude, not to test P1. It cannot fail.

  B. MECHANISTIC / the real test.  Component 3 is switched OFF (lam = 0) and
     irreversibility is left to act only through the dynamics: after a
     squash+force-push, revert no longer has anything to revert to, so an
     option is genuinely gone. Phi is the plain causal cone of SPEC section 2.
     The treatment state is one from which the option-destroying strategy is
     live; the control is the same state in an environment where a policy
     forbids that strategy, so it is a harmless no-op. Nothing forces this
     comparison to come out in P1's direction, and it is measured under both
     definitions of state identity (`key_mode`).

  C. POST-HOC.  Within the same environment, a state whose history has already
     been rewritten versus one whose history is intact. This is "after", not
     "preceding", and is reported for completeness.

Writes results/p1_irreversibility.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fmcphi.phi import phi_cone                                  # noqa: E402
from phi_repo import (GitProbe, classify_irreversibility,        # noqa: E402
                      phi_irreversibility, phi_repo_from_git)
from repo_env import RepoEnv                                     # noqa: E402
from stats import bootstrap_paired_diff, fmt, summarize          # noqa: E402

E3 = HERE.parent
RESULTS = E3 / "results"
FIXTURES = E3 / "fixtures"

REPS = 400          # replicate cone estimates per cell
PHI_M, PHI_H = 8, 4  # fan-out and depth of the cone estimate
# Nodes where both arms are alive and the comparison is meaningful.
LIVE_NODES = ["bug", "part1", "refactored", "fixed", "wrapped"]


# ---------------------------------------------------------------------------
# Arm A: static, on the real fixture repository
# ---------------------------------------------------------------------------

def arm_static() -> dict:
    nodes = json.loads((FIXTURES / "nodes.json").read_text())
    repo = Path(nodes["repo"])
    probe = GitProbe(repo)
    rows = []
    for name, rec in nodes["nodes"].items():
        branch = rec["branch"]
        treat = phi_repo_from_git(
            probe, rec["sha"],
            pending_command=f"git push --force origin {branch}",
            budgets={"tokens": (6.0, 6.0)},
            n_viable_actions=5.0, k_actions=6)
        ctrl = phi_repo_from_git(
            probe, rec["sha"],
            pending_command=f"git push origin {branch}",
            budgets={"tokens": (6.0, 6.0)},
            n_viable_actions=5.0, k_actions=6)
        rows.append({
            "node": name, "sha": rec["sha"][:10], "branch": branch,
            "published_real_git": treat["published"],
            "phi_irreversible_pending": treat["phi_repo"],
            "phi_reversible_pending": ctrl["phi_repo"],
            "ops": [o["name"] for o in treat["ops"]],
        })
    diff = bootstrap_paired_diff(
        [r["phi_irreversible_pending"] for r in rows],
        [r["phi_reversible_pending"] for r in rows], seed=11)
    return {"rows": rows, "paired_diff": diff,
            "note": "true by construction; reports magnitude, not evidence"}


def arm_static_commands() -> dict:
    """The classifier's own table, so the weights are auditable."""
    cmds = [
        "pytest -q", "git add -A && git commit -m wip", "git push origin feature",
        "git push --force-with-lease origin feature", "git push --force origin main",
        "git commit --amend --no-edit", "git rebase -i HEAD~3",
        "git reset --hard origin/main", "rm -rf src/legacy",
        "psql -c 'ALTER TABLE users DROP COLUMN email'", "npm publish",
        "terraform apply -auto-approve", "gh pr merge 42 --squash",
    ]
    out = []
    for c in cmds:
        for pub in (False, True):
            ops = classify_irreversibility(c, published=pub)
            out.append({"command": c, "published": pub,
                        "phi_irreversibility": phi_irreversibility(ops),
                        "ops": [o.name for o in ops]})
    return {"table": out}


# ---------------------------------------------------------------------------
# Arm B: mechanistic cone, component 3 off
# ---------------------------------------------------------------------------

def arm_mechanistic(key_mode: str) -> dict:
    treat_env = RepoEnv(allow_irreversible=True, key_mode=key_mode)
    ctrl_env = RepoEnv(allow_irreversible=False, key_mode=key_mode)
    per_node = {}
    all_t, all_c = [], []
    for node in LIVE_NODES:
        t_vals, c_vals = [], []
        for r in range(REPS):
            # Paired: the same seed drives both arms, so the only difference
            # is whether the cone can be contracted.
            rng_t = np.random.default_rng(1000 + r)
            rng_c = np.random.default_rng(1000 + r)
            t_vals.append(phi_cone(treat_env, treat_env.state(node),
                                   rng_t, m=PHI_M, h=PHI_H))
            c_vals.append(phi_cone(ctrl_env, ctrl_env.state(node),
                                   rng_c, m=PHI_M, h=PHI_H))
        per_node[node] = {
            "phi_irreversible_available": summarize(t_vals, seed=r),
            "phi_irreversible_forbidden": summarize(c_vals, seed=r + 1),
            "paired_diff": bootstrap_paired_diff(t_vals, c_vals, seed=7),
        }
        all_t += t_vals
        all_c += c_vals
    return {
        "key_mode": key_mode, "reps_per_node": REPS,
        "phi_m": PHI_M, "phi_h": PHI_H,
        "per_node": per_node,
        "pooled": {
            "phi_irreversible_available": summarize(all_t, seed=3),
            "phi_irreversible_forbidden": summarize(all_c, seed=4),
            "paired_diff": bootstrap_paired_diff(all_t, all_c, seed=5),
        },
    }


# ---------------------------------------------------------------------------
# Arm C: post-hoc, history already rewritten
# ---------------------------------------------------------------------------

def arm_posthoc(key_mode: str) -> dict:
    env = RepoEnv(allow_irreversible=True, key_mode=key_mode)
    per_node = {}
    all_p, all_i = [], []
    for node in LIVE_NODES + ["broken"]:
        p_vals, i_vals = [], []
        for r in range(REPS):
            rng_p = np.random.default_rng(2000 + r)
            rng_i = np.random.default_rng(2000 + r)
            p_vals.append(phi_cone(env, env.state(node, pushed=True),
                                   rng_p, m=PHI_M, h=PHI_H))
            i_vals.append(phi_cone(env, env.state(node, pushed=False),
                                   rng_i, m=PHI_M, h=PHI_H))
        per_node[node] = {
            "phi_history_rewritten": summarize(p_vals, seed=1),
            "phi_history_intact": summarize(i_vals, seed=2),
            "paired_diff": bootstrap_paired_diff(p_vals, i_vals, seed=6),
        }
        all_p += p_vals
        all_i += i_vals
    return {"key_mode": key_mode, "per_node": per_node,
            "pooled": {
                "phi_history_rewritten": summarize(all_p, seed=8),
                "phi_history_intact": summarize(all_i, seed=9),
                "paired_diff": bootstrap_paired_diff(all_p, all_i, seed=10),
            }}


def main():
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {
        "experiment": "E3 / P1 irreversibility",
        "arm_A_static_git": arm_static(),
        "arm_A_classifier_table": arm_static_commands(),
        "arm_B_mechanistic_key_full": arm_mechanistic("full"),
        "arm_B_mechanistic_key_content": arm_mechanistic("content"),
        "arm_C_posthoc_key_full": arm_posthoc("full"),
        "arm_C_posthoc_key_content": arm_posthoc("content"),
    }
    out["wall_clock_s"] = round(time.time() - t0, 2)
    (RESULTS / "p1_irreversibility.json").write_text(json.dumps(out, indent=2) + "\n")

    print("=== ARM A (static, by construction) ===")
    for r in out["arm_A_static_git"]["rows"]:
        print(f"  {r['node']:12s} {r['sha']} published={int(r['published_real_git'])} "
              f"phi(force-push pending)={r['phi_irreversible_pending']:.4f} "
              f"phi(plain push)={r['phi_reversible_pending']:.4f}")
    d = out["arm_A_static_git"]["paired_diff"]
    print(f"  paired diff {d['mean_diff']:+.4f} CI95 [{d['ci95'][0]:+.4f},"
          f" {d['ci95'][1]:+.4f}] n={d['n']}")

    for label in ("arm_B_mechanistic_key_full", "arm_B_mechanistic_key_content",
                  "arm_C_posthoc_key_full", "arm_C_posthoc_key_content"):
        blk = out[label]
        p = blk["pooled"]
        k = [v for v in p if v.startswith("phi_")]
        d = p["paired_diff"]
        print(f"\n=== {label} ===")
        print(f"  {k[0]}: {fmt(p[k[0]], 4)}   {k[1]}: {fmt(p[k[1]], 4)}")
        print(f"  paired diff {d['mean_diff']:+.4f} CI95 [{d['ci95'][0]:+.4f},"
              f" {d['ci95'][1]:+.4f}] n={d['n']} "
              f"significant={d['significant']}")
        for node, rec in blk["per_node"].items():
            dd = rec["paired_diff"]
            print(f"    {node:12s} diff {dd['mean_diff']:+.4f} "
                  f"CI95 [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]")
    print(f"\nwall clock {out['wall_clock_s']}s -> {RESULTS/'p1_irreversibility.json'}")


if __name__ == "__main__":
    main()
