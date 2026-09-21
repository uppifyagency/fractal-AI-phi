#!/usr/bin/env python3
"""Why do components 2 and 3 add nothing to P2? Look at the actions chosen.

`fmcphi.planner.run_episode` reports outcomes, not the action trace, so this
script reruns the same episodes with its own outer loop around the *same*
`fmcphi.planner.plan`, recording the chosen strategy at every step. The loop is
checked against `run_episode` on the same seeds (`outcome_match`), so any
divergence would be visible rather than silent.

Writes results/p3_action_trace.json.
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

from fmcphi.planner import plan, run_episode        # noqa: E402
from repo_env import ACTION_NAMES, RepoEnv          # noqa: E402
from stats import rate_ci, summarize                # noqa: E402

E3 = HERE.parent
RESULTS = E3 / "results"

N, M, PHI_M, PHI_H = 24, 8, 3, 2
SEEDS = 100
TOKENS = 12.0   # well-posed: the M-step rollout is affordable within the budget

ARMS = {
    "gamma0_equal_budget": dict(gamma=0.0, N=N, M=M * (1 + PHI_M * PHI_H)),
    "gamma1_cone": dict(gamma=1.0, N=N, M=M, phi_m=PHI_M, phi_h=PHI_H),
    "gamma1_phi_repo": dict(gamma=1.0, N=N, M=M, phi_m=PHI_M, phi_h=PHI_H,
                            budget_attr="slack_irrev"),
    "gamma1_phi_repo_keycontent": dict(gamma=1.0, N=N, M=M, phi_m=PHI_M,
                                       phi_h=PHI_H, budget_attr="slack_irrev",
                                       _key_mode="content"),
    # Can component 3 be made to bite at all, or does the soft
    # relativize(.)^gamma composition swallow any penalty weight?
    "gamma1_phi_repo_lam2": dict(gamma=1.0, N=N, M=M, phi_m=PHI_M, phi_h=PHI_H,
                                 budget_attr="slack_irrev", _lam=2.0),
    "gamma1_phi_repo_lam5": dict(gamma=1.0, N=N, M=M, phi_m=PHI_M, phi_h=PHI_H,
                                 budget_attr="slack_irrev", _lam=5.0),
    "gamma1_phi_repo_lam20": dict(gamma=1.0, N=N, M=M, phi_m=PHI_M, phi_h=PHI_H,
                                  budget_attr="slack_irrev", _lam=20.0),
}


def traced_episode(env, seed, max_steps, **kw):
    """Same loop as fmcphi.planner.run_episode, with the action trace kept."""
    rng = np.random.default_rng(seed)
    s = env.reset()
    trace, nodes = [], [s.node]
    for step in range(max_steps):
        res = plan(env, s, rng=rng, alpha=1.0, beta=1.0, **kw)
        trace.append(int(res.action))
        s = env.step(s, res.action)
        nodes.append(s.node)
        if not env.viable(s):
            return env.death_cause(s) or "dead", trace, nodes, s
        if env.is_goal(s):
            return "goal", trace, nodes, s
    return "timeout", trace, nodes, s


def main():
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    max_steps = int(TOKENS * 2)
    out = {"experiment": "E3 / action trace", "n": SEEDS, "arms": {}}

    for label, kw in ARMS.items():
        env = RepoEnv(tokens=TOKENS, key_mode=kw.get("_key_mode", "full"),
                      lam=kw.get("_lam", 0.7))
        kw = {k: v for k, v in kw.items() if not k.startswith("_")}
        counts = Counter()
        first = Counter()
        outcomes = Counter()
        irrevs, mismatches = [], 0
        for s in range(SEEDS):
            oc, trace, nodes, final = traced_episode(env, s, max_steps, **kw)
            counts.update(trace)
            if trace:
                first[trace[0]] += 1
            outcomes[oc] += 1
            irrevs.append(final.irrev)
            ref = run_episode(env, env.reset(), max_steps=max_steps, seed=s,
                              alpha=1.0, beta=1.0, **kw)
            if ref.outcome != oc:
                mismatches += 1
        total = sum(counts.values())
        out["arms"][label] = {
            "config": kw,
            "outcomes": dict(outcomes),
            "outcome_match_vs_run_episode": {
                "mismatches": mismatches, "n": SEEDS},
            "action_share": {
                ACTION_NAMES[a]: counts[a] / total for a in sorted(counts)},
            "first_action_share": {
                ACTION_NAMES[a]: first[a] / SEEDS for a in sorted(first)},
            "irreversible_ops_committed": summarize(irrevs, seed=1),
            "delete_test_chosen_rate": rate_ci(
                sum(1 for a in counts.elements() if a == 4), total, seed=2),
        }
        print(f"{label}:")
        print(f"  outcomes {dict(outcomes)}  mismatches vs run_episode: {mismatches}")
        for name, share in out["arms"][label]["action_share"].items():
            print(f"    {name:22s} {share:.3f}")
        io = out["arms"][label]["irreversible_ops_committed"]
        print(f"    irrev ops committed {io['mean']:.3f} "
              f"[{io['ci95'][0]:.3f}, {io['ci95'][1]:.3f}] n={io['n']}")

    out["wall_clock_s"] = round(time.time() - t0, 2)
    (RESULTS / "p3_action_trace.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwall clock {out['wall_clock_s']}s -> {RESULTS/'p3_action_trace.json'}")


if __name__ == "__main__":
    main()
