"""How many simulator calls does each arm actually make, versus how many it is charged?

`planner.plan` charges a flat `N * phi_m * phi_h` for the Phi evaluation, but
`phi_cone` returns 0 immediately on a non-viable state and breaks out of a
continuation as soon as it dies, so the true cost is lower. This script counts
every `env.step` call for real, so REPORT.md can state a measured ratio instead
of asserting that the flat charge was "verified".

    uv run python experiments/E4_no_goal/scripts/cost_audit_e4.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import AbsorbingTrapGrid, dump_json, rollout_fmc  # noqa: E402

from fmcphi.envs.e4_funnel_rooms import FunnelRooms  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))

SEEDS = 10
N, M, PHI_M, PHI_H = 32, 10, 4, 3
M_EQ = M * (1 + PHI_M * PHI_H)


def counting(cls):
    class Counting(cls):
        calls = 0

        def step(self, state, action):
            type(self).calls += 1
            return cls.step(self, state, action)

    Counting.__name__ = "Counting" + cls.__name__
    return Counting


def main():
    out = {}
    cases = {
        "trapgrid": (counting(AbsorbingTrapGrid), dict(fuel=24.0), 24.0),
        "funnel": (counting(FunnelRooms), dict(fuel=20.0), 20.0),
    }
    arms = {
        "a0g1": dict(N=N, M=M, alpha=0.0, beta=1.0, gamma=1.0,
                     phi_m=PHI_M, phi_h=PHI_H, phi_every=1, budget_attr="fuel"),
        "killdead_only": dict(N=N, M=M, alpha=0.0, beta=1.0, gamma=1e-12,
                              phi_m=PHI_M, phi_h=PHI_H, phi_every=1,
                              budget_attr="fuel"),
        "a0g0_eq": dict(N=N, M=M_EQ, alpha=0.0, beta=1.0, gamma=0.0),
    }
    for env_name, (cls, kwargs, fuel) in cases.items():
        for arm, kw in arms.items():
            kw = dict(kw)
            if "budget_attr" in kw:
                kw["budget_max"] = fuel
            env = cls(**kwargs)
            charged = 0
            real_steps = 0
            cls.calls = 0
            for s in range(SEEDS):
                r = rollout_fmc(env, arm, s, max_steps=60, **kw)
                charged += r.sim_steps
                real_steps += r.steps
            # rollout_fmc also calls env.step once per *real* step; that one is
            # the environment advancing, not deliberation, and is not charged.
            actual = cls.calls - real_steps
            ratio = actual / float(charged)
            out[f"{env_name}__{arm}"] = dict(
                seeds=SEEDS, real_steps=real_steps, charged=charged,
                actual_planning_calls=actual, actual_over_charged=round(ratio, 4),
                charged_per_real_step=round(charged / real_steps, 1),
                actual_per_real_step=round(actual / real_steps, 1),
            )
            print(f"{env_name:9s} {arm:14s} charged={charged:9d} "
                  f"actual={actual:9d}  actual/charged={ratio:.4f}  "
                  f"charged/real step={charged/real_steps:7.1f}  "
                  f"actual/real step={actual/real_steps:7.1f}")
    dump_json(os.path.join(RESULTS, "cost_audit.json"), out)


if __name__ == "__main__":
    main()
