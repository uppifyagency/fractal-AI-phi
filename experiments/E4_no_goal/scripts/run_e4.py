"""E4 — competence with no goal. Run all arms on both maps, dump one JSON per run.

Usage
-----
    uv run python experiments/E4_no_goal/scripts/run_e4.py --seeds 50
    uv run python experiments/E4_no_goal/scripts/run_e4.py --selftest

Arms
----
    random        uniform random policy (floor)
    safe_random   uniform over actions that keep the next state viable
    a0g0          alpha=0 beta=1 gamma=0   canonical Common Sense, diversity only
    a0g1          alpha=0 beta=1 gamma=1   the E4 agent: no goal, Phi only
    a1g0          alpha=1 beta=1 gamma=0   canonical FMC, goal-seeking
    a1g1          alpha=1 beta=1 gamma=1   both (SPEC section 6 pilot config)
    a0g1_noslack  a0g1 with the fuel-slack factor of Phi switched off
    a0g0_eq       a0g0 at the *same simulator budget* as a0g1 (M x 13)
    a1g0_eq       a1g0 at the same simulator budget as a0g1 (M x 13, deeper)
    a1g0_eqN      a1g0 at the same simulator budget as a0g1 (N x 13, wider)
    killdead_only alpha=0 beta=1, Phi computed and `kill_dead` applied, but the
                  Phi *exponent* set to an inert epsilon so the graded
                  causal-cone weighting is numerically switched off

Why `killdead_only` exists
--------------------------
`planner.plan` gates two separate mechanisms behind the single test
`use_phi = gamma != 0 and phi_every > 0`: the graded factor
`relativize(Phi)^gamma`, and the hard zeroing `kill_dead(vr, phis)` of every
walker with Phi exactly 0. a0g1 minus a0g0 therefore confounds an entropy term
with a viability filter. This arm turns only the second one on: gamma is set to
GAMMA_INERT = 1e-12, so `p_hat ** gamma` differs from 1 by ~1e-11 (relativize is
bounded well inside [1e-3, 3] for N=32), while `kill_dead` stays fully live and
Phi is still evaluated, so the simulator cost is identical to a0g1's.

All arms run against an absorbing-death environment: `FunnelRooms.step` now
returns a non-viable state unchanged, and TrapGrid is wrapped by
`common.AbsorbingTrapGrid`. Without that, gamma=0 swarms plan through their own
death (see REPORT.md, "Review response").

Phi costs phi_m * phi_h = 12 extra simulator steps per walker per inner tick, so
a gamma=1 tick is *charged* 13x a gamma=0 tick. The `_eq` arms multiply M by 13,
which equalises the charged sim_steps per real step exactly. The charge is
nominal for the Phi arms: `phi_cone` breaks out of a continuation the moment it
dies, so the treatment is over-billed, never under-billed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (  # noqa: E402
    AbsorbingTrapGrid,
    dump_json,
    rollout_fmc,
    rollout_policy,
)

from fmcphi.envs.e4_funnel_rooms import FunnelRooms  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))

N = 32
M = 10
PHI_M = 4
PHI_H = 3
PHI_COST = 1 + PHI_M * PHI_H          # 13
M_EQ = M * PHI_COST                    # 130
MAX_STEPS = 60
GAMMA_INERT = 1e-12                    # kill_dead on, graded Phi factor off


def envs():
    return {
        "trapgrid": (AbsorbingTrapGrid(fuel=24.0), 24.0),
        "funnel": (FunnelRooms(fuel=20.0), 20.0),
    }


def arm_specs(fuel_max: float):
    base = dict(N=N, M=M, beta=1.0)
    phi = dict(phi_m=PHI_M, phi_h=PHI_H, phi_every=1,
               budget_attr="fuel", budget_max=fuel_max)
    return {
        "random": ("policy", dict(safe=False)),
        "safe_random": ("policy", dict(safe=True)),
        "a0g0": ("fmc", dict(base, alpha=0.0, gamma=0.0)),
        "a0g1": ("fmc", dict(base, alpha=0.0, gamma=1.0, **phi)),
        "a1g0": ("fmc", dict(base, alpha=1.0, gamma=0.0)),
        "a1g1": ("fmc", dict(base, alpha=1.0, gamma=1.0, **phi)),
        "a0g1_noslack": ("fmc", dict(base, alpha=0.0, gamma=1.0,
                                     phi_m=PHI_M, phi_h=PHI_H, phi_every=1)),
        "a0g0_eq": ("fmc", dict(base, alpha=0.0, gamma=0.0, M=M_EQ)),
        "a1g0_eq": ("fmc", dict(base, alpha=1.0, gamma=0.0, M=M_EQ)),
        "a1g0_eqN": ("fmc", dict(base, alpha=1.0, gamma=0.0, N=N * PHI_COST)),
        "killdead_only": ("fmc", dict(base, alpha=0.0, gamma=GAMMA_INERT, **phi)),
    }


def selftest():
    """rollout_fmc must reproduce fmcphi.planner.run_episode exactly."""
    from fmcphi.planner import run_episode
    env = AbsorbingTrapGrid(fuel=24.0)
    for seed in (0, 1, 2):
        for kw in (
            dict(N=16, M=6, alpha=1.0, beta=1.0, gamma=0.0),
            dict(N=16, M=6, alpha=0.0, beta=1.0, gamma=1.0,
                 phi_m=3, phi_h=2, budget_attr="fuel", budget_max=24.0),
        ):
            a = run_episode(env, env.reset(), max_steps=25, seed=seed, **kw)
            b = rollout_fmc(env, "x", seed, max_steps=25, **kw)
            assert a.outcome == b.outcome, (a, b)
            assert a.steps == b.steps and a.sim_steps == b.sim_steps, (a, b)
            assert abs(a.total_reward - b.total_reward) < 1e-9
    # null-test guard: Phi must not be constant on either map (SPEC section 5)
    from fmcphi.phi import phi_cone
    for name, (env, _f) in envs().items():
        rng = np.random.default_rng(0)
        cells = env.free_cells() if hasattr(env, "free_cells") else [
            (x, y) for x in range(env.width) for y in range(env.height)
            if not env.is_trap(x, y)
        ]
        s0 = env.reset()
        vals = []
        for (x, y) in cells:
            s = env.clone_state(s0)
            s.x, s.y = x, y
            vals.append(phi_cone(env, s, rng, m=8, h=3))
        sd = float(np.std(vals))
        assert sd > 0.2, f"{name}: Phi looks constant (sd={sd})"
        print(f"selftest: {name} Phi over free cells "
              f"mean={np.mean(vals):.3f} sd={sd:.3f} "
              f"min={np.min(vals):.3f} max={np.max(vals):.3f}")
    # death must be a fixed point of the dynamics on every map used here
    for name, (env, _f) in envs().items():
        s0 = env.reset()
        dead = env.clone_state(s0)
        if hasattr(env, "pits"):
            dead.x, dead.y = sorted(env.pits)[0]
        else:
            dead.x, dead.y = 1, env.trap_row
        assert not env.viable(dead), name
        for a in env.actions():
            nxt = env.step(env.clone_state(dead), a)
            assert (nxt.x, nxt.y, nxt.fuel) == (dead.x, dead.y, dead.fuel), (
                f"{name}: death is not absorbing under action {a}")
        starved = env.clone_state(s0)
        starved.fuel = 0.0
        assert not env.viable(starved), name
        for a in env.actions():
            nxt = env.step(env.clone_state(starved), a)
            assert nxt.fuel == 0.0, f"{name}: fuel death is not absorbing"
    print("selftest: death is absorbing on both maps (all actions, both causes)")

    # GAMMA_INERT must leave the graded Phi factor numerically at 1
    from fmcphi.core import relativize
    probe = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0] * 6, dtype=np.float64)
    p_hat = relativize(probe)
    dev = float(np.max(np.abs(p_hat ** GAMMA_INERT - 1.0)))
    assert dev < 1e-9, f"GAMMA_INERT is not inert: max |p^g - 1| = {dev}"
    print(f"selftest: GAMMA_INERT={GAMMA_INERT:g} leaves p_hat**gamma within "
          f"{dev:.2e} of 1")

    print("selftest: OK (rollout == run_episode, Phi non-constant on both maps)")


def run(env_name, seeds, baseline_seeds, only=None):
    env, fuel_max = envs()[env_name]
    specs = arm_specs(fuel_max)
    os.makedirs(RESULTS, exist_ok=True)
    for arm, (kind, kw) in specs.items():
        if only and arm not in only:
            continue
        n = baseline_seeds if kind == "policy" else seeds
        t0 = time.time()
        rows = []
        for s in range(n):
            if kind == "policy":
                r = rollout_policy(env, arm, s, max_steps=MAX_STEPS, **kw)
            else:
                r = rollout_fmc(env, arm, s, max_steps=MAX_STEPS, **kw)
            rows.append(r.as_dict(with_path=(s < 3)))
        dt = time.time() - t0
        out = {
            "env": env_name,
            "arm": arm,
            "kind": kind,
            "n": n,
            "max_steps": MAX_STEPS,
            "fuel": fuel_max,
            "config": {k: v for k, v in kw.items()},
            "wall_seconds": round(dt, 2),
            "episodes": rows,
        }
        path = os.path.join(RESULTS, f"{env_name}__{arm}.json")
        dump_json(path, out)
        outc = {}
        for r in rows:
            outc[r["outcome"]] = outc.get(r["outcome"], 0) + 1
        print(f"{env_name:9s} {arm:13s} n={n:4d} {dt:6.1f}s  "
              f"steps={np.mean([r['steps'] for r in rows]):5.1f}  {outc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--baseline-seeds", type=int, default=500)
    ap.add_argument("--env", default="both")
    ap.add_argument("--only", default=None,
                    help="comma-separated arm names")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    only = a.only.split(",") if a.only else None
    names = list(envs()) if a.env == "both" else [a.env]
    t0 = time.time()
    for nm in names:
        run(nm, a.seeds, a.baseline_seeds, only)
    print(f"total wall {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
