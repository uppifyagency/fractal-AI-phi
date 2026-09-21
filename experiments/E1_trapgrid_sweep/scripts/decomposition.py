"""E1 addendum - factorial decomposition of the Phi layer (review response).

The reviewer's blocking point: `fmcphi.planner.plan` enables TWO things at once
when `gamma != 0 and phi_every > 0`:

    (1) the exponent   VR *= relativize(Phi)^gamma
    (2) the hard mask  kill_dead(vr, phis)  -> VR = 0 where Phi == 0

Sweep 1 varied gamma, which switches both on together, so the report could not
attribute the effect to either. This script runs the 2x2 factorial at the same
N=32 M=10 phi_m=4 phi_h=3 configuration and the same 30 seeds, plus the
death-aware gamma=0 control that sweep 2 was missing.

`plan_variant` below is a line-by-line copy of `fmcphi.planner.plan` with two
extra boolean flags and nothing else changed. It consumes the rng in exactly
the same order, and `--parity` asserts that it reproduces the library planner
bit-for-bit on the two configurations that exist in both.

Nothing outside experiments/E1_trapgrid_sweep/ is written or modified.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from e1_lib import Episode, summarize, write_json  # noqa: E402
from fmcphi.core import (clone_step, decide, effective_branching_factor,  # noqa: E402
                         effective_sample_size, virtual_reward)
from fmcphi.envs.trapgrid import TrapGrid  # noqa: E402
from fmcphi.phi import kill_dead, phi_composite, virtual_reward_phi  # noqa: E402
from fmcphi.planner import PlanResult, plan as lib_plan  # noqa: E402

FUEL = 24.0
MAX_STEPS = 40
N_SEEDS = 30
BASE = dict(N=32, M=10, alpha=1.0, beta=1.0, phi_m=4, phi_h=3, phi_every=1)


# ---------------------------------------------------------------------------
# plan with the two factors separated
# ---------------------------------------------------------------------------

def plan_variant(
    env, x0, N=64, M=30, alpha=1.0, beta=1.0, gamma=0.0,
    phi_m=4, phi_h=3, phi_every=1, phi_weight_by_survival=True,
    budget_attr=None, budget_max=1.0,
    use_exponent: bool = True,     # apply relativize(Phi)^gamma
    use_phi_mask: bool = True,     # apply kill_dead(vr, phis)
    compute_phi: bool = True,      # run the phi_cone rollouts at all
    viability_mask: bool = False,  # zero VR for non-viable walkers (costs 0 sim steps)
    seed: Optional[int] = None, rng: Optional[np.random.Generator] = None,
) -> PlanResult:
    rng = rng if rng is not None else np.random.default_rng(seed)
    actions = list(env.actions())

    states = [env.clone_state(x0) for _ in range(N)]
    labels = np.array(
        [actions[rng.integers(0, len(actions))] for _ in range(N)], dtype=object)

    sim_steps = 0
    phis = np.ones(N, dtype=np.float64)
    mean_phi_acc: List[float] = []
    min_phi_acc: List[float] = []

    for t in range(M):
        for i in range(N):
            a = labels[i] if t == 0 else env.sample_action(states[i], rng)
            states[i] = env.step(states[i], a)
        sim_steps += N

        rewards = np.array([env.reward(s) for s in states], dtype=np.float64)
        obs = np.stack(
            [np.asarray(env.observe(s), dtype=np.float64).ravel() for s in states])

        partners_dist = rng.permutation(N)
        for i in range(N):
            if partners_dist[i] == i:
                partners_dist[i] = (i + 1) % N

        if compute_phi and phi_every > 0 and t % phi_every == 0:
            phis = np.array(
                [phi_composite(env, s, rng, m=phi_m, h=phi_h,
                               budget_attr=budget_attr, budget_max=budget_max,
                               weight_by_survival=phi_weight_by_survival)
                 for s in states], dtype=np.float64)
            sim_steps += N * phi_m * phi_h

        if compute_phi:
            g = gamma if use_exponent else 0.0
            vr = virtual_reward_phi(rewards, obs, partners_dist, phis,
                                    alpha=alpha, beta=beta, gamma=g)
            if use_phi_mask:
                vr = kill_dead(vr, phis)
            mean_phi_acc.append(float(phis.mean()))
            min_phi_acc.append(float(phis.min()))
        else:
            vr = virtual_reward(rewards, obs, partners_dist, alpha=alpha, beta=beta)

        if viability_mask:
            # Free by construction: env.viable is a predicate on the state the
            # walker is already in. No simulator call, no extra sim_steps.
            alive = np.array([1.0 if env.viable(s) else 0.0 for s in states])
            vr = np.asarray(vr, dtype=np.float64).copy()
            vr[alive == 0.0] = 0.0

        clone_idx = clone_step(vr, rng)
        states = [env.clone_state(states[k]) for k in clone_idx]
        labels = labels[clone_idx]

    dead = sum(1 for s in states if not env.viable(s))
    return PlanResult(
        action=decide(labels),
        b_eff=effective_branching_factor(labels),
        ess=effective_sample_size(vr),
        mean_phi=float(np.mean(mean_phi_acc)) if mean_phi_acc else float("nan"),
        min_phi=float(np.min(min_phi_acc)) if min_phi_acc else float("nan"),
        dead_walkers=dead,
        sim_steps=sim_steps,
        labels=labels.tolist(),
    )


def run_episode_variant(env, x0, max_steps=40, seed=None, **kw) -> Episode:
    rng = np.random.default_rng(seed)
    state = env.clone_state(x0)
    total_reward = 0.0
    sim_steps = 0
    per_decision = 0
    b_effs: List[float] = []
    phis: List[float] = []
    esss: List[float] = []
    deads: List[float] = []

    outcome, steps = "timeout", max_steps
    for step in range(max_steps):
        res = plan_variant(env, state, rng=rng, **kw)
        sim_steps += res.sim_steps
        per_decision = res.sim_steps
        b_effs.append(res.b_eff)
        esss.append(res.ess)
        deads.append(float(res.dead_walkers))
        if not np.isnan(res.mean_phi):
            phis.append(res.mean_phi)
        state = env.step(state, res.action)
        total_reward += env.reward(state)
        if not env.viable(state):
            outcome = env.death_cause(state) or "dead"
            steps = step + 1
            break
        if env.is_goal(state):
            outcome, steps = "goal", step + 1
            break

    gx, gy = env.goal
    return Episode(
        outcome=outcome, steps=steps, total_reward=total_reward,
        sim_steps=sim_steps, sim_steps_per_decision=per_decision,
        mean_b_eff=float(np.mean(b_effs)),
        mean_phi=float(np.mean(phis)) if phis else float("nan"),
        mean_ess=float(np.mean(esss)), mean_dead_walkers=float(np.mean(deads)),
        final_x=int(state.x), final_y=int(state.y), final_fuel=float(state.fuel),
        goal_dist=int(abs(state.x - gx) + abs(state.y - gy)), trajectory=[])


# ---------------------------------------------------------------------------

def parity() -> None:
    """plan_variant must reproduce fmcphi.planner.plan where they overlap."""
    env = TrapGrid(fuel=FUEL)
    x0 = env.reset()
    for seed in range(6):
        for g, kw in ((1.0, dict(compute_phi=True, use_exponent=True, use_phi_mask=True)),
                      (0.0, dict(compute_phi=False, use_exponent=False, use_phi_mask=False))):
            a = lib_plan(env, x0, gamma=g, seed=seed, **BASE)
            b = plan_variant(env, x0, gamma=g, seed=seed, **BASE, **kw)
            assert a.action == b.action and a.sim_steps == b.sim_steps, (seed, g, a, b)
            assert abs(a.b_eff - b.b_eff) < 1e-12 and abs(a.ess - b.ess) < 1e-12
    print("parity plan_variant vs fmcphi.planner.plan: OK (6 seeds x 2 configs)")


def run_arm(label, fname, **kw):
    env = TrapGrid(fuel=FUEL)
    x0 = env.reset()
    seeds = list(range(N_SEEDS))
    t0 = time.time()
    eps = [run_episode_variant(env, x0, max_steps=MAX_STEPS, seed=s, **kw) for s in seeds]
    dt = time.time() - t0
    cfg = {k: v for k, v in kw.items()}
    cfg.update(fuel=FUEL, max_steps=MAX_STEPS)
    s = summarize(eps, seeds, cfg, label)
    s["wall_clock_s"] = dt
    print(f"  {label:<44s} goal={s['counts']['goal']:<3d} trap={s['counts']['trap']:<3d} "
          f"fuel={s['counts']['fuel']:<3d} timeout={s['counts']['timeout']:<3d} "
          f"steps={s['steps']['mean']:5.1f} sim/dec={s['sim_steps_per_decision']['mean']:.0f} "
          f"[{dt:.1f}s]")
    write_json(fname, s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity-only", action="store_true")
    args = ap.parse_args()
    parity()
    if args.parity_only:
        return
    t0 = time.time()

    print("\nFactorial decomposition (N=32 M=10 phi_m=4 phi_h=3, 30 seeds, 4160 sim/dec)")
    run_arm("both: exponent gamma=1 + mask", "decomp_both.json",
            gamma=1.0, compute_phi=True, use_exponent=True, use_phi_mask=True, **BASE)
    run_arm("mask only: gamma exponent 0, kill_dead on", "decomp_mask_only.json",
            gamma=1.0, compute_phi=True, use_exponent=False, use_phi_mask=True, **BASE)
    run_arm("mask only (gamma=1e-9 via library plan)", "decomp_mask_only_1e9.json",
            gamma=1e-9, compute_phi=True, use_exponent=True, use_phi_mask=True, **BASE)
    run_arm("exponent only: gamma=1, kill_dead off", "decomp_exponent_only.json",
            gamma=1.0, compute_phi=True, use_exponent=True, use_phi_mask=False, **BASE)
    run_arm("neither: phi computed, unused", "decomp_neither.json",
            gamma=1.0, compute_phi=True, use_exponent=False, use_phi_mask=False, **BASE)

    print("\nDeath-aware gamma=0 controls (viability mask, 0 extra sim steps)")
    kw = dict(BASE)
    kw.pop("N"); kw.pop("M")
    run_arm("gamma=0 + viability mask  N=32 M=10", "decomp_g0_viable_base.json",
            N=32, M=10, gamma=0.0, compute_phi=False, use_exponent=False,
            use_phi_mask=False, viability_mask=True, **kw)
    run_arm("gamma=0 + viability mask  N=416 M=10", "decomp_g0_viable_wide.json",
            N=416, M=10, gamma=0.0, compute_phi=False, use_exponent=False,
            use_phi_mask=False, viability_mask=True, **kw)
    run_arm("gamma=0 plain             N=32 M=10 (repro)", "decomp_g0_plain_base.json",
            N=32, M=10, gamma=0.0, compute_phi=False, use_exponent=False,
            use_phi_mask=False, viability_mask=False, **kw)

    print(f"\ntotal wall clock: {(time.time()-t0)/60:.2f} min")


if __name__ == "__main__":
    main()
