"""E1 helpers: an instrumented episode runner and bootstrap statistics.

The runner is a line-by-line copy of `fmcphi.planner.run_episode` with extra
recording only (final position, per-step diagnostics). It consumes the rng in
exactly the same order, so for a given seed and kwargs it must return the same
outcome/steps/sim_steps as the library function. `check_parity` asserts that.
Nothing outside experiments/E1_trapgrid_sweep/ is modified.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from fmcphi.planner import plan, run_episode

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


# ---------------------------------------------------------------------------
# instrumented episode
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    outcome: str
    steps: int
    total_reward: float
    sim_steps: int            # total simulator steps over the whole episode
    sim_steps_per_decision: int
    mean_b_eff: float
    mean_phi: float
    mean_ess: float
    mean_dead_walkers: float
    final_x: int
    final_y: int
    final_fuel: float
    goal_dist: int            # Manhattan distance from final cell to the goal
    trajectory: List[List[int]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def run_episode_instrumented(env, x0, max_steps: int = 40, seed: int | None = None,
                             keep_trajectory: bool = False, **plan_kwargs) -> Episode:
    rng = np.random.default_rng(seed)
    state = env.clone_state(x0)
    total_reward = 0.0
    sim_steps = 0
    per_decision = 0
    b_effs: List[float] = []
    phis: List[float] = []
    esss: List[float] = []
    deads: List[float] = []
    traj: List[List[int]] = [[state.x, state.y]]

    outcome = "timeout"
    steps = max_steps
    for step in range(max_steps):
        res = plan(env, state, rng=rng, **plan_kwargs)
        sim_steps += res.sim_steps
        per_decision = res.sim_steps
        b_effs.append(res.b_eff)
        esss.append(res.ess)
        deads.append(float(res.dead_walkers))
        if not np.isnan(res.mean_phi):
            phis.append(res.mean_phi)

        state = env.step(state, res.action)
        total_reward += env.reward(state)
        traj.append([state.x, state.y])

        if not env.viable(state):
            outcome = getattr(env, "death_cause", lambda _s: "dead")(state) or "dead"
            steps = step + 1
            break
        if env.is_goal(state):
            outcome = "goal"
            steps = step + 1
            break

    gx, gy = env.goal
    return Episode(
        outcome=outcome,
        steps=steps,
        total_reward=total_reward,
        sim_steps=sim_steps,
        sim_steps_per_decision=per_decision,
        mean_b_eff=float(np.mean(b_effs)),
        mean_phi=float(np.mean(phis)) if phis else float("nan"),
        mean_ess=float(np.mean(esss)),
        mean_dead_walkers=float(np.mean(deads)),
        final_x=int(state.x),
        final_y=int(state.y),
        final_fuel=float(state.fuel),
        goal_dist=int(abs(state.x - gx) + abs(state.y - gy)),
        trajectory=traj if keep_trajectory else [],
    )


def check_parity(env, x0, seeds=range(8), **plan_kwargs) -> None:
    """The instrumented runner must agree with fmcphi.planner.run_episode."""
    for s in seeds:
        a = run_episode(env, x0, max_steps=40, seed=s, **plan_kwargs)
        b = run_episode_instrumented(env, x0, max_steps=40, seed=s, **plan_kwargs)
        assert (a.outcome, a.steps, a.sim_steps) == (b.outcome, b.steps, b.sim_steps), (
            f"parity broken at seed {s}: {a} vs {b}")
        assert abs(a.total_reward - b.total_reward) < 1e-9
        assert abs(a.mean_b_eff - b.mean_b_eff) < 1e-9


# ---------------------------------------------------------------------------
# bootstrap statistics
# ---------------------------------------------------------------------------

def bootstrap_ci(values, n_boot: int = 10000, seed: int = 12345, stat=np.mean):
    """Percentile bootstrap CI95 of `stat` over `values`. Returns (mean, lo, hi)."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(stat(v))
    if len(v) == 1 or np.all(v == v[0]):
        return (point, point, point)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = stat(v[idx], axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (point, float(lo), float(hi))


def rate_ci(flags, n_boot: int = 10000, seed: int = 12345):
    """CI95 on a proportion, by the same bootstrap."""
    return bootstrap_ci(np.asarray(flags, dtype=np.float64), n_boot=n_boot, seed=seed)


def summarize(episodes: List[Episode], seeds: List[int], config: Dict[str, Any],
              label: str) -> Dict[str, Any]:
    out = {o: [1.0 if e.outcome == o else 0.0 for e in episodes]
           for o in ("goal", "trap", "fuel", "timeout")}
    steps = [e.steps for e in episodes]
    sim_tot = [e.sim_steps for e in episodes]
    per_dec = [e.sim_steps_per_decision for e in episodes]
    b_eff = [e.mean_b_eff for e in episodes]
    ess = [e.mean_ess for e in episodes]
    phi = [e.mean_phi for e in episodes if not np.isnan(e.mean_phi)]
    gd = [e.goal_dist for e in episodes]
    # "hover": episode ended in timeout within 2 cells of the goal
    hover = [1.0 if (e.outcome == "timeout" and e.goal_dist <= 2) else 0.0 for e in episodes]
    hover1 = [1.0 if (e.outcome == "timeout" and e.goal_dist <= 1) else 0.0 for e in episodes]

    def c(x, stat=np.mean):
        m, lo, hi = bootstrap_ci(x, stat=stat)
        return {"mean": m, "lo": lo, "hi": hi, "n": len(x)}

    return {
        "label": label,
        "config": config,
        "n": len(episodes),
        "seeds": seeds,
        "rates": {k: c(v) for k, v in out.items()},
        "counts": {k: int(sum(v)) for k, v in out.items()},
        "steps": c(steps),
        "sim_steps_total": c(sim_tot),
        "sim_steps_per_decision": {"mean": float(np.mean(per_dec)),
                                   "min": int(np.min(per_dec)),
                                   "max": int(np.max(per_dec))},
        "mean_b_eff": c(b_eff),
        "mean_ess": c(ess),
        "mean_phi": c(phi) if phi else None,
        "final_goal_dist": c(gd),
        "timeout_near_goal_le2": c(hover),
        "timeout_near_goal_le1": c(hover1),
        "episodes": [e.as_dict() for e in episodes],
    }


def write_json(name: str, payload: Dict[str, Any]) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
    return path
