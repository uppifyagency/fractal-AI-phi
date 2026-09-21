"""Shared machinery for E4: rollouts with trajectory logging, bootstrap CIs.

Nothing here touches src/fmcphi except by import. `rollout_fmc` replicates
`fmcphi.planner.run_episode` step for step (same rng construction, same order of
calls, same termination tests) and only adds trajectory recording, so an E4
number and a `run_episode` number for the same seed and kwargs are the same
number. `test_rollout_matches_run_episode` in run_e4.py checks that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from fmcphi.envs.trapgrid import TrapGrid
from fmcphi.planner import plan


# ---------------------------------------------------------------------------
# absorbing-death wrapper
# ---------------------------------------------------------------------------

class AbsorbingTrapGrid(TrapGrid):
    """`TrapGrid` with death as a fixed point of the transition function.

    Stock `TrapGrid.step` transitions normally out of a trap cell and keeps
    burning fuel past zero: death exists only as a predicate in `viable`. Any
    arm that never consults `viable` (every gamma = 0 arm, since `plan` gates
    the Phi evaluation and `kill_dead` behind `gamma != 0`) therefore rolls its
    swarm straight through its own death and scores futures that cannot happen.
    That is a defect of the *measurement*, not of the agent, and it inflates
    every Phi-minus-no-Phi contrast in this experiment.

    `src/fmcphi/envs/trapgrid.py` is shared with E1 and E5 and is not owned by
    E4, so the guard lives here as a subclass instead of in the environment.
    Every E4 arm on TrapGrid, FMC and baseline policy alike, runs on this class,
    so the comparison stays internally consistent. `FunnelRooms` is owned by E4
    and got the same guard in place.
    """

    def step(self, state, action):
        if not self.viable(state):
            return self.clone_state(state)
        return TrapGrid.step(self, state, action)


# ---------------------------------------------------------------------------
# rollouts
# ---------------------------------------------------------------------------

@dataclass
class Rollout:
    arm: str
    seed: int
    outcome: str
    steps: int
    sim_steps: int
    total_reward: float
    mean_b_eff: float
    mean_phi: float
    cells_visited: int
    min_goal_dist: int
    final_goal_dist: int
    frac_steps_in_goal_region: float
    path: List[Any] = field(default_factory=list)

    def as_dict(self, with_path: bool = False) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "path"}
        if with_path:
            d["path"] = [list(p) for p in self.path]
        return d


def _goal_dist(env, state) -> int:
    return int(abs(state.x - env.goal[0]) + abs(state.y - env.goal[1]))


def _goal_region(env, state) -> bool:
    """Is this cell in the option-poor neighbourhood of the goal?

    TrapGrid: the last two columns. FunnelRooms: room C or past the BC door.
    Both are "the agent has committed toward the goal" tests, used only as a
    descriptive diagnostic, never as a success criterion.
    """
    if hasattr(env, "region"):
        return env.region(state.x, state.y) in ("door_BC", "C", "goal")
    return state.x >= env.width - 2


def _finish(env, arm, seed, state, step, total_reward, sim_steps,
            b_effs, phis, path, outcome=None) -> Rollout:
    dists = [abs(p[0] - env.goal[0]) + abs(p[1] - env.goal[1]) for p in path]
    in_goal_region = sum(1 for p in path if _goal_region(env, _P(*p)))
    return Rollout(
        arm=arm,
        seed=seed,
        outcome=outcome,
        steps=step,
        sim_steps=sim_steps,
        total_reward=float(total_reward),
        mean_b_eff=float(np.mean(b_effs)) if b_effs else float("nan"),
        mean_phi=float(np.mean(phis)) if phis else float("nan"),
        cells_visited=len(set(path)),
        min_goal_dist=int(min(dists)) if dists else _goal_dist(env, state),
        final_goal_dist=int(dists[-1]) if dists else _goal_dist(env, state),
        frac_steps_in_goal_region=float(in_goal_region) / max(1, len(path)),
        path=list(path),
    )


class _P:
    """Tiny positional shim so _goal_region can take a (x, y) tuple."""

    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y


def rollout_fmc(env, arm: str, seed: int, max_steps: int = 60,
                **plan_kwargs) -> Rollout:
    """Replica of fmcphi.planner.run_episode with trajectory logging."""
    rng = np.random.default_rng(seed)
    state = env.clone_state(env.reset())
    total_reward = 0.0
    sim_steps = 0
    b_effs: List[float] = []
    phis: List[float] = []
    path = [(state.x, state.y)]

    for step in range(max_steps):
        res = plan(env, state, rng=rng, **plan_kwargs)
        sim_steps += res.sim_steps
        b_effs.append(res.b_eff)
        if not np.isnan(res.mean_phi):
            phis.append(res.mean_phi)

        state = env.step(state, res.action)
        total_reward += env.reward(state)
        path.append((state.x, state.y))

        if not env.viable(state):
            cause = getattr(env, "death_cause", lambda _s: "dead")(state) or "dead"
            return _finish(env, arm, seed, state, step + 1, total_reward,
                           sim_steps, b_effs, phis, path, cause)
        if env.is_goal(state):
            return _finish(env, arm, seed, state, step + 1, total_reward,
                           sim_steps, b_effs, phis, path, "goal")

    return _finish(env, arm, seed, state, max_steps, total_reward, sim_steps,
                   b_effs, phis, path, "timeout")


def rollout_policy(env, arm: str, seed: int, max_steps: int = 60,
                   safe: bool = False) -> Rollout:
    """Baseline policies. `safe=False` is uniform random over the action set.

    `safe=True` is uniform over the actions that leave the next state viable
    (falling back to uniform when every action is lethal). It is the honest
    control for P3: it survives without any goal information at all, so it
    isolates "reaches the goal" from "is alive long enough to bump into it".
    """
    rng = np.random.default_rng(seed)
    state = env.clone_state(env.reset())
    total_reward = 0.0
    acts = list(env.actions())
    path = [(state.x, state.y)]
    # SPEC section 4 invariant 2: sim_steps counts every simulator call. The
    # safe policy probes one step per action to build its viable pool, so it
    # costs |A| calls per real step, not zero. The uniform policy costs zero.
    sim_steps = 0

    for step in range(max_steps):
        if safe:
            ok = [a for a in acts if env.viable(env.step(env.clone_state(state), a))]
            sim_steps += len(acts)
            pool = ok if ok else acts
        else:
            pool = acts
        a = pool[int(rng.integers(0, len(pool)))]
        state = env.step(state, a)
        total_reward += env.reward(state)
        path.append((state.x, state.y))

        if not env.viable(state):
            cause = getattr(env, "death_cause", lambda _s: "dead")(state) or "dead"
            return _finish(env, arm, seed, state, step + 1, total_reward, sim_steps,
                           [], [], path, cause)
        if env.is_goal(state):
            return _finish(env, arm, seed, state, step + 1, total_reward, sim_steps,
                           [], [], path, "goal")

    return _finish(env, arm, seed, state, max_steps, total_reward, sim_steps,
                   [], [], path, "timeout")


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def boot_ci(values, n_boot: int = 10000, seed: int = 12345, stat=np.mean):
    """Percentile bootstrap CI95 of `stat` over `values`."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots = stat(v[idx], axis=1)
    return (float(stat(v)), float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)))


def boot_diff_ci(a, b, n_boot: int = 10000, seed: int = 999):
    """Percentile bootstrap CI95 of mean(a) - mean(b), independent resampling."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, a.size, size=(n_boot, a.size))
    ib = rng.integers(0, b.size, size=(n_boot, b.size))
    boots = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return (float(a.mean() - b.mean()), float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)))


def fmt_ci(triple, nd: int = 2) -> str:
    m, lo, hi = triple
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def dump_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
