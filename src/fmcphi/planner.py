"""FMC planner with the optional Phi factor, instrumented for experiments.

`plan` here is the fmc-core `plan` loop with three changes:

  1. the virtual reward gains the Phi factor (gamma exponent),
  2. every call returns diagnostics, not just the chosen action,
  3. simulator-step cost is counted, so Phi-on and Phi-off runs can be compared
     at equal budget instead of equal N and M.

With gamma = 0 and phi_every = 0 the loop is numerically identical to
fmc.core.plan on the same seed. tests/test_parity.py enforces that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from fmcphi.core import (
    clone_step,
    decide,
    effective_branching_factor,
    effective_sample_size,
    virtual_reward,
)
from fmcphi.phi import kill_dead, phi_composite, virtual_reward_phi


@dataclass
class PlanResult:
    action: Any
    b_eff: float
    ess: float
    mean_phi: float
    min_phi: float
    dead_walkers: int
    sim_steps: int
    labels: List[Any] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d.pop("labels")
        d["action"] = str(self.action)
        return d


def plan(
    env,
    x0,
    N: int = 64,
    M: int = 30,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.0,
    phi_m: int = 4,
    phi_h: int = 3,
    phi_every: int = 1,
    phi_weight_by_survival: bool = True,
    budget_attr: Optional[str] = None,
    budget_max: float = 1.0,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> PlanResult:
    """One FMC decision from x0.

    Parameters
    ----------
    gamma : float
        Exponent of the Phi factor. 0 disables the layer.
    phi_m, phi_h : int
        Fanout and depth of the causal-cone estimate. Cost per evaluation is
        phi_m * phi_h simulator steps.
    phi_every : int
        Recompute Phi every k ticks (1 = every tick, 0 = never). Phi is always
        needed on the tick where cloning uses it, so k > 1 reuses a stale value;
        that is the amortisation knob referenced in docs/SPEC.md section 6.
    """
    rng = rng if rng is not None else np.random.default_rng(seed)
    actions = list(env.actions())
    use_phi = gamma != 0.0 and phi_every > 0

    states = [env.clone_state(x0) for _ in range(N)]
    labels = np.array(
        [actions[rng.integers(0, len(actions))] for _ in range(N)],
        dtype=object,
    )

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
            [np.asarray(env.observe(s), dtype=np.float64).ravel() for s in states]
        )

        partners_dist = rng.permutation(N)
        for i in range(N):
            if partners_dist[i] == i:
                partners_dist[i] = (i + 1) % N

        if use_phi and t % phi_every == 0:
            phis = np.array(
                [
                    phi_composite(
                        env, s, rng, m=phi_m, h=phi_h,
                        budget_attr=budget_attr, budget_max=budget_max,
                        weight_by_survival=phi_weight_by_survival,
                    )
                    for s in states
                ],
                dtype=np.float64,
            )
            sim_steps += N * phi_m * phi_h

        if use_phi:
            vr = virtual_reward_phi(
                rewards, obs, partners_dist, phis,
                alpha=alpha, beta=beta, gamma=gamma,
            )
            vr = kill_dead(vr, phis)
            mean_phi_acc.append(float(phis.mean()))
            min_phi_acc.append(float(phis.min()))
        else:
            vr = virtual_reward(rewards, obs, partners_dist, alpha=alpha, beta=beta)

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


@dataclass
class EpisodeResult:
    outcome: str           # "goal" | "trap" | "fuel" | "dead" | "timeout"
    steps: int
    total_reward: float
    sim_steps: int
    mean_b_eff: float
    mean_phi: float

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def run_episode(
    env,
    x0,
    max_steps: int = 100,
    seed: Optional[int] = None,
    **plan_kwargs,
) -> EpisodeResult:
    """Outer loop: replan at every real step until goal, death or timeout.

    The distinction between "dead" and "timeout" is the whole point of the
    experiment ladder: the Phi layer is supposed to trade timeouts for deaths,
    not to improve the goal rate.
    """
    rng = np.random.default_rng(seed)
    state = env.clone_state(x0)
    total_reward = 0.0
    sim_steps = 0
    b_effs: List[float] = []
    phis: List[float] = []

    for step in range(max_steps):
        res = plan(env, state, rng=rng, **plan_kwargs)
        sim_steps += res.sim_steps
        b_effs.append(res.b_eff)
        if not np.isnan(res.mean_phi):
            phis.append(res.mean_phi)

        state = env.step(state, res.action)
        total_reward += env.reward(state)

        if not env.viable(state):
            cause = getattr(env, "death_cause", lambda _s: "dead")(state) or "dead"
            return EpisodeResult(cause, step + 1, total_reward, sim_steps,
                                 float(np.mean(b_effs)),
                                 float(np.mean(phis)) if phis else float("nan"))
        if env.is_goal(state):
            return EpisodeResult("goal", step + 1, total_reward, sim_steps,
                                 float(np.mean(b_effs)),
                                 float(np.mean(phis)) if phis else float("nan"))

    return EpisodeResult("timeout", max_steps, total_reward, sim_steps,
                         float(np.mean(b_effs)),
                         float(np.mean(phis)) if phis else float("nan"))
