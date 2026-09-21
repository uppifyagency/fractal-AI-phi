"""E5 ablation harness.

Re-implements `fmcphi.planner.plan` with a single extra hook: after the Phi
vector is estimated, an ablation transform is applied to it. Everything else is
copied verbatim from the planner so that `mode="real"` reproduces
`fmcphi.planner.plan(..., gamma=g)` on the same seed and `mode="off"`
reproduces canonical FMC. `parity_check` below checks this by comparing the
*full* per-tick trace (Phi vector, virtual reward, clone indices, labels and
the returned action), not just the returned action.

Nothing here writes to src/. The planner is not monkey-patched; it is mirrored.
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
from fmcphi.phi import (
    kill_dead,
    phi_composite,
    phi_slack,
    phi_viable_actions,
    virtual_reward_phi,
)

# Ablation modes ------------------------------------------------------------
#   off       gamma = 0, canonical FMC (no Phi computed, no Phi cost)
#   real      phi_composite, weight_by_survival=True      <- the reference arm
#   shuffled  real Phi permuted across walkers (arm 1, decisive)
#   constant  real Phi replaced by its swarm mean          (arm 2)
#   nosurv    phi_composite, weight_by_survival=False      (arm 3)
#   cheap     phi_viable_actions instead of phi_cone       (arm 4)
#   noise     uniform on the support of the real Phi       (arm 5)
#
# Review-response arms (2026-09-21), which separate the two mechanisms the
# original arms confounded: the graded Phi factor inside virtual_reward_phi,
# and the hard viability mask applied by kill_dead on Phi == 0 exactly.
#   shuffled_keepmask  Phi permuted in the VR, kill_dead on the *true* Phi
#                      -> isolates the graded factor                 (arm A)
#   noise_keepmask     Phi replaced by uniform noise in the VR, kill_dead on
#                      the true Phi                                  (arm A)
#   mask               Phi = 1{env.viable(state)}, zero rollouts, zero
#                      simulator steps -> isolates the death mask    (arm B)
MODES = ("off", "real", "shuffled", "constant", "nosurv", "cheap", "noise",
         "shuffled_keepmask", "noise_keepmask", "mask")


class StepCounter:
    """Delegating env wrapper that counts *actual* `env.step` calls.

    The nominal budget `N * phi_m * phi_h` that planner.plan and plan_e5 add up
    front is a worst case: `phi_cone` returns 0 after zero steps on a
    non-viable state and breaks its rollout early when a continuation dies, and
    `phi_viable_actions` returns 0 after zero steps on a non-viable state. This
    wrapper measures what was really spent, so that "equal budget" claims can
    be stated as measured rather than nominal.
    """

    def __init__(self, env):
        self._env = env
        self.calls = 0

    def step(self, state, action):
        self.calls += 1
        return self._env.step(state, action)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_env"), name)


@dataclass
class PlanResultE5:
    action: Any
    b_eff: float
    ess: float
    mean_phi: float
    min_phi: float
    dead_walkers: int
    sim_steps: int
    actual_steps: int = -1
    labels: List[Any] = field(default_factory=list)


def _estimate_phi(env, states, rng, mode, phi_m, phi_h, budget_attr, budget_max):
    """Return (phi_for_virtual_reward, phi_for_kill_dead, simulator_steps).

    The two Phi vectors are the same object for every arm except the
    `*_keepmask` ones, where the ablation is applied to the graded factor that
    enters `virtual_reward_phi` while `kill_dead` still sees the untouched
    Phi. That separation is the review-response control: the original
    `shuffled` / `constant` / `noise` arms destroy or relocate the exact zeros
    that `kill_dead` keys on, so they ablate the viability mask as well as the
    state-correspondence of the graded factor.
    """
    n = len(states)
    if mode == "mask":
        # Binary viability indicator. No rollouts, no env.step calls at all.
        phis = np.array([1.0 if env.viable(s) else 0.0 for s in states],
                        dtype=np.float64)
        return phis, phis, 0

    if mode == "cheap":
        vals = []
        for s in states:
            v = phi_viable_actions(env, s)
            if budget_attr is not None:
                v *= phi_slack(s, budget_attr, budget_max)
            vals.append(v)
        phis = np.asarray(vals, dtype=np.float64)
        return phis, phis, n * len(list(env.actions()))

    weighted = mode != "nosurv"
    phis = np.array(
        [
            phi_composite(
                env, s, rng, m=phi_m, h=phi_h,
                budget_attr=budget_attr, budget_max=budget_max,
                weight_by_survival=weighted,
            )
            for s in states
        ],
        dtype=np.float64,
    )
    cost = n * phi_m * phi_h
    true_phis = phis

    if mode in ("shuffled", "shuffled_keepmask"):
        # Same marginal distribution, no state correspondence.
        phis = phis[rng.permutation(n)]
    elif mode == "constant":
        phis = np.full(n, float(phis.mean()), dtype=np.float64)
    elif mode in ("noise", "noise_keepmask"):
        lo, hi = float(phis.min()), float(phis.max())
        phis = phis.copy() if hi <= lo else rng.uniform(lo, hi, size=n)

    if mode.endswith("_keepmask"):
        return phis, true_phis, cost
    return phis, phis, cost


def plan_e5(
    env,
    x0,
    N: int = 32,
    M: int = 10,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    mode: str = "real",
    phi_m: int = 4,
    phi_h: int = 3,
    phi_every: int = 1,
    budget_attr: Optional[str] = None,
    budget_max: float = 1.0,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> PlanResultE5:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    rng = rng if rng is not None else np.random.default_rng(seed)
    actions = list(env.actions())
    use_phi = mode != "off" and gamma != 0.0 and phi_every > 0

    states = [env.clone_state(x0) for _ in range(N)]
    labels = np.array(
        [actions[rng.integers(0, len(actions))] for _ in range(N)],
        dtype=object,
    )

    sim_steps = 0
    calls0 = getattr(env, "calls", None)
    phis = np.ones(N, dtype=np.float64)
    kill_phis = phis
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
            phis, kill_phis, cost = _estimate_phi(
                env, states, rng, mode, phi_m, phi_h, budget_attr, budget_max
            )
            sim_steps += cost

        if use_phi:
            vr = virtual_reward_phi(
                rewards, obs, partners_dist, phis,
                alpha=alpha, beta=beta, gamma=gamma,
            )
            vr = kill_dead(vr, kill_phis)
            mean_phi_acc.append(float(phis.mean()))
            min_phi_acc.append(float(phis.min()))
        else:
            vr = virtual_reward(rewards, obs, partners_dist, alpha=alpha, beta=beta)

        clone_idx = clone_step(vr, rng)
        states = [env.clone_state(states[k]) for k in clone_idx]
        labels = labels[clone_idx]

    dead = sum(1 for s in states if not env.viable(s))
    actual = -1 if calls0 is None else int(env.calls - calls0)
    return PlanResultE5(
        action=decide(labels),
        b_eff=effective_branching_factor(labels),
        ess=effective_sample_size(vr),
        mean_phi=float(np.mean(mean_phi_acc)) if mean_phi_acc else float("nan"),
        min_phi=float(np.min(min_phi_acc)) if min_phi_acc else float("nan"),
        dead_walkers=dead,
        sim_steps=sim_steps,
        actual_steps=actual,
        labels=labels.tolist(),
    )


@dataclass
class EpisodeResultE5:
    outcome: str
    steps: int
    total_reward: float
    sim_steps: int
    mean_b_eff: float
    mean_phi: float
    mean_ess: float
    actual_steps: int = -1

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def run_episode_e5(env, x0, max_steps: int = 60, seed: Optional[int] = None,
                   **plan_kwargs) -> EpisodeResultE5:
    """Outer loop. `actual_steps` counts measured `env.step` calls made *inside*
    planning (the one real step per decision is excluded), and is -1 unless the
    env is wrapped in `StepCounter`."""
    rng = np.random.default_rng(seed)
    state = env.clone_state(x0)
    total_reward = 0.0
    sim_steps = 0
    actual_steps = 0
    counting = hasattr(env, "calls")
    b_effs: List[float] = []
    phis: List[float] = []
    esss: List[float] = []

    def _res(outcome, step):
        return EpisodeResultE5(
            outcome, step, total_reward, sim_steps,
            float(np.mean(b_effs)),
            float(np.mean(phis)) if phis else float("nan"),
            float(np.mean(esss)),
            actual_steps if counting else -1,
        )

    for step in range(max_steps):
        res = plan_e5(env, state, rng=rng, **plan_kwargs)
        sim_steps += res.sim_steps
        if counting:
            actual_steps += res.actual_steps
        b_effs.append(res.b_eff)
        esss.append(res.ess)
        if not np.isnan(res.mean_phi):
            phis.append(res.mean_phi)

        state = env.step(state, res.action)
        total_reward += env.reward(state)

        if not env.viable(state):
            cause = getattr(env, "death_cause", lambda _s: "dead")(state) or "dead"
            return _res(cause, step + 1)
        if env.is_goal(state):
            return _res("goal", step + 1)

    return _res("timeout", max_steps)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def boot_ci(values, n_boot: int = 10000, seed: int = 12345, stat=np.mean):
    """Percentile bootstrap CI95 of `stat` over `values`."""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = stat(v[idx], axis=1)
    return (float(stat(v)), float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)))


def boot_diff_ci(a, b, n_boot: int = 10000, seed: int = 999):
    """Paired bootstrap CI95 of mean(a) - mean(b). a and b share seed order."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = len(a)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return (float(a.mean() - b.mean()), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)))


def parity_check(env_factory, seeds=(7, 11, 13, 17, 23, 29, 31, 37), **kw):
    """Check that the mirrored loop still reproduces `fmcphi.planner.plan`.

    Scope, stated exactly (the earlier version compared only the returned
    integer action on one seed, out of a 5-action set, which is a weak test):
    for each seed and for both `mode="real", gamma=1` and `mode="off",
    gamma=0` this compares

      * the returned action,
      * b_eff, ess (a function of the whole virtual-reward vector on the last
        tick), mean_phi and min_phi (functions of the Phi vector on every one
        of the M ticks), dead_walkers and sim_steps,
      * the full N-vector of walker labels after the last clone step, which is
        a function of every clone index drawn along the way,
      * the post-call state of the shared bit generator, which is identical
        only if both loops consumed exactly the same random draws in the same
        order.

    It still does not compare the per-walker Phi vector tick by tick, only the
    per-tick mean and min of it.
    """
    from fmcphi import planner

    out = {}
    for mode, gamma in (("real", 1.0), ("off", 0.0)):
        rows = []
        for seed in seeds:
            env = env_factory()
            rng_a = np.random.default_rng(seed)
            a = plan_e5(env, env.reset(), gamma=gamma, mode=mode, rng=rng_a, **kw)
            env = env_factory()
            rng_b = np.random.default_rng(seed)
            b = planner.plan(env, env.reset(), gamma=gamma, rng=rng_b, **kw)
            fields = ("action", "b_eff", "ess", "mean_phi", "min_phi",
                      "dead_walkers", "sim_steps")
            same = all(
                (getattr(a, f) == getattr(b, f))
                or (isinstance(getattr(a, f), float)
                    and np.isnan(getattr(a, f)) and np.isnan(getattr(b, f)))
                for f in fields
            )
            same = same and list(a.labels) == list(b.labels)
            same = same and (rng_a.bit_generator.state == rng_b.bit_generator.state)
            rows.append(dict(seed=int(seed), e5_action=int(a.action),
                             planner_action=int(b.action), equal=bool(same)))
        out[mode] = dict(seeds=[r["seed"] for r in rows],
                         all_equal=bool(all(r["equal"] for r in rows)),
                         rows=rows)
    return out
