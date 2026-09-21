"""The Phi layer: causal-cone freedom as a third virtual-reward factor.

Motivation
----------
Canonical FMC computes VR_i = relativize(R_i)^alpha * relativize(d(W_i, W_j))^beta.
The distance term d measures dispersion *between walkers* (hypothesis diversity),
not the entropy of futures reachable *from one state*. The object Wissner-Gross &
Freer (2013) call the causal entropic force, F = T_c grad_X S_c, is the latter.

Empirically the two are not interchangeable: in the reference repo's rocket sweep,
raising beta *lowers* the effective branching factor (5.45 -> 3.52 -> 1.89 for
beta = 0, 1, 5), so beta acts as an extra selector, not as an option preserver.

This module adds the missing factor:

    Phi(x) = exp( H( distribution of distinct viable states reachable from x
                     in h steps ) )

which is Definition 6 (effective branching factor) evaluated *forward from one
state*, instead of backward over surviving swarm labels. The full virtual reward
becomes

    VR_i = Rhat_i^alpha * Dhat_i^beta * Phihat_i^gamma

See docs/SPEC.md for the formal statement.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Optional

import numpy as np

from fmcphi.core import relativize

EPS = 1e-10


# ---------------------------------------------------------------------------
# Phi estimators
# ---------------------------------------------------------------------------

def phi_cone(
    env,
    state,
    rng: np.random.Generator,
    m: int = 4,
    h: int = 3,
    weight_by_survival: bool = True,
) -> float:
    """Monte-Carlo estimate of the causal-cone freedom at `state`.

    Fans out `m` random continuations of depth `h` from `state` and returns

        Phi = p_survive * exp( H( endpoint keys | survived ) )

    The perplexity term counts *how many different* futures are still open. The
    survival factor weights it by *how much probability mass* actually reaches
    them: a path that hits an absorbing state contributes no further branching,
    so the causal path entropy of Wissner-Gross & Freer (2013) sees it as mass
    removed, not as one more endpoint. Set `weight_by_survival=False` to recover
    the unweighted perplexity (used as an ablation arm in E5).

    Returns
    -------
    float in [0, m]
        0.0  -> the state is already dead, or every continuation died.
        1.0  -> one distinct viable future reached with certainty (a corridor).
        m    -> every continuation survived and ended somewhere different.

    Notes
    -----
    Cost is at most m*h simulator steps per call, which is why `phi_every` in
    planner.plan exists. Wall clamping lowers Phi too: a corner really is
    option-poor, and the layer is not meant to encode "danger" specifically, it
    encodes "futures still reachable". Danger enters because a trap state has
    Phi = 0 exactly.
    """
    if not env.viable(state):
        return 0.0

    endpoints = []
    for _ in range(m):
        s = env.clone_state(state)
        alive = True
        for _ in range(h):
            a = env.sample_action(s, rng)
            s = env.step(s, a)
            if not env.viable(s):
                alive = False
                break
        if alive:
            endpoints.append(env.key(s))

    if not endpoints:
        return 0.0

    counts = np.array(list(Counter(endpoints).values()), dtype=np.float64)
    p = counts / counts.sum()
    perplexity = float(np.exp(-np.sum(p * np.log(p))))
    if not weight_by_survival:
        return perplexity
    return perplexity * (len(endpoints) / float(m))


def phi_viable_actions(env, state) -> float:
    """Cheap depth-1 Phi: how many single actions keep the state viable.

    Cost is K simulator steps, deterministic, no rng. Returns a float in [0, K].
    Use when phi_cone is too expensive; it is the h=1, exhaustive-fanout limit
    of the same quantity with the distinctness test dropped.
    """
    if not env.viable(state):
        return 0.0
    n = 0
    for a in env.actions():
        if env.viable(env.step(env.clone_state(state), a)):
            n += 1
    return float(n)


def phi_slack(state, budget_attr: str = "fuel", budget_max: float = 1.0) -> float:
    """Resource-slack component of Phi.

    A state can be viable and out of fuel, which is dead one step later. Returns
    the remaining budget normalised to [0, 1]. Multiply into the other Phi terms:
    a state with no slack has no futures regardless of its topology.
    """
    value = float(getattr(state, budget_attr))
    if budget_max <= 0:
        return 1.0
    return float(max(0.0, min(1.0, value / budget_max)))


def phi_composite(
    env,
    state,
    rng: np.random.Generator,
    m: int = 4,
    h: int = 3,
    budget_attr: Optional[str] = None,
    budget_max: float = 1.0,
    weight_by_survival: bool = True,
) -> float:
    """phi_cone times phi_slack, multiplicative as in paper section 2.2.2.

    Multiplicative and not additive on purpose: with an additive composition a
    large enough goal reward can buy a state with zero futures.
    """
    value = phi_cone(env, state, rng, m=m, h=h,
                     weight_by_survival=weight_by_survival)
    if budget_attr is not None:
        value *= phi_slack(state, budget_attr, budget_max)
    return value


# ---------------------------------------------------------------------------
# Three-factor virtual reward
# ---------------------------------------------------------------------------

def virtual_reward_phi(
    rewards: np.ndarray,
    states_obs: np.ndarray,
    partners: np.ndarray,
    phis: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.0,
) -> np.ndarray:
    """VR_i = relativize(R_i)^alpha * relativize(d_i)^beta * relativize(Phi_i)^gamma.

    gamma = 0 reproduces canonical FMC exactly (the relativize of the Phi vector
    is still computed but raised to the power 0), which is what makes the E5
    ablation a true A/B.

    A walker with Phi = 0 keeps a small but non-zero relativized value, because
    `relativize` is strictly positive by construction. The hard kill is applied
    separately by `kill_dead`, so that "dead" and "merely cornered" stay
    distinguishable in the diagnostics.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    states_obs = np.asarray(states_obs, dtype=np.float64)
    partners = np.asarray(partners, dtype=np.int64)
    phis = np.asarray(phis, dtype=np.float64)

    flat = states_obs.reshape(states_obs.shape[0], -1)
    dist = np.sqrt(((flat - flat[partners]) ** 2).sum(axis=1))

    r_hat = relativize(rewards)
    d_hat = relativize(dist)
    p_hat = relativize(phis)
    return (r_hat ** alpha) * (d_hat ** beta) * (p_hat ** gamma)


def kill_dead(vr: np.ndarray, phis: np.ndarray) -> np.ndarray:
    """Zero the virtual reward of walkers whose Phi is exactly 0.

    Keeps the multiplicative semantics of paper section 2.2.2 for hard
    constraints: a state with no reachable future is not merely unattractive,
    it is excluded. clone_step already treats VR = 0 as "always clone away".
    """
    vr = np.asarray(vr, dtype=np.float64).copy()
    vr[np.asarray(phis, dtype=np.float64) <= 0.0] = 0.0
    return vr
