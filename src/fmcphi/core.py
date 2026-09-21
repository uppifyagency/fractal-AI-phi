"""Core Fractal Monte Carlo math layer.

Maps 1-to-1 to definitions in docs/MATH_CANON_ORIGIN.md:

    Definition 2 -> relativize
    Definition 3 -> virtual_reward
    Definition 4 -> clone_step
    Definition 5 -> effective_sample_size
    Definition 6 -> effective_branching_factor
    Definition 1 -> plan (full iteration loop), decide (final marginal)

The Python implementation here and the JS port in ../../../js/fmc.js are
designed to produce identical virtual reward vectors given the same input
arrays and seed. The bit-for-bit equivalence is enforced by tests/test_math.py.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from fmcphi.envs.base import Environment

EPS = 1e-10


# ---------------------------------------------------------------------------
# Definition 2 — relativize
# ---------------------------------------------------------------------------

def relativize(vector: np.ndarray) -> np.ndarray:
    """Apply the FMC relativize transform.

    Identical to FractalAI_old.swarm.relativize_vector. The standard deviation
    is computed with NumPy's default biased estimator (population, ddof=0) to
    match the reference implementation exactly.

    Parameters
    ----------
    vector : np.ndarray, shape (N,)
        Raw reward or distance values. Sign is unrestricted.

    Returns
    -------
    np.ndarray, shape (N,), strictly positive.
    """
    vector = np.asarray(vector, dtype=np.float64)
    std = vector.std()
    if std == 0:
        return np.ones(len(vector), dtype=np.float64)
    z = (vector - vector.mean()) / std
    out = np.empty_like(z)
    pos = z > 0
    out[pos] = np.log(1.0 + z[pos]) + 1.0
    out[~pos] = np.exp(z[~pos])
    return out


# ---------------------------------------------------------------------------
# Definition 3 — virtual reward
# ---------------------------------------------------------------------------

def virtual_reward(
    rewards: np.ndarray,
    states: np.ndarray,
    partners: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> np.ndarray:
    """Compute the FMC virtual reward.

    VR_i = relativize(R_i)^alpha * relativize(d(W_i, W_partner_i))^beta

    Parameters
    ----------
    rewards : np.ndarray, shape (N,)
        Raw rewards R_i (pre-relativize).
    states : np.ndarray, shape (N, ...)
        Walker states (used for euclidean pairwise distance).
    partners : np.ndarray, shape (N,), int
        Index permutation: partners[i] = j(i), the partner of walker i.
    alpha : float, default 1.0
        Reward exponent. alpha=0 -> Common Sense mode.
    beta : float, default 1.0
        Distance exponent. beta=0 -> no anti-collapse term.

    Returns
    -------
    np.ndarray, shape (N,), strictly positive virtual reward.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    states = np.asarray(states, dtype=np.float64)
    partners = np.asarray(partners, dtype=np.int64)

    flat = states.reshape(states.shape[0], -1)
    dist = np.sqrt(((flat - flat[partners]) ** 2).sum(axis=1))

    r_hat = relativize(rewards)
    d_hat = relativize(dist)
    return (r_hat ** alpha) * (d_hat ** beta)


# ---------------------------------------------------------------------------
# Definition 5 — effective sample size
# ---------------------------------------------------------------------------

def effective_sample_size(vr: np.ndarray) -> float:
    """ESS = (sum vr)^2 / sum vr^2.  Range [1, N]."""
    vr = np.asarray(vr, dtype=np.float64)
    s = vr.sum()
    s2 = (vr * vr).sum()
    if s2 == 0:
        return float(len(vr))
    return float(s * s / s2)


# ---------------------------------------------------------------------------
# Definition 6 — effective branching factor
# ---------------------------------------------------------------------------

def effective_branching_factor(labels: np.ndarray) -> float:
    """exp(H(label_distribution)) over the surviving walker labels.

    Range: [1, K] where K is the number of distinct possible labels.
    1 = palmera (all walkers same label).
    K = matorral (uniform over labels).
    Sergio's sweet spot conjecture: ~6 (see Conjecture A in MATH_CANON).
    """
    labels = np.asarray(labels)
    if len(labels) == 0:
        return 1.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))


# ---------------------------------------------------------------------------
# Definition 4 — cloning step
# ---------------------------------------------------------------------------

def clone_step(
    vr: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pairwise probabilistic clone selection.

    For each walker i, picks a partner k uniformly at random (k != i),
    then with probability max(0, (VR_k - VR_i) / VR_i) flags i as cloning to k.

    Returns
    -------
    clone_idx : np.ndarray, shape (N,), int
        clone_idx[i] = i if walker stays, else partner index k to clone from.
    """
    vr = np.asarray(vr, dtype=np.float64)
    n = len(vr)
    if n < 2:
        return np.arange(n)

    # Random partner != self.
    partners = rng.integers(0, n, size=n)
    same = partners == np.arange(n)
    while same.any():
        partners[same] = rng.integers(0, n, size=int(same.sum()))
        same = partners == np.arange(n)

    vr_safe = np.where(vr > 0, vr, EPS)
    p_clone = np.clip((vr[partners] - vr) / vr_safe, 0.0, 1.0)
    # VR_i = 0 -> always clone (matches FractalAI_old branch).
    p_clone[vr == 0] = 1.0

    will_clone = rng.random(n) < p_clone
    out = np.where(will_clone, partners, np.arange(n))
    return out


# ---------------------------------------------------------------------------
# Definition 1 — full iteration & final decision
# ---------------------------------------------------------------------------

def decide(labels: np.ndarray) -> int:
    """Final FMC decision: argmax bincount(labels)."""
    labels = np.asarray(labels)
    return int(Counter(labels.tolist()).most_common(1)[0][0])


def plan(
    env: Environment,
    x0,
    N: int = 64,
    M: int = 30,
    alpha: float = 1.0,
    beta: float = 1.0,
    seed: Optional[int] = None,
) -> int:
    """Run an FMC plan from x0 and return the chosen action.

    Implements the iteration W_{t+1} = S_t o C_t (W_t) of MATH_CANON
    Definition 1, then returns decide(labels).

    Parameters
    ----------
    env : Environment
        Must implement clone_state, step, observe, reward, sample_action.
    x0
        Initial state (whatever env.clone_state expects).
    N : int
        Number of walkers.
    M : int
        Planning horizon (simulator ticks).
    alpha, beta : float
        Virtual reward exponents.
    seed : int or None
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    actions = list(env.actions())

    # Init swarm: same x0, random labels (initial decisions).
    states = [env.clone_state(x0) for _ in range(N)]
    labels = np.array(
        [actions[rng.integers(0, len(actions))] for _ in range(N)],
        dtype=object,
    )

    for t in range(M):
        # PREDICT: step every walker by an action.
        for i in range(N):
            a = labels[i] if t == 0 else env.sample_action(states[i], rng)
            states[i] = env.step(states[i], a)

        # WEIGHT: compute virtual reward.
        rewards = np.array([env.reward(s) for s in states], dtype=np.float64)
        obs = np.stack([np.asarray(env.observe(s), dtype=np.float64).ravel() for s in states])
        partners_dist = rng.permutation(N)
        # Avoid self-pair where possible.
        for i in range(N):
            if partners_dist[i] == i:
                partners_dist[i] = (i + 1) % N
        vr = virtual_reward(rewards, obs, partners_dist, alpha=alpha, beta=beta)

        # RESAMPLE: cloning kernel.
        clone_idx = clone_step(vr, rng)
        states = [env.clone_state(states[k]) for k in clone_idx]
        labels = labels[clone_idx]

    return decide(labels)
