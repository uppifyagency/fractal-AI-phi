"""Unit tests for the E2 mock backend and harness.

Run with:
    cd /Users/vladvrinceanu/fractal-AI-phi && \
      uv run pytest experiments/E2_fot_phi/scripts/test_e2_mock.py -q

These do not touch the repo's own tests, and they do not import anything from
another experiment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[2] / "src"))

import fmcphi.core as core                                   # noqa: E402
from fmcphi.core import relativize                           # noqa: E402
from fmcphi.phi import phi_cone, phi_composite               # noqa: E402
from fmcphi.planner import plan                              # noqa: E402

import mock_llm                                              # noqa: E402
from mock_llm import ReasoningEnv, make_problem              # noqa: E402
import fot_phi                                               # noqa: E402
from fot_phi import Arm, make_arms, assert_equal_budget, run_chain_episode  # noqa: E402


# ---------------------------------------------------------------------------
# The environment satisfies the fmcphi contract
# ---------------------------------------------------------------------------

def test_env_implements_protocol():
    env = ReasoningEnv(make_problem(0))
    s = env.reset()
    assert list(env.actions()) == [0, 1, 2, 3, 4]
    assert env.viable(s)
    assert isinstance(env.key(s), tuple)
    assert env.observe(s).ndim == 1
    assert isinstance(env.reward(s), float)


def test_clone_state_is_a_deep_copy():
    env = ReasoningEnv(make_problem(1))
    s = env.reset()
    c = env.clone_state(s)
    c.stage = 99
    assert s.stage == 0


def test_answered_state_is_absorbing_and_free():
    env = ReasoningEnv(make_problem(2))
    s = env.reset()
    while not s.answered:
        s = env.step(s, 0)
    before = (s.stage, s.branch, s.tokens)
    for a in range(5):
        s = env.step(s, a)
    assert (s.stage, s.branch, s.tokens) == before
    assert env.viable(s), "a finished chain is not a dead chain"
    assert env.key(s)[0] == "ans"


def test_death_is_reachable_so_the_cone_can_narrow():
    """docs/SPEC.md section 5: no death, no test. Both causes must be live."""
    env = ReasoningEnv(make_problem(3))
    s = env.reset()
    for _ in range(20):                      # burn context with backtracks
        s = env.step(s, 3)
    assert not env.viable(s)
    assert env.death_cause(s) in {"budget", "contradiction"}
    assert phi_cone(env, s, np.random.default_rng(0), m=3, h=2) == 0.0


# ---------------------------------------------------------------------------
# The verifier never sees the ground truth
# ---------------------------------------------------------------------------

def test_judge_and_geometry_are_ground_truth_free():
    """Change the ground-truth key; every planner-facing quantity is unchanged.

    If `reward`, `viable`, `key`, `observe` or `sample_action` leaked the
    answer, the experiment would be measuring an oracle, not a verifier.
    """
    rng_states = []
    outs = []
    for gt in ("GT", "SOMETHING_ELSE"):
        prob = make_problem(4)
        prob.answer = gt
        env = ReasoningEnv(prob)
        rng = np.random.default_rng(7)
        s = env.reset()
        rec = []
        for _ in range(12):
            a = env.sample_action(s, rng)
            s = env.step(s, a)
            rec.append((a, env.reward(s), env.viable(s),
                        tuple(np.round(env.observe(s), 9)),
                        env.key(s) if not s.answered else "ans"))
        outs.append(rec)
        rng_states.append(rng.bit_generator.state["state"])
    assert outs[0] == outs[1]
    assert rng_states[0] == rng_states[1]


# ---------------------------------------------------------------------------
# Parity with canonical FMC at gamma = 0 (SPEC section 4, invariant 1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_gamma_zero_is_canonical_fmc(seed):
    env = ReasoningEnv(make_problem(5))
    x0 = env.reset()
    a_phi = plan(env, x0, N=16, M=4, alpha=1.0, beta=1.0, gamma=0.0,
                 seed=seed).action
    a_core = core.plan(env, x0, N=16, M=4, alpha=1.0, beta=1.0, seed=seed)
    assert a_phi == a_core


# ---------------------------------------------------------------------------
# Budget accounting (task constraint 6)
# ---------------------------------------------------------------------------

def test_arms_are_budget_matched_and_the_planner_agrees():
    arms = make_arms(N=16, M=4, phi_m=3, phi_h=2)
    budget = assert_equal_budget(arms)
    env = ReasoningEnv(make_problem(6))
    x0 = env.reset()
    for a in arms:
        res = plan(env, x0, rng=np.random.default_rng(0), **a.plan_kwargs(env))
        assert res.sim_steps == a.budget_per_plan(), a.name
        if a.name != "fot_nm":
            assert res.sim_steps == budget, a.name


def test_phi_ladder_solves_the_budget_equation_exactly():
    for M in (2, 3, 4, 6):
        for m, h in ((3, 2), (2, 2), (4, 3)):
            ref = M * (1 + m * h)
            for M_a, k in fot_phi.phi_ladder(M, m, h):
                import math
                assert M_a + math.ceil(M_a / k) * m * h == ref


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_episode_is_deterministic():
    arm = make_arms(16, 4, 3, 2)[2]
    r1 = run_chain_episode(ReasoningEnv(make_problem(7)), arm, seed=123)
    r2 = run_chain_episode(ReasoningEnv(make_problem(7)), arm, seed=123)
    for k in ("correct", "answer", "steps", "sim_steps", "actions"):
        assert r1[k] == r2[k]


def test_problem_generation_is_deterministic():
    a = make_problem(11, phi_signal=0.75, p_detect=0.6)
    b = make_problem(11, phi_signal=0.75, p_detect=0.6)
    assert a.rank_correct == b.rank_correct
    assert a.narrow == b.narrow and a.detectable == b.detectable


# ---------------------------------------------------------------------------
# The two signal channels are independent (no circularity)
# ---------------------------------------------------------------------------

def test_narrow_and_detectable_are_independent_draws():
    n_both = n_nar = n_det = n = 0
    for pid in range(400):
        p = make_problem(pid, phi_signal=0.5, p_detect=0.5)
        for bid in p.narrow:
            n += 1
            n_nar += p.narrow[bid]
            n_det += p.detectable[bid]
            n_both += p.narrow[bid] and p.detectable[bid]
    # P(both) should be near P(narrow)*P(detectable) = 0.25, not near 0.5.
    assert 0.20 < n_both / n < 0.30, n_both / n
    assert 0.45 < n_nar / n < 0.55
    assert 0.45 < n_det / n < 0.55


def test_phi_separates_committed_wrong_branches_from_open_ones():
    """The load-bearing modelling assumption, stated as a test.

    A narrow wrong branch must score lower Phi than an open correct one, and a
    decoy (non-narrow) wrong branch must not. If this fails the mock encodes no
    cone signal at all and every downstream number is meaningless.
    """
    rng = np.random.default_rng(0)
    buckets = {"b0": [], "narrow": [], "decoy": []}
    for pid in range(30):
        p = make_problem(pid, phi_signal=0.6)
        env = ReasoningEnv(p)
        for _ in range(60):
            s = env.reset()
            for _ in range(int(rng.integers(0, 3))):
                s = env.step(s, env.sample_action(s, rng))
            if s.answered:
                continue
            v = phi_cone(env, s, rng, m=4, h=2)
            if s.branch == 0:
                buckets["b0"].append(v)
            elif p.narrow.get(s.branch, False):
                buckets["narrow"].append(v)
            else:
                buckets["decoy"].append(v)
    mb0 = float(np.mean(buckets["b0"]))
    mnar = float(np.mean(buckets["narrow"]))
    mdec = float(np.mean(buckets["decoy"]))
    assert mnar < mb0 - 0.5, (mnar, mb0)
    assert mdec > mnar + 0.3, (mdec, mnar)


# ---------------------------------------------------------------------------
# The single-step tier is the SPEC section 5 null by construction
# ---------------------------------------------------------------------------

def test_single_tier_phi_is_exactly_flat_across_every_decision():
    """docs/SPEC.md section 5, stated as an exact assertion.

    No irreversibility and no context cost means the cone never narrows, so the
    three continuations a decision chooses between all score the same Phi and
    `relativize` maps them to exactly ones. Not approximately: exactly. That
    matters because `relativize` is scale free and would amplify any residual.
    """
    rng = np.random.default_rng(0)
    n_checked = 0
    for pid in range(12):
        p = make_problem(pid, tier="single")
        env = ReasoningEnv(p)
        for _ in range(24):
            s = env.reset()
            for _ in range(int(rng.integers(0, 3))):
                s = env.step(s, env.sample_action(s, rng))
            assert env.viable(s), "single tier must have no absorbing death"
            sibs = [env.step(env.clone_state(s), a) for a in (0, 1, 2)]
            ph = np.array([phi_composite(env, c, rng, m=4, h=2,
                                         budget_attr="tokens",
                                         budget_max=p.tokens0) for c in sibs])
            assert ph.std() == 0.0, ph
            assert np.all(relativize(ph) == 1.0), ph
            n_checked += 1
    assert n_checked >= 200


def test_multi_tier_phi_is_not_flat():
    """The complement of the test above: the multi tier is a live test bed."""
    rng = np.random.default_rng(0)
    spreads = []
    for pid in range(12):
        p = make_problem(pid, tier="multi")
        env = ReasoningEnv(p)
        for _ in range(24):
            s = env.reset()
            for _ in range(int(rng.integers(0, 3))):
                s = env.step(s, env.sample_action(s, rng))
            if not env.viable(s):
                continue
            sibs = [env.step(env.clone_state(s), a) for a in (0, 1, 2)]
            ph = np.array([phi_composite(env, c, rng, m=4, h=2,
                                         budget_attr="tokens",
                                         budget_max=p.tokens0) for c in sibs])
            spreads.append(float(ph.std()))
    assert float(np.mean(spreads)) > 0.2, float(np.mean(spreads))


def test_single_tier_gamma_does_not_change_the_virtual_reward():
    """Flat Phi over the options of a decision -> gamma is the identity factor.

    Checked on the set the decision actually ranges over, the three sibling
    continuations of a state. Over a *mixed* cloud (some chains finished, some
    not) Phi is not constant even here, because finishing is absorbing; that
    residual cannot move a decision, but it is a real caveat about reading
    "Phi is flat" as "Phi is constant everywhere". See REPORT.md.
    """
    from fmcphi.phi import virtual_reward_phi
    from fmcphi.core import virtual_reward
    rng = np.random.default_rng(0)
    checked = 0
    for pid in range(8):
        env = ReasoningEnv(make_problem(pid, tier="single"))
        for _ in range(20):
            s = env.reset()
            for _ in range(int(rng.integers(0, 3))):
                s = env.step(s, env.sample_action(s, rng))
            sibs = [env.step(env.clone_state(s), a) for a in (0, 1, 2)]
            phis = np.array([phi_composite(env, c, rng, m=4, h=2,
                                           budget_attr="tokens",
                                           budget_max=env.p.tokens0)
                             for c in sibs])
            rewards = np.array([env.reward(c) for c in sibs])
            obs = np.stack([env.observe(c) for c in sibs])
            partners = np.array([1, 2, 0])
            vr0 = virtual_reward(rewards, obs, partners, alpha=1.0, beta=1.0)
            for gamma in (0.5, 1.0, 3.0):
                vr1 = virtual_reward_phi(rewards, obs, partners, phis,
                                         alpha=1.0, beta=1.0, gamma=gamma)
                assert np.array_equal(vr0, vr1)
            checked += 1
    assert checked >= 100
