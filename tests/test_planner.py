"""The planner contract, including exact parity with the vendored reference at gamma = 0."""

from collections import Counter

import numpy as np
import pytest

import fmcphi.core as core
from fmcphi import plan, run_episode
from fmcphi.envs.trapgrid import TrapGrid


@pytest.fixture
def env():
    return TrapGrid(fuel=24.0)


def test_gamma_zero_matches_the_vendored_core_exactly(env):
    """Same seed, same action. Guards the A/B: any delta must come from Phi."""
    s = env.reset()
    for seed in range(5):
        ours = plan(env, s, N=16, M=6, alpha=1.0, beta=1.0, gamma=0.0, seed=seed)
        theirs = core.plan(env, s, N=16, M=6, alpha=1.0, beta=1.0, seed=seed)
        assert ours.action == theirs


def test_phi_costs_simulator_steps(env):
    s = env.reset()
    off = plan(env, s, N=16, M=6, gamma=0.0, seed=0)
    on = plan(env, s, N=16, M=6, gamma=1.0, phi_m=4, phi_h=3, seed=0)
    assert off.sim_steps == 16 * 6
    assert on.sim_steps == 16 * 6 + 16 * 4 * 3 * 6


def test_phi_every_amortises_the_cost(env):
    s = env.reset()
    every = plan(env, s, N=16, M=6, gamma=1.0, phi_every=1, seed=0)
    third = plan(env, s, N=16, M=6, gamma=1.0, phi_every=3, seed=0)
    assert third.sim_steps < every.sim_steps


def test_diagnostics_are_populated(env):
    s = env.reset()
    r = plan(env, s, N=16, M=6, gamma=1.0, phi_m=4, phi_h=3, seed=0)
    assert 1.0 <= r.b_eff <= 5.0
    assert 1.0 <= r.ess <= 16.0
    assert not np.isnan(r.mean_phi)
    assert r.dead_walkers >= 0


def test_gamma_zero_walks_into_the_trap(env):
    """The baseline failure mode. If this stops holding, the env got too easy."""
    s = env.reset()
    outs = [run_episode(env, s, max_steps=40, N=32, M=10, gamma=0.0, seed=k).outcome
            for k in range(10)]
    assert Counter(outs)["trap"] >= 8


def test_gamma_one_never_walks_into_the_trap(env):
    """The headline E1 claim, at pilot scale. E1 runs it properly with CI."""
    s = env.reset()
    outs = [run_episode(env, s, max_steps=40, N=32, M=10, gamma=1.0,
                        phi_m=4, phi_h=3, seed=k).outcome for k in range(10)]
    assert Counter(outs)["trap"] == 0
