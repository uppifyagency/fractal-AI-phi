"""Properties the Phi estimator must satisfy for the experiment ladder to mean anything."""

import numpy as np
import pytest

from fmcphi import phi_cone, phi_slack, phi_viable_actions, virtual_reward_phi, kill_dead
from fmcphi.envs.trapgrid import TrapGrid


def cell(env, x, y, fuel=16.0):
    s = env.reset()
    s.x, s.y, s.fuel = x, y, fuel
    return s


@pytest.fixture
def env():
    return TrapGrid()


def test_phi_is_zero_on_a_trap(env):
    rng = np.random.default_rng(0)
    assert phi_cone(env, cell(env, 5, 0), rng, m=16, h=3) == 0.0


def test_phi_is_zero_without_fuel(env):
    rng = np.random.default_rng(0)
    assert phi_cone(env, cell(env, 5, 2, fuel=0.0), rng, m=16, h=3) == 0.0


def test_phi_increases_with_distance_from_the_trap_row(env):
    """The gradient the layer steers along. Averaged over seeds: it is a MC estimate."""
    rng = np.random.default_rng(0)
    near = np.mean([phi_cone(env, cell(env, 5, 1), rng, m=32, h=3) for _ in range(20)])
    far = np.mean([phi_cone(env, cell(env, 5, 2), rng, m=32, h=3) for _ in range(20)])
    assert far > near


def test_phi_is_bounded_by_the_fanout(env):
    rng = np.random.default_rng(0)
    for _ in range(50):
        v = phi_cone(env, cell(env, 5, 2), rng, m=8, h=3)
        assert 0.0 <= v <= 8.0


def test_survival_weighting_lowers_phi_near_danger(env):
    rng = np.random.default_rng(1)
    weighted = np.mean([phi_cone(env, cell(env, 5, 1), rng, m=32, h=3,
                                 weight_by_survival=True) for _ in range(20)])
    raw = np.mean([phi_cone(env, cell(env, 5, 1), rng, m=32, h=3,
                            weight_by_survival=False) for _ in range(20)])
    assert weighted < raw


def test_corner_is_option_poor(env):
    """Wall clamping lowers Phi too. The layer measures futures, not danger."""
    rng = np.random.default_rng(0)
    corner = np.mean([phi_cone(env, cell(env, 0, 3), rng, m=32, h=3) for _ in range(20)])
    open_ = np.mean([phi_cone(env, cell(env, 5, 2), rng, m=32, h=3) for _ in range(20)])
    assert corner < open_


def test_viable_actions_counts_non_lethal_moves(env):
    assert phi_viable_actions(env, cell(env, 5, 0)) == 0.0   # already dead
    assert phi_viable_actions(env, cell(env, 5, 1)) == 4.0   # down is lethal
    assert phi_viable_actions(env, cell(env, 5, 2)) == 5.0   # all safe


def test_slack_is_normalised(env):
    assert phi_slack(cell(env, 5, 2, fuel=8.0), "fuel", 16.0) == 0.5
    assert phi_slack(cell(env, 5, 2, fuel=99.0), "fuel", 16.0) == 1.0
    assert phi_slack(cell(env, 5, 2, fuel=-1.0), "fuel", 16.0) == 0.0


def test_gamma_zero_reproduces_the_two_factor_virtual_reward():
    from fmcphi import virtual_reward
    rng = np.random.default_rng(0)
    r = rng.normal(size=16)
    obs = rng.normal(size=(16, 2))
    partners = rng.permutation(16)
    phis = rng.random(16) * 5
    a = virtual_reward(r, obs, partners, alpha=1.0, beta=1.0)
    b = virtual_reward_phi(r, obs, partners, phis, alpha=1.0, beta=1.0, gamma=0.0)
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_kill_dead_zeroes_only_zero_phi_walkers():
    vr = np.array([1.0, 2.0, 3.0, 4.0])
    phis = np.array([0.0, 0.5, 0.0, 2.0])
    np.testing.assert_array_equal(kill_dead(vr, phis), [0.0, 2.0, 0.0, 4.0])
