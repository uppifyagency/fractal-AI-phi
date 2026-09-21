"""Unit tests for the E3 instrument.

Run with:
    uv run pytest experiments/E3_coding_agent/tests -q

They are outside the project's `testpaths = ["tests"]`, so `uv run pytest -q`
at the root is unaffected.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import fmcphi.core as core
from fmcphi.envs.base import Environment
from fmcphi.phi import phi_cone, phi_composite, phi_slack
from fmcphi.planner import plan

from phi_repo import (GitProbe, classify_irreversibility, phi_irreversibility,
                      phi_repo, phi_repo_deep, phi_slack_multi,
                      phi_viable_actions_repo)
from repo_env import (ACTIONS, DELETE_TEST, FORCE_PUSH, PATCH, REVERT, TEST,
                      RepoEnv, RepoState)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Component 3: the static irreversibility classifier
# ---------------------------------------------------------------------------

def test_reversible_commands_are_not_flagged():
    for cmd in ("pytest -q", "git status", "git add -A",
                "git commit -m 'fix'", "git push origin feature",
                "ruff check .", "cat src/main.py"):
        assert classify_irreversibility(cmd) == [], cmd


def test_force_push_is_flagged():
    ops = classify_irreversibility("git push --force origin main")
    assert [o.name for o in ops] == ["force-push"]


def test_force_with_lease_scores_strictly_better_than_bare_force():
    """The safer operation must not be punished harder. Regression test."""
    lease = phi_irreversibility(
        classify_irreversibility("git push --force-with-lease origin main"))
    bare = phi_irreversibility(
        classify_irreversibility("git push --force origin main"))
    assert lease > bare


def test_publication_amplifies_history_rewriting():
    local = phi_irreversibility(
        classify_irreversibility("git commit --amend --no-edit", published=False))
    shared = phi_irreversibility(
        classify_irreversibility("git commit --amend --no-edit", published=True))
    assert shared < local


def test_publication_does_not_amplify_a_local_rm():
    local = phi_irreversibility(classify_irreversibility("rm -rf build", published=False))
    shared = phi_irreversibility(classify_irreversibility("rm -rf build", published=True))
    assert local == shared


@pytest.mark.parametrize("cmd,name", [
    ("rm -rf src/legacy", "rm-rf"),
    ("psql -c 'DROP TABLE users'", "destructive-sql"),
    ("psql -c 'ALTER TABLE users DROP COLUMN email'", "drop-column"),
    ("npm publish", "publish-package"),
    ("terraform apply -auto-approve", "terraform-apply"),
    ("git reset --hard origin/main", "reset-hard"),
    ("git push origin --delete feature", "delete-remote-branch"),
    ("git filter-branch --tree-filter 'rm -f secrets' HEAD", "history-rewrite"),
])
def test_detector_catalogue(cmd, name):
    assert name in [o.name for o in classify_irreversibility(cmd)]


def test_diff_side_detection_of_a_deleted_test_file():
    diff = ("diff --git a/tests/test_core.py b/tests/test_core.py\n"
            "deleted file mode 100644\n"
            "--- a/tests/test_core.py\n"
            "+++ /dev/null\n"
            "-def test_mean_empty():\n")
    ops = classify_irreversibility(command="git rm tests/test_core.py", diff=diff)
    assert "delete-test-file" in [o.name for o in ops]


def test_phi_irreversibility_is_in_unit_interval_and_monotone():
    from phi_repo import IrreversibleOp
    assert phi_irreversibility([]) == 1.0
    prev = 1.0
    for w in (0.1, 0.5, 1.0, 3.0):
        v = phi_irreversibility([IrreversibleOp("x", w, "")])
        assert 0.0 < v <= 1.0
        assert v < prev
        prev = v


# ---------------------------------------------------------------------------
# Component 2: budget slack
# ---------------------------------------------------------------------------

def test_slack_multi_empty_is_neutral():
    assert phi_slack_multi({}) == 1.0


def test_slack_multi_min_is_the_binding_constraint():
    b = {"money": (90.0, 100.0), "context": (2.0, 100.0)}
    assert phi_slack_multi(b, mode="min") == pytest.approx(0.02)
    assert phi_slack_multi(b, mode="prod") == pytest.approx(0.9 * 0.02)


def test_slack_multi_clamps():
    assert phi_slack_multi({"t": (-5.0, 10.0)}) == 0.0
    assert phi_slack_multi({"t": (50.0, 10.0)}) == 1.0
    assert phi_slack_multi({"t": (1.0, 0.0)}) == 1.0


def test_repo_state_slack_matches_fmcphi_phi_slack():
    """RepoState.slack must be what fmcphi.phi.phi_slack would compute."""
    s = RepoState("bug", tokens=3.0, tokens0=12.0)
    assert s.slack == pytest.approx(phi_slack(s, "tokens", 12.0))


def test_slack_irrev_is_slack_times_the_penalty():
    s = RepoState("bug", tokens=6.0, tokens0=12.0, irrev=2, lam=0.7)
    assert s.slack_irrev == pytest.approx(0.5 * np.exp(-1.4))


# ---------------------------------------------------------------------------
# Component 1 and the composite
# ---------------------------------------------------------------------------

def test_phi_viable_actions_repo_is_normalised():
    env = RepoEnv()
    v = phi_viable_actions_repo(env, env.state("bug"))
    assert 0.0 <= v <= 1.0


def test_phi_is_zero_on_a_dead_state():
    env = RepoEnv()
    dead = env.state("deleted")
    assert not env.viable(dead)
    assert phi_viable_actions_repo(env, dead) == 0.0
    assert phi_repo(env, dead) == 0.0
    assert phi_repo_deep(env, dead, np.random.default_rng(0)) == 0.0


def test_phi_repo_is_the_product_of_its_three_components():
    env = RepoEnv()
    s = env.state("bug", tokens=3.0)
    ops = classify_irreversibility("git push --force origin main")
    budgets = {"tokens": (3.0, 12.0)}
    expected = (phi_viable_actions_repo(env, s)
                * phi_slack_multi(budgets)
                * phi_irreversibility(ops))
    assert phi_repo(env, s, budgets=budgets, ops=ops) == pytest.approx(expected)


def test_phi_repo_in_unit_interval():
    env = RepoEnv()
    for node in ("bug", "part1", "fixed", "refactored", "wrapped", "broken"):
        v = phi_repo(env, env.state(node), budgets={"tokens": (6.0, 12.0)})
        assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# RepoEnv
# ---------------------------------------------------------------------------

def test_repo_env_satisfies_the_environment_protocol():
    assert isinstance(RepoEnv(), Environment)


def test_measured_node_table_is_what_drives_the_reward():
    data = json.loads((FIXTURES / "nodes.json").read_text())
    env = RepoEnv()
    # `deleted` really did pass every test it still had, in a real pytest run.
    d = data["nodes"]["deleted"]
    assert d["tests_passed"] == d["tests_total"] and d["build_ok"]
    assert not d["acceptance_ok"]
    # and the visible reward therefore ties it with the real fix.
    assert env.reward(env.state("deleted")) == pytest.approx(
        env.reward(env.state("fixed")))


def test_the_trap_is_absorbing_and_that_is_derived_not_asserted():
    env = RepoEnv()
    s = env.state("bug")
    dead = env.step(s, DELETE_TEST)
    assert dead.node == "deleted"
    assert not env.viable(dead)
    assert env.death_cause(dead) == "trap"
    for a in ACTIONS:
        assert env.step(dead, a).node == "deleted"


def test_the_trap_is_one_step_away_and_the_fix_is_two():
    env = RepoEnv()
    s = env.state("bug")
    assert env.is_goal(env.step(env.step(s, PATCH), PATCH))
    assert not env.is_goal(env.step(s, PATCH))
    assert env.step(s, DELETE_TEST).node == "deleted"


def test_force_push_removes_the_revert_option():
    env = RepoEnv()
    s = env.step(env.state("bug"), PATCH)          # part1, history intact
    assert env.step(s, REVERT).node == "bug"
    pushed = env.step(s, FORCE_PUSH)
    assert pushed.pushed and pushed.irrev == 1
    assert env.step(pushed, REVERT).node == "part1"  # nothing to revert to


def test_a_broken_build_is_recoverable_only_while_history_is_intact():
    env = RepoEnv()
    assert env.viable(env.state("broken", pushed=False))
    assert not env.viable(env.state("broken", pushed=True))


def test_allow_irreversible_false_makes_force_push_a_no_op():
    env = RepoEnv(allow_irreversible=False)
    s = env.step(env.state("bug"), FORCE_PUSH)
    assert not s.pushed and s.irrev == 0
    assert env.viable(env.state("broken", pushed=False))


def test_budget_exhaustion_kills():
    env = RepoEnv(tokens=1.0)
    s = env.step(env.state("bug", tokens=1.0), PATCH)
    assert s.tokens <= 0.0
    assert not env.viable(s)
    assert env.death_cause(s) == "budget"


def test_key_mode_controls_state_identity():
    full = RepoEnv(key_mode="full")
    content = RepoEnv(key_mode="content")
    a, b = full.state("bug"), full.state("bug", pushed=True)
    assert full.key(a) != full.key(b)
    assert content.key(a) == content.key(b)
    with pytest.raises(ValueError):
        RepoEnv(key_mode="nonsense")


def test_clone_state_is_a_real_copy():
    env = RepoEnv()
    s = env.state("bug")
    c = env.clone_state(s)
    c.tokens = 0.0
    assert s.tokens != 0.0


# ---------------------------------------------------------------------------
# SPEC section 4 invariants, checked on this environment
# ---------------------------------------------------------------------------

def test_gamma_zero_reproduces_canonical_fmc_on_repo_env():
    """SPEC invariant 1, on E3's environment."""
    env = RepoEnv()
    for seed in (0, 1, 2, 3, 4):
        got = plan(env, env.reset(), N=16, M=5, gamma=0.0, seed=seed).action
        want = core.plan(env, env.reset(), N=16, M=5, seed=seed)
        assert got == want, seed


def test_sim_steps_counts_the_phi_rollouts():
    """SPEC invariant 2: arms are comparable at equal budget, not equal (N, M)."""
    env = RepoEnv()
    off = plan(env, env.reset(), N=8, M=4, gamma=0.0, seed=0)
    on = plan(env, env.reset(), N=8, M=4, gamma=1.0, phi_m=3, phi_h=2, seed=0)
    assert off.sim_steps == 8 * 4
    assert on.sim_steps == 8 * 4 * (1 + 3 * 2)


def test_phi_composite_through_budget_attr_equals_the_explicit_product():
    """The slack_irrev property is how components 2 and 3 reach the stock planner."""
    env = RepoEnv(lam=0.7)
    s = env.state("bug", tokens=6.0, irrev=1)
    cone = phi_cone(env, s, np.random.default_rng(3), m=4, h=3)
    got = phi_composite(env, s, np.random.default_rng(3), m=4, h=3,
                        budget_attr="slack_irrev", budget_max=1.0)
    assert got == pytest.approx(cone * s.slack_irrev)


def test_phi_cone_is_bounded_by_m():
    env = RepoEnv()
    rng = np.random.default_rng(0)
    for node in ("bug", "part1", "fixed", "broken"):
        v = phi_cone(env, env.state(node), rng, m=6, h=3)
        assert 0.0 <= v <= 6.0


# ---------------------------------------------------------------------------
# The fixture repository is a real repository
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (FIXTURES / "nodes.json").exists(),
                    reason="run scripts/build_fixture.py first")
def test_published_is_a_real_git_fact():
    data = json.loads((FIXTURES / "nodes.json").read_text())
    repo = Path(data["repo"])
    if not repo.exists():
        pytest.skip("fixture working repo not materialised; rebuild it")
    probe = GitProbe(repo)
    for name, rec in data["nodes"].items():
        assert probe.is_published(rec["sha"]) == rec["published"], name
    # and at least one of each, or the comparison would be vacuous
    flags = {rec["published"] for rec in data["nodes"].values()}
    assert flags == {True, False}


@pytest.mark.skipif(not (FIXTURES / "nodes.json").exists(),
                    reason="run scripts/build_fixture.py first")
def test_every_node_was_measured_by_a_real_test_run():
    data = json.loads((FIXTURES / "nodes.json").read_text())
    for name, rec in data["nodes"].items():
        assert len(rec["sha"]) == 40, name
        assert rec["tests_total"] >= 1, name
        assert "passed" in rec["pytest_tail"] or "error" in rec["pytest_tail"], name
    assert data["nodes"]["broken"]["build_ok"] is False
    assert data["nodes"]["fixed"]["acceptance_ok"] is True
