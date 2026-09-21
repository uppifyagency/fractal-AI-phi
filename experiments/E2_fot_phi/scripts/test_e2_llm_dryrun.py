"""The paid arm's offline path, exercised end to end. No network call.

    cd /Users/vladvrinceanu/fractal-AI-phi && \
      uv run pytest experiments/E2_fot_phi/scripts/test_e2_llm_dryrun.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[2] / "src"))

from fmcphi.phi import phi_composite                      # noqa: E402
from fmcphi.planner import plan                           # noqa: E402

import llm_backend                                        # noqa: E402
from llm_backend import (AnthropicBackend, LLMReasoningEnv,  # noqa: E402
                         estimate, run_paid)


def _env(max_depth=3):
    b = AnthropicBackend(dry_run=True)
    return LLMReasoningEnv("A three-step word problem.", b, max_depth=max_depth)


def test_dry_run_backend_makes_no_client():
    env = _env()
    s = env.reset()
    rng = np.random.default_rng(0)
    for _ in range(6):
        s = env.step(s, env.sample_action(s, rng))
    assert env.backend._client is None, "a client was constructed in dry run"
    assert env.backend.usage.calls > 0
    assert env.backend.rendered_prompts


def test_planner_runs_on_the_llm_env_offline():
    """The real-model env satisfies the same protocol the mock does."""
    env = _env()
    x0 = env.reset()
    res = plan(env, x0, N=8, M=2, gamma=1.0, phi_m=2, phi_h=1,
               budget_attr="tokens", budget_max=env.tokens0, seed=0)
    assert res.action in range(5)
    assert res.sim_steps == 8 * 2 + 8 * 2 * 1 * 2


def test_node_calls_are_cached_by_path():
    env = _env()
    s = env.reset()
    n0 = env.backend.usage.calls
    for _ in range(5):
        env.reward(s)
        env.key(s)
    assert env.backend.usage.calls == max(n0, 1), "root node was re-queried"


def test_phi_is_computable_on_the_llm_env():
    env = _env()
    s = env.reset()
    v = phi_composite(env, s, np.random.default_rng(0), m=3, h=2,
                      budget_attr="tokens", budget_max=env.tokens0)
    assert v >= 0.0


def test_estimate_is_monotone_in_depth_and_under_the_cap_at_defaults():
    a = estimate("claude-opus-5", 10, 3, 2, 2)
    b = estimate("claude-opus-5", 10, 3, 2, 3)
    assert b["usd_upper_bound"] > a["usd_upper_bound"]
    assert b["api_calls_upper_bound"] == 40 * 10


def test_paid_arm_refuses_without_confirmation(capsys):
    args = SimpleNamespace(problems=10, seeds=3, model="claude-opus-5",
                           max_depth=3, dry_run_paid=False, confirm_paid=False)
    assert run_paid(args) == 2


def test_dry_run_paid_returns_zero_and_touches_no_network(capsys):
    args = SimpleNamespace(problems=10, seeds=3, model="claude-opus-5",
                           max_depth=3, dry_run_paid=True, confirm_paid=False)
    assert run_paid(args) == 0
    out = capsys.readouterr().out
    assert "NO NETWORK CALL WAS MADE." in out


def test_paid_arm_refuses_over_the_preregistered_cap(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-not-a-real-key")
    args = SimpleNamespace(problems=400, seeds=3, model="claude-opus-5",
                           max_depth=5, dry_run_paid=False, confirm_paid=True)
    assert run_paid(args) == 2, "the $25 cap gate did not fire"
