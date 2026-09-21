"""E2: Fractal-of-Thought with Phi. Harness.

One FoT episode = repeatedly call `fmcphi.planner.plan` from the current partial
chain of thought, commit the action the swarm voted for, until the chain answers
or dies. Walkers are partial chains, cloning is FMC cloning, R is the process
verifier in `mock_llm.ReasoningEnv.reward`, and Phi is `fmcphi.phi.phi_composite`
with `budget_attr="tokens"` so `phi_slack` maps to remaining context tokens
exactly as the brief specifies.

Arms
----
    fot_nm          gamma = 0, M cycles                   (equal N and M, cheap)
    fot_budget      gamma = 0, M * (1 + m*h) cycles       (the budget reference)
    fot_phi         gamma = 1, M cycles, Phi every tick   (equal budget)
    fot_phi_amort   gamma = 1, more cycles, Phi every k   (equal budget)

Budget matching is exact, not approximate. `plan` charges N*M for the rollouts
plus N*m*h on every tick where Phi is recomputed, i.e.

    sim_steps_per_plan = N*M + [gamma != 0] * N * ceil(M / phi_every) * m * h

`fot_phi` spends the Phi surcharge on every tick, so it can only afford M cycles
against the baseline's M*(1 + m*h). `fot_phi_amort` uses the amortisation knob
of docs/SPEC.md section 4 (`phi_every = k`, stale Phi in between) to buy back
most of the depth at the same price. All three arms after `fot_nm` are matched
to the same sim_steps per decision; `assert_equal_budget` checks it.

Constraint 6 of the task: the verdict is read off the equal-budget arms, never
off the equal-(N, M) pair.

Tokens billed = sim_steps * TOKENS_PER_CALL. One simulator step is one mock
continuation, so sim_steps and tokens are proportional; both are reported.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[2] / "src"))

from fmcphi.planner import plan                       # noqa: E402
from fmcphi.phi import phi_composite                  # noqa: E402
from fmcphi.core import relativize                    # noqa: E402

from mock_llm import ReasoningEnv, make_problem       # noqa: E402

RESULTS = _HERE.parent / "results"

#: Output tokens a single continuation would cost on a real model. Only a
#: constant of proportionality on sim_steps; stated so the paid arm's budget
#: table and the mock's are in the same units.
TOKENS_PER_CALL = 180


# ---------------------------------------------------------------------------
# Arm configuration
# ---------------------------------------------------------------------------

@dataclass
class Arm:
    name: str
    gamma: float
    N: int
    M: int
    phi_m: int = 3
    phi_h: int = 2
    phi_every: int = 1
    alpha: float = 1.0
    beta: float = 1.0

    def plan_kwargs(self, env) -> Dict:
        return dict(
            N=self.N, M=self.M,
            alpha=self.alpha, beta=self.beta, gamma=self.gamma,
            phi_m=self.phi_m, phi_h=self.phi_h, phi_every=self.phi_every,
            budget_attr="tokens" if self.gamma != 0.0 else None,
            budget_max=env.p.tokens0,
        )

    def budget_per_plan(self) -> int:
        """sim_steps one `plan` call will charge. Mirrors planner.plan exactly."""
        cost = self.N * self.M
        if self.gamma != 0.0 and self.phi_every > 0:
            n_phi_ticks = math.ceil(self.M / self.phi_every)
            cost += self.N * n_phi_ticks * self.phi_m * self.phi_h
        return cost


def phi_ladder(M: int, phi_m: int, phi_h: int) -> List[tuple]:
    """(M_a, phi_every) pairs that cost exactly the reference budget.

    Reference budget (in cycles per walker) is M*(1 + m*h): what the gamma = 0
    baseline gets. A Phi arm with M_a cycles refreshing Phi every k ticks costs
    M_a + ceil(M_a/k)*m*h. Solving that for equality gives one arm per number of
    Phi refreshes, which is a one-dimensional sweep of *how to spend a fixed
    budget* between rollout depth and cone-estimate freshness. Returned deepest
    first is not useful; returned densest first so `fot_phi` (the pre-registered
    naive arm, Phi every tick) comes out at index 0.
    """
    ref = M * (1 + phi_m * phi_h)
    mh = phi_m * phi_h
    out = []
    for n_ticks in range(ref, 0, -1):
        M_a = ref - n_ticks * mh
        if M_a < 1:
            continue
        k = None
        for kk in range(1, M_a + 1):
            if math.ceil(M_a / kk) == n_ticks:
                k = kk
                break
        if k is not None:
            out.append((M_a, k))
    return out


def make_arms(N: int, M: int, phi_m: int, phi_h: int) -> List[Arm]:
    """The equal-(N,M) reference, the budget baseline, and the Phi ladder."""
    ref = M * (1 + phi_m * phi_h)
    arms = [
        Arm("fot_nm", 0.0, N, M, phi_m, phi_h, 1),
        Arm("fot_budget", 0.0, N, ref, phi_m, phi_h, 1),
    ]
    ladder = phi_ladder(M, phi_m, phi_h)
    for i, (M_a, k) in enumerate(ladder):
        name = "fot_phi" if i == 0 else f"fot_phi_M{M_a}k{k}"
        arms.append(Arm(name, 1.0, N, M_a, phi_m, phi_h, k))
    return arms


def assert_equal_budget(arms: List[Arm]) -> int:
    """Every arm but `fot_nm` must charge the same sim_steps per decision."""
    budgets = {a.name: a.budget_per_plan() for a in arms if a.name != "fot_nm"}
    vals = set(budgets.values())
    if len(vals) != 1:
        raise AssertionError(f"arms are not budget matched: {budgets}")
    return vals.pop()


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------

def run_chain_episode(env: ReasoningEnv, arm: Arm, seed: int,
                      max_steps: int = 12) -> Dict:
    """One FoT episode. Returns a scored record.

    Not `planner.run_episode`, because that one does not hand back the final
    state and E2 needs the answer string to score against ground truth. The
    inner call is the shared, tested `planner.plan`, so the gamma = 0 arm is
    still canonical FMC.
    """
    rng = np.random.default_rng(seed)
    s = env.reset()
    sim_steps = 0
    b_effs: List[float] = []
    phis: List[float] = []
    wrong_after_1 = None
    actions: List[int] = []

    for t in range(max_steps):
        res = plan(env, s, rng=rng, **arm.plan_kwargs(env))
        sim_steps += res.sim_steps
        b_effs.append(res.b_eff)
        if not np.isnan(res.mean_phi):
            phis.append(res.mean_phi)
        actions.append(int(res.action))
        s = env.step(s, res.action)
        if t == 0:
            wrong_after_1 = bool(s.branch != 0)
        if env.is_terminal(s):
            break

    cause = env.death_cause(s)
    outcome = "answered" if s.answered else (cause or "timeout")
    return {
        "pid": env.p.pid,
        "tier": env.p.tier,
        "arm": arm.name,
        "gamma": arm.gamma,
        "seed": seed,
        "correct": bool(env.is_correct(s)),
        "answered": bool(s.answered),
        "outcome": outcome,
        "answer": s.answer,
        "steps": len(actions),
        "sim_steps": int(sim_steps),
        "tokens": int(sim_steps * TOKENS_PER_CALL),
        "mean_b_eff": float(np.mean(b_effs)),
        "mean_phi": float(np.mean(phis)) if phis else None,
        "flagged": bool(s.flagged),
        "final_branch": int(s.branch),
        "final_stage": int(s.stage),
        "tokens_left": float(s.tokens),
        "contradictions": int(s.contradictions),
        "wrong_after_step1": wrong_after_1,
        "trap_at_0": bool(env.p.trap_at_0),
        "actions": actions,
    }


def run_grid(tier: str, arms: List[Arm], n_problems: int, n_seeds: int,
             phi_signal: float, p_trap: float, p_detect: float,
             n_stages: int, tokens0: float,
             max_steps: int = 12, verbose: bool = True,
             pid_offset: int = 0) -> List[Dict]:
    """Every arm on every (problem, seed). Same problems for every arm.

    `pid_offset` selects a disjoint block of problems, used to replicate a
    finding on a held-out set rather than on the set it was found in.
    """
    recs: List[Dict] = []
    t0 = time.time()
    for pid in range(pid_offset, pid_offset + n_problems):
        prob = make_problem(pid, tier=tier, p_trap=p_trap,
                            phi_signal=phi_signal, p_detect=p_detect,
                            n_stages=n_stages, tokens0=tokens0)
        for arm in arms:
            env = ReasoningEnv(prob)
            for seed in range(n_seeds):
                recs.append(run_chain_episode(env, arm, seed=1000 * pid + seed,
                                              max_steps=max_steps))
        if verbose and (pid - pid_offset + 1) % 10 == 0:
            print(f"    problem {pid - pid_offset + 1}/{n_problems}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return recs


# ---------------------------------------------------------------------------
# Phi flatness diagnostic (the direct test of P3 / SPEC section 5)
# ---------------------------------------------------------------------------

def phi_action_spread(tier: str, n_problems: int, n_decisions: int = 64,
                      phi_m: int = 6, phi_h: int = 2, seed: int = 0,
                      phi_signal: float = 0.75, p_trap: float = 0.6,
                      p_detect: float = 0.6,
                      n_stages: int = 3, tokens0: float = 8.0) -> Dict:
    """How much Phi varies across the sibling continuations of one decision.

    This is the decisive measurement for P3, and it is sharper than "is Phi
    constant over the whole cloud". What a gamma > 0 planner can act on is the
    *spread of Phi between the options a decision has to choose between*. If the
    three candidate continuations of a state all score the same Phi, the factor
    cannot move the decision, whatever its absolute level.

    docs/SPEC.md section 5 predicts zero spread wherever the cone never narrows.
    Reported as the mean within-decision std of Phi and as the mean std of
    `relativize(Phi)` over the same triples. The second matters on its own:
    `relativize` is scale free, so any *non-zero* spread, however tiny, is
    rescaled to full strength. Only an exactly constant Phi actually vanishes.
    """
    rng = np.random.default_rng(seed)
    stds, rel_stds, levels, zeros = [], [], [], []
    for pid in range(n_problems):
        prob = make_problem(pid, tier=tier, p_trap=p_trap,
                            phi_signal=phi_signal, p_detect=p_detect,
                            n_stages=n_stages, tokens0=tokens0)
        env = ReasoningEnv(prob)
        for _ in range(n_decisions):
            s = env.reset()
            for _ in range(int(rng.integers(0, max(1, prob.n_stages)))):
                s = env.step(s, env.sample_action(s, rng))
            if not env.viable(s):
                continue
            sibs = [env.step(env.clone_state(s), a) for a in (0, 1, 2)]
            ph = np.array([
                phi_composite(env, c, rng, m=phi_m, h=phi_h,
                              budget_attr="tokens", budget_max=prob.tokens0)
                for c in sibs
            ])
            stds.append(float(ph.std()))
            rel_stds.append(float(relativize(ph).std()))
            levels.append(float(ph.mean()))
            zeros.append(float((ph <= 0.0).mean()))
    stds_a = np.array(stds)
    return {
        "tier": tier,
        "n_problems": n_problems,
        "n_decisions_sampled": len(stds),
        "phi_m": phi_m,
        "phi_h": phi_h,
        "phi_level_mean": float(np.mean(levels)),
        "phi_std_within_decision": float(stds_a.mean()),
        "frac_decisions_with_zero_spread": float((stds_a == 0.0).mean()),
        "relativized_phi_std_within_decision": float(np.mean(rel_stds)),
        "frac_phi_exactly_zero": float(np.mean(zeros)),
    }


def phi_by_node_type(n_problems: int = 30, n_samples: int = 60,
                     phi_m: int = 6, phi_h: int = 2, seed: int = 0,
                     phi_signal: float = 0.75, p_trap: float = 0.6,
                     p_detect: float = 0.6, n_stages: int = 3,
                     tokens0: float = 8.0) -> Dict:
    """Phi split by what the node actually is. The mechanism check.

    `b0` = still on the correct line, `narrow` = committed to a wrong line whose
    cone collapsed, `decoy` = on a wrong line that still looks option rich,
    `answered` = finished. If `narrow` does not score below `b0`, the mock
    carries no cone signal and nothing downstream means anything.
    """
    rng = np.random.default_rng(seed)
    buckets: Dict[str, List[float]] = {"b0": [], "narrow": [], "decoy": [],
                                       "answered": []}
    for pid in range(n_problems):
        prob = make_problem(pid, p_trap=p_trap, phi_signal=phi_signal,
                            p_detect=p_detect, n_stages=n_stages,
                            tokens0=tokens0)
        env = ReasoningEnv(prob)
        for _ in range(n_samples):
            s = env.reset()
            for _ in range(int(rng.integers(0, 3))):
                s = env.step(s, env.sample_action(s, rng))
            v = phi_composite(env, s, rng, m=phi_m, h=phi_h,
                              budget_attr="tokens", budget_max=prob.tokens0)
            if s.answered:
                k = "answered"
            elif s.branch == 0:
                k = "b0"
            elif prob.narrow.get(s.branch, False):
                k = "narrow"
            else:
                k = "decoy"
            buckets[k].append(float(v))
    return {k: {"n": len(v), "mean": float(np.mean(v)) if v else None,
                "std": float(np.std(v)) if v else None}
            for k, v in buckets.items()}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save(name: str, payload: Dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  wrote {path}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="E2 Fractal-of-Thought with Phi")
    ap.add_argument("--backend", choices=["mock", "anthropic"], default="mock",
                    help="mock is free and deterministic; anthropic costs money")
    ap.add_argument("--tier", choices=["multi", "single"], default="multi")
    ap.add_argument("--problems", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--phi-m", type=int, default=3)
    ap.add_argument("--phi-h", type=int, default=2)
    ap.add_argument("--phi-signal", type=float, default=0.75)
    ap.add_argument("--p-trap", type=float, default=0.6)
    ap.add_argument("--p-detect", type=float, default=0.6)
    ap.add_argument("--n-stages", type=int, default=3)
    ap.add_argument("--tokens0", type=float, default=8.0)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="claude-opus-5",
                    help="paid arm only; see llm_backend.PRICING")
    ap.add_argument("--max-depth", type=int, default=3,
                    help="paid arm only; caps the node cache and the bill")
    ap.add_argument("--confirm-paid", action="store_true",
                    help="required for --backend anthropic; without it the "
                         "paid arm refuses to run")
    ap.add_argument("--dry-run-paid", action="store_true",
                    help="with --backend anthropic, print the call plan and "
                         "cost estimate and exit without calling anything")
    args = ap.parse_args(argv)

    if args.backend == "anthropic":
        import llm_backend
        return llm_backend.run_paid(args)

    arms = make_arms(args.N, args.M, args.phi_m, args.phi_h)
    budget = assert_equal_budget(arms)
    print(f"E2 mock run: tier={args.tier} problems={args.problems} "
          f"seeds={args.seeds}")
    for a in arms:
        print(f"    {a.name:14s} gamma={a.gamma} M={a.M} phi_every={a.phi_every} "
              f"sim_steps/decision={a.budget_per_plan()}")
    print(f"    matched budget = {budget} sim_steps/decision "
          f"= {budget * TOKENS_PER_CALL} tokens")
    t0 = time.time()
    recs = run_grid(args.tier, arms, args.problems, args.seeds,
                    args.phi_signal, args.p_trap, args.p_detect,
                    args.n_stages, args.tokens0, max_steps=args.max_steps)
    payload = {
        "run": args.out or f"fot_{args.tier}",
        "backend": "mock",
        "config": vars(args),
        "arms": [asdict(a) for a in arms],
        "tokens_per_call": TOKENS_PER_CALL,
        "matched_budget_sim_steps_per_decision": budget,
        "wall_clock_s": round(time.time() - t0, 1),
        "records": recs,
    }
    save(args.out or f"fot_{args.tier}", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
