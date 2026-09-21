"""E2 mock-LLM backend: a reasoning environment over chains of thought.

Why a mock at all
-----------------
The brief asks for a Fractal-of-Thought harness whose reward comes from a real
judge instead of `1/(1+|cot|)`, plus a Phi term over reasoning states, plus a
deterministic free backend so the whole thing runs in CI. This module is that
backend. It is *not* a language model: it is an explicit generative model of the
only structure the Phi layer can possibly react to, namely

  1. a finite context budget (phi_slack),
  2. irreversible commitment to a wrong line of reasoning (cone narrowing),
  3. a process verifier that catches only *some* wrong branches (the ones with
     a checkable slip in them) and is blind to the rest, so R carries real
     signal but not enough of it.

Everything in the state is ground-truth free except `Problem.answer`, which is
used only by the scorer in `fot_phi.py`, never by `reward`, `viable`, `key` or
`sample_action`. That separation is asserted in `test_e2_mock.py`.

State of one walker
-------------------
A partial chain of thought:

    ChainState(stage, branch, branch_stage, tokens, contradictions,
               answered, answer)

`stage` is how many reasoning steps have been written, `branch` is which line of
reasoning the chain is on (0 = the correct one, but nothing in the environment
knows that), `tokens` is the remaining context budget.

Action set (5 labels, integers, so `fmcphi.core.decide` works unchanged)
-----------------------------------------------------------------------
    0,1,2  take the mock LLM's 1st / 2nd / 3rd ranked continuation
    3      BACKTRACK: revise the last step (costs more context)
    4      ANSWER: emit the final answer from the current branch

`sample_action` is the mock LLM's own proposal distribution: mass concentrated
on the top-ranked continuation. The *trap* is that with probability `p_trap` the
top-ranked continuation at stage 0 is not the correct one, so the model's own
likelihood pushes it onto a wrong branch immediately. This is the reasoning
analogue of TrapGrid's short lethal route.

Cone geometry (this is the load-bearing modelling assumption)
-------------------------------------------------------------
On branch 0 the three candidates lead to three *distinct* nodes, so the cone is
wide. On a wrong branch flagged `narrow`, all three candidates lead to the *same*
node: they are paraphrases of one already-committed line of reasoning, so the
cone does not widen and Phi collapses toward 1.0.

A wrong branch is *not* shorter than the correct one. Both reach a forced answer
at `stage == n_stages`, so the process verifier sees identical progress on both
and cannot break the tie: the only thing that separates them is the shape of
their cones. This is deliberate. If wrong branches were shorter, R alone would
solve the task and the whole experiment would be a length proxy again, which is
the exact defect the brief says to remove.

Two independent channels
------------------------
Each wrong branch draws, independently:

  `detectable` with probability `p_detect`  -> the process verifier flags it and
      R penalises it. This is the signal the gamma = 0 baseline runs on, and it
      is why the baseline is well above chance.
  `narrow`     with probability `phi_signal` -> the cone collapses on it. This
      is the only signal Phi adds.

Because the draws are independent, Phi's incremental value can only come from
branches that are narrow *and* undetectable. If the two were the same draw the
experiment would be circular.

The fraction of wrong branches that are narrow is `phi_signal`. At
`phi_signal = 0` a wrong branch looks exactly as option-rich as the correct one,
so Phi carries no information about correctness. That arm is the falsification
control: if Phi-on still beats Phi-off at `phi_signal = 0` and equal budget, the
gain is a budget artifact and not the Phi layer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Action set
# ---------------------------------------------------------------------------

CAND0, CAND1, CAND2, BACKTRACK, ANSWER = 0, 1, 2, 3, 4
ACTIONS: Tuple[int, ...] = (CAND0, CAND1, CAND2, BACKTRACK, ANSWER)
N_CAND = 3

ACTION_NAMES = {0: "cand0", 1: "cand1", 2: "cand2", 3: "backtrack", 4: "answer"}

# Mock LLM proposal distribution over the 5 actions. Mass on the top-ranked
# continuation; ANSWER is common enough that a short wrong branch is tempting.
PROPOSAL = np.array([0.42, 0.20, 0.13, 0.10, 0.15], dtype=np.float64)
PROPOSAL = PROPOSAL / PROPOSAL.sum()


def _h(*parts) -> int:
    """Deterministic 64-bit hash of the parts, stable across processes."""
    s = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big")


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    """One reasoning problem as a branching structure over chains of thought."""

    pid: int
    tier: str                    # "multi" | "single"
    n_stages: int                # stages on the correct line of reasoning
    rank_correct: List[int]      # per stage: rank of the correct continuation
    narrow: Dict[int, bool]      # branch id -> is the wrong branch cone-narrowing
    detectable: Dict[int, bool]  # branch id -> does the verifier catch the slip
    tokens0: float               # context budget
    step_cost: float
    backtrack_cost: float
    answer_cost: float
    max_contradictions: int
    answer: str = "GT"           # ground truth key. SCORER ONLY.

    @property
    def trap_at_0(self) -> bool:
        """True iff the mock LLM's top continuation at stage 0 is wrong."""
        return self.rank_correct[0] != 0


def branch_id(stage: int, slot: int, parent: int = 0) -> int:
    """Stable non-zero id for the wrong branch entered at (stage, slot)."""
    return 1 + (_h("br", parent, stage, slot) % 100000)


def make_problem(
    pid: int,
    tier: str = "multi",
    p_trap: float = 0.6,
    phi_signal: float = 0.75,
    p_detect: float = 0.6,
    n_stages: int = 3,
    tokens0: float = 8.0,
) -> Problem:
    """Deterministic problem generator. Same pid + params -> same problem."""
    rng = np.random.default_rng(_h("prob", pid, tier, n_stages) % (2**32))

    if tier == "single":
        # P3 control: one reasoning step, no contradiction death, and no
        # context cost at all, so `phi_slack` is exactly 1.0 and the only
        # structure left in Phi is the branching of the cone. docs/SPEC.md
        # section 5 by construction.
        return Problem(
            pid=pid, tier="single", n_stages=1,
            rank_correct=[int(rng.integers(0, N_CAND))],
            narrow={}, detectable={}, tokens0=1.0,
            step_cost=0.0, backtrack_cost=0.0, answer_cost=0.0,
            max_contradictions=10**9,
        )

    rank = []
    for k in range(n_stages):
        if k == 0:
            trap = rng.random() < p_trap
            rank.append(int(rng.integers(1, N_CAND)) if trap else 0)
        else:
            # Later stages are easier: the correct continuation is usually top.
            rank.append(int(rng.integers(0, N_CAND)) if rng.random() < 0.35 else 0)

    narrow: Dict[int, bool] = {}
    detectable: Dict[int, bool] = {}
    for k in range(n_stages):
        for slot in range(N_CAND):
            if slot == rank[k]:
                continue
            bid = branch_id(k, slot)
            narrow[bid] = bool(rng.random() < phi_signal)
            detectable[bid] = bool(rng.random() < p_detect)

    return Problem(
        pid=pid, tier="multi", n_stages=n_stages, rank_correct=rank,
        narrow=narrow, detectable=detectable, tokens0=tokens0,
        step_cost=1.0, backtrack_cost=2.0, answer_cost=1.0,
        max_contradictions=2,
    )


# ---------------------------------------------------------------------------
# Walker state
# ---------------------------------------------------------------------------

@dataclass
class ChainState:
    stage: int
    branch: int
    branch_stage: int          # stage at which the chain left branch 0 (-1 if on it)
    tokens: float
    contradictions: int
    flagged: bool              # the verifier caught a slip on the current branch
    answered: bool
    answer: Optional[str]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ReasoningEnv:
    """fmcphi Environment over partial chains of thought.

    Implements actions/clone_state/step/observe/reward/sample_action/viable/key,
    so `fmcphi.planner.plan` and `fmcphi.phi.phi_composite` run on it unchanged
    and the gamma = 0 arm is canonical FMC bit for bit.
    """

    def __init__(self, problem: Problem, judge: str = "process"):
        self.p = problem
        self.judge = judge
        self._emb_cache: Dict[int, np.ndarray] = {}

    # -- construction --------------------------------------------------------

    def reset(self) -> ChainState:
        return ChainState(stage=0, branch=0, branch_stage=-1,
                          tokens=self.p.tokens0, contradictions=0,
                          flagged=False, answered=False, answer=None)

    # -- Environment protocol ------------------------------------------------

    def actions(self):
        return ACTIONS

    def clone_state(self, s: ChainState) -> ChainState:
        return ChainState(s.stage, s.branch, s.branch_stage, s.tokens,
                          s.contradictions, s.flagged, s.answered, s.answer)

    def step(self, s: ChainState, a: int) -> ChainState:
        p = self.p
        n = self.clone_state(s)
        if n.answered:
            return n                      # absorbing terminal, no further cost
        if n.tokens <= 0.0 or n.contradictions >= p.max_contradictions:
            return n                      # already dead

        if a == ANSWER:
            n.tokens -= p.answer_cost
            n.answered = True
            n.answer = self._answer_of(n)
            return n

        if a == BACKTRACK:
            n.tokens -= p.backtrack_cost
            if n.stage == 0:
                if p.tier != "single":
                    n.contradictions += 1  # revising nothing = incoherent chain
            else:
                n.stage -= 1
                if n.branch != 0 and n.stage <= n.branch_stage:
                    n.branch = 0
                    n.branch_stage = -1
                    n.flagged = False     # the flagged step has been revised
            return n

        # a in {0, 1, 2}: take the a-th ranked continuation.
        n.tokens -= p.step_cost
        if n.branch == 0:
            k = min(n.stage, p.n_stages - 1)
            if a == p.rank_correct[k]:
                n.stage += 1              # stay on the correct line
            else:
                n.branch = branch_id(n.stage, a)
                n.branch_stage = n.stage
                n.flagged = p.detectable.get(n.branch, False)
                n.stage += 1
        elif p.narrow.get(n.branch, False):
            # Committed line of reasoning: every paraphrase lands on the same
            # node, so the cone does not widen.
            n.stage += 1
        else:
            # Decoy: a wrong branch that still looks option-rich.
            n.branch = branch_id(n.stage, a, parent=n.branch)
            n.stage += 1

        if n.stage >= p.n_stages:
            # Reasoning complete on whichever line the chain is on. The forced
            # answer fires at the same depth on every branch, so the verifier
            # sees identical progress for a right and a wrong chain.
            n.tokens -= p.answer_cost
            n.answered = True
            n.answer = self._answer_of(n)
        return n

    def _answer_of(self, s: ChainState) -> str:
        if s.branch == 0 and s.stage >= self.p.n_stages:
            return self.p.answer
        if s.branch == 0:
            return f"early@{s.stage}"
        return f"wrong@{s.branch}"

    def observe(self, s: ChainState) -> np.ndarray:
        """Stand-in for the sentence embedding of the chain.

        The prior art used all-MiniLM L2 distance between chains-of-thought.
        Here the identity of a line of reasoning is its branch id, so the
        embedding is a fixed pseudo-random vector per branch plus the two
        scalars a real embedding would also expose.
        """
        return np.concatenate(
            [np.array([float(s.stage), float(s.tokens)]), self._emb(s.branch)]
        )

    def _emb(self, branch: int) -> np.ndarray:
        v = self._emb_cache.get(branch)
        if v is None:
            r = np.random.default_rng(_h("emb", branch) % (2**32))
            v = r.normal(size=3)
            self._emb_cache[branch] = v
        return v

    def reward(self, s: ChainState) -> float:
        """Process verifier. Ground-truth free by construction.

        It scores what a real step-level verifier can check: how many sub-goals
        look resolved, whether the chain contradicted itself, whether it reached
        a final answer at all. It cannot check whether the *right* operation was
        chosen, which is exactly why a valid-but-wrong branch is attractive.
        """
        if self.judge != "process":
            raise ValueError(f"unknown judge: {self.judge}")
        progress = min(s.stage, self.p.n_stages) / float(self.p.n_stages)
        r = 3.0 * progress
        r -= 2.0 * float(s.flagged)
        if s.answered:
            # A finished chain is worth more than an open one, but only in
            # proportion to the sub-goals it actually resolved: an answer with
            # no derivation behind it scores 0, which is what stops the swarm
            # from collapsing onto "answer immediately".
            r += 1.0 * progress
        r -= 1.5 * s.contradictions
        if not self.viable(s):
            r -= 5.0
        return float(r)

    def sample_action(self, s: ChainState, rng: np.random.Generator) -> int:
        return int(rng.choice(len(ACTIONS), p=PROPOSAL))

    def viable(self, s: ChainState) -> bool:
        if s.answered:
            return True                   # a complete chain is not a dead one
        if s.tokens <= 0.0:
            return False                  # ran out of context
        if s.contradictions >= self.p.max_contradictions:
            return False                  # self-contradictory chain
        return True

    def key(self, s: ChainState) -> Hashable:
        """Answer-key identity used by phi_cone to count distinct futures."""
        if s.answered:
            return ("ans", s.answer)
        return ("node", s.branch, s.stage)

    # -- harness helpers (never used by the planner) -------------------------

    def is_terminal(self, s: ChainState) -> bool:
        return s.answered or not self.viable(s)

    def death_cause(self, s: ChainState) -> Optional[str]:
        if s.answered:
            return None
        if s.tokens <= 0.0:
            return "budget"
        if s.contradictions >= self.p.max_contradictions:
            return "contradiction"
        return None

    def is_correct(self, s: ChainState) -> bool:
        """SCORER ONLY. Not reachable from reward/viable/key/sample_action."""
        return bool(s.answered and s.answer == self.p.answer)
