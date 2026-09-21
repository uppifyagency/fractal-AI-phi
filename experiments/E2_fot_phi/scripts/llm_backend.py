"""E2 paid arm: the same FoT+Phi harness over a real model's own continuations.

NOT RUN. This module has never made a network call. It is the specification of
the paid arm, written so a human can run it, plus a `--dry-run-paid` mode that
renders every prompt and prices the run without contacting anything. The dry-run
path *is* exercised (see `test_e2_llm_dryrun.py`); the network path is not, and
REPORT.md says so.

Why the shape is what it is
---------------------------
`phi_cone` costs m*h simulator steps per walker per Phi tick. On a real model a
"simulator step" is a generation, so the naive port is unaffordable: the mock's
matched budget is 448 steps per decision, which at ~2c a call is ~$9 per
decision. Two things make it affordable:

1. **Node caching.** The environment is the model's own branching structure, and
   that structure is a property of the node, not of the walker standing on it.
   One call per *distinct* node yields its three ranked continuations, the
   verifier's verdict on the path so far, and the model's current best-guess
   answer. Every walker that visits that node reuses them. The bill is then
   bounded by the number of distinct nodes reachable within `--max-depth`, not
   by N*M*m*h.
2. **A depth cap.** Nodes reachable within depth d is at most (5^(d+1)-1)/4.

Cost is therefore predictable before the run, and `--dry-run-paid` prints it.

Mapping onto the Phi layer
--------------------------
    walker state    a path through the model's own continuation tree
    R               the verifier's verdict on the chain so far (a real judge,
                    not `1/(1+|cot|)`)
    viable(s)       the chain has not been judged self-contradictory and has
                    context budget left
    key(s)          the model's current best-guess answer at that node, so
                    `phi_cone` counts *distinct answer-keys still reachable*,
                    which is the definition the brief asks for
    phi_slack       remaining context tokens over the budget
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Hashable, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# Anthropic first-party API pricing, USD per million tokens, as of the model
# table bundled with the claude-api skill (cached 2026-06-24). Used only by the
# estimator; nothing here bills anything.
PRICING: Dict[str, Tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
DEFAULT_MODEL = "claude-opus-5"

N_CAND = 3
CAND0, CAND1, CAND2, BACKTRACK, ANSWER = 0, 1, 2, 3, 4
ACTIONS: Tuple[int, ...] = (CAND0, CAND1, CAND2, BACKTRACK, ANSWER)

SYSTEM = (
    "You are one node in a search over chains of reasoning. You never give a "
    "final answer unless asked for one. You always reply with a single JSON "
    "object and nothing else."
)

NODE_PROMPT = """\
Problem:
{question}

Reasoning so far ({n_steps} step(s)):
{chain}

Reply with a single JSON object with exactly these keys:
  "continuations": a list of exactly {k} strings. Each is ONE next reasoning
      step that could follow, ordered from the one you consider most likely to
      the least. They must be genuinely different lines of reasoning, not
      paraphrases of each other.
  "verdict": "ok" if every step above is arithmetically and logically sound
      given the problem, "slip" if some step contains a checkable error,
      "contradiction" if the steps contradict each other.
  "progress": an integer 0-100, how much of the problem the steps above have
      actually resolved.
  "best_guess": your current best final answer given only the steps above,
      as a bare number or short string.

JSON only."""


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def cost(self, model: str) -> float:
        pin, pout = PRICING.get(model, PRICING[DEFAULT_MODEL])
        return (self.input_tokens * pin + self.output_tokens * pout) / 1e6


@dataclass
class NodeInfo:
    continuations: List[str]
    verdict: str
    progress: float
    best_guess: str


class AnthropicBackend:
    """One cached call per distinct reasoning node.

    UNVALIDATED against the live API. Run `--dry-run-paid` first, then a single
    problem, and read the per-call cost before letting it loose on a set.
    """

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 1024,
                 dry_run: bool = True):
        self.model = model
        self.max_tokens = max_tokens
        self.dry_run = dry_run
        self.usage = Usage()
        self._cache: Dict[Tuple[str, Tuple[int, ...]], NodeInfo] = {}
        self._client = None
        self.rendered_prompts: List[str] = []

    def _client_or_die(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:                       # pragma: no cover
                raise RuntimeError(
                    "the paid arm needs the anthropic SDK: "
                    "`uv add anthropic`, or run with "
                    "`uv run --with anthropic python ...`") from e
            self._client = anthropic.Anthropic()
        return self._client

    def node(self, question: str, chain: List[str],
             path: Tuple[int, ...]) -> NodeInfo:
        ck = (question, path)
        if ck in self._cache:
            return self._cache[ck]
        prompt = NODE_PROMPT.format(
            question=question, n_steps=len(chain), k=N_CAND,
            chain="\n".join(f"{i + 1}. {s}" for i, s in enumerate(chain))
            or "(nothing yet)")
        if self.dry_run:
            self.rendered_prompts.append(prompt)
            # Deterministic placeholder so the harness can be walked end to end
            # without a network call.
            info = NodeInfo(
                continuations=[f"<dry-run step {len(chain) + 1}.{i}>"
                               for i in range(N_CAND)],
                verdict="ok", progress=min(100.0, 25.0 * len(chain)),
                best_guess=f"<dry-run guess d{len(chain)}>")
            self.usage.calls += 1
            self.usage.input_tokens += _rough_tokens(SYSTEM + prompt)
            self.usage.output_tokens += 260
            self._cache[ck] = info
            return info

        client = self._client_or_die()
        resp = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        self.usage.calls += 1
        self.usage.input_tokens += resp.usage.input_tokens
        self.usage.output_tokens += resp.usage.output_tokens
        info = _parse_node(text)
        self._cache[ck] = info
        return info


def _rough_tokens(s: str) -> int:
    """Only for the estimator. Use `client.messages.count_tokens` for exactness."""
    return max(1, len(s) // 4)


def _parse_node(text: str) -> NodeInfo:
    """Defensive JSON extraction. The model is asked for JSON only; assume it
    sometimes wraps it in a fence anyway."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return NodeInfo([""] * N_CAND, "contradiction", 0.0, "")
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return NodeInfo([""] * N_CAND, "contradiction", 0.0, "")
    cands = [str(c) for c in (d.get("continuations") or [])][:N_CAND]
    while len(cands) < N_CAND:
        cands.append(cands[-1] if cands else "")
    return NodeInfo(
        continuations=cands,
        verdict=str(d.get("verdict", "ok")),
        progress=float(d.get("progress", 0) or 0),
        best_guess=str(d.get("best_guess", "")),
    )


# ---------------------------------------------------------------------------
# Environment over real chains of thought
# ---------------------------------------------------------------------------

@dataclass
class LLMChainState:
    path: Tuple[int, ...] = ()
    chain: List[str] = field(default_factory=list)
    tokens: float = 0.0            # remaining context budget
    flagged: bool = False
    contradicted: bool = False
    answered: bool = False
    answer: Optional[str] = None


class LLMReasoningEnv:
    """Same fmcphi Environment protocol as the mock, backed by a real model."""

    def __init__(self, question: str, backend: AnthropicBackend,
                 tokens0: float = 8.0, max_depth: int = 3,
                 step_cost: float = 1.0, backtrack_cost: float = 2.0):
        self.question = question
        self.backend = backend
        self.tokens0 = tokens0
        self.max_depth = max_depth
        self.step_cost = step_cost
        self.backtrack_cost = backtrack_cost

    def reset(self) -> LLMChainState:
        return LLMChainState(tokens=self.tokens0)

    def actions(self):
        return ACTIONS

    def clone_state(self, s: LLMChainState) -> LLMChainState:
        return LLMChainState(s.path, list(s.chain), s.tokens, s.flagged,
                             s.contradicted, s.answered, s.answer)

    def _info(self, s: LLMChainState) -> NodeInfo:
        return self.backend.node(self.question, s.chain, s.path)

    def step(self, s: LLMChainState, a: int) -> LLMChainState:
        n = self.clone_state(s)
        if n.answered or not self.viable(n):
            return n
        if a == ANSWER or len(n.path) >= self.max_depth:
            n.tokens -= self.step_cost
            n.answered = True
            n.answer = self._info(n).best_guess
            return n
        if a == BACKTRACK:
            n.tokens -= self.backtrack_cost
            if n.path:
                n.path = n.path[:-1]
                n.chain = n.chain[:-1]
                n.flagged = False
            else:
                n.contradicted = True
            return n
        info = self._info(n)
        n.tokens -= self.step_cost
        n.chain = n.chain + [info.continuations[a]]
        n.path = n.path + (a,)
        after = self._info(n)
        n.flagged = after.verdict == "slip"
        n.contradicted = after.verdict == "contradiction"
        return n

    def observe(self, s: LLMChainState) -> np.ndarray:
        """Stand-in for an embedding of the chain, kept cheap and offline.

        A real E2 paid run should swap this for a sentence-embedding of
        `" ".join(s.chain)` as in the prior art. That is an extra dependency,
        not an extra API cost, and it is the one thing in this file that would
        change the numbers rather than the bill.
        """
        h = abs(hash(s.path)) % (2 ** 32)
        r = np.random.default_rng(h)
        return np.concatenate([np.array([float(len(s.path)), s.tokens]),
                               r.normal(size=3)])

    def reward(self, s: LLMChainState) -> float:
        info = self._info(s)
        progress = info.progress / 100.0
        r = 3.0 * progress
        r -= 2.0 * float(s.flagged)
        if s.answered:
            r += 1.0 * progress
        if not self.viable(s):
            r -= 5.0
        return float(r)

    def sample_action(self, s: LLMChainState, rng: np.random.Generator) -> int:
        from mock_llm import PROPOSAL
        return int(rng.choice(len(ACTIONS), p=PROPOSAL))

    def viable(self, s: LLMChainState) -> bool:
        if s.answered:
            return True
        return (s.tokens > 0.0) and not s.contradicted

    def key(self, s: LLMChainState) -> Hashable:
        if s.answered:
            return ("ans", s.answer)
        return ("guess", self._info(s).best_guess)

    def is_terminal(self, s: LLMChainState) -> bool:
        return s.answered or not self.viable(s)

    def death_cause(self, s: LLMChainState) -> Optional[str]:
        if s.answered:
            return None
        if s.contradicted:
            return "contradiction"
        if s.tokens <= 0.0:
            return "budget"
        return None


# ---------------------------------------------------------------------------
# Cost estimate and the gate
# ---------------------------------------------------------------------------

def estimate(model: str, n_problems: int, n_seeds: int, n_arms: int,
             max_depth: int, in_tok: int = 700, out_tok: int = 260) -> Dict:
    """Upper bound on the bill: one call per distinct node within `max_depth`.

    Nodes reachable with 3 branching actions and depth d is (3^(d+1)-1)/2.
    Caching is per (question, path), so seeds and arms share the tree: the
    node count does not multiply by them. This is an upper bound because the
    search rarely reaches every node.
    """
    nodes = (3 ** (max_depth + 1) - 1) // 2
    calls = nodes * n_problems
    pin, pout = PRICING.get(model, PRICING[DEFAULT_MODEL])
    cost = calls * (in_tok * pin + out_tok * pout) / 1e6
    return {
        "model": model, "max_depth": max_depth,
        "nodes_per_problem_upper_bound": nodes,
        "problems": n_problems, "seeds": n_seeds, "arms": n_arms,
        "api_calls_upper_bound": calls,
        "assumed_input_tokens_per_call": in_tok,
        "assumed_output_tokens_per_call": out_tok,
        "usd_upper_bound": round(cost, 2),
        "note": "seeds and arms share the per-problem node cache, so they do "
                "not multiply the bill; only distinct (problem, path) pairs do",
    }


def run_paid(args) -> int:
    """Entry point reached from `fot_phi.py --backend anthropic`.

    Refuses to spend anything unless the human passes `--confirm-paid` *and*
    a credential is resolvable. `--dry-run-paid` always wins over both.
    """
    model = getattr(args, "model", DEFAULT_MODEL)
    est = estimate(model, args.problems, args.seeds, 2,
                   getattr(args, "max_depth", 3))
    print(json.dumps(est, indent=2))

    if getattr(args, "dry_run_paid", False):
        backend = AnthropicBackend(model=model, dry_run=True)
        env = LLMReasoningEnv("DRY RUN: a three-step word problem.", backend,
                              max_depth=getattr(args, "max_depth", 3))
        s = env.reset()
        rng = np.random.default_rng(0)
        for _ in range(4):
            s = env.step(s, env.sample_action(s, rng))
            if env.is_terminal(s):
                break
        print("\n--- first rendered prompt "
              "(this is exactly what would be sent) ---")
        print(backend.rendered_prompts[0] if backend.rendered_prompts
              else "(none)")
        print(f"\ndry run walked {len(s.path)} step(s), "
              f"{backend.usage.calls} node call(s) simulated, "
              f"terminal={env.is_terminal(s)}")
        print("NO NETWORK CALL WAS MADE.")
        return 0

    if not getattr(args, "confirm_paid", False):
        print("\nREFUSING. The paid arm needs --confirm-paid. Run "
              "--dry-run-paid first and read the estimate above.",
              file=sys.stderr)
        return 2
    if not (os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print("\nREFUSING. No ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN in the "
              "environment. If you authenticated with `ant auth login`, the "
              "SDK will pick the profile up, but this gate wants the choice "
              "to be explicit: export the key for this run.", file=sys.stderr)
        return 2
    if est["usd_upper_bound"] > 25.0:
        print(f"\nREFUSING. Upper bound ${est['usd_upper_bound']} exceeds the "
              f"$25 cap pre-registered in BRIEF.md. Lower --problems or "
              f"--max-depth, or raise the cap in the brief first.",
              file=sys.stderr)
        return 2

    print("\nThe paid arm has never been executed. It is specified, not "
          "validated. Read llm_backend.py before continuing, then remove this "
          "line.", file=sys.stderr)
    return 3
