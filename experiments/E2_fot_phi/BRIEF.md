# E2 — Fractal-of-Thought with Phi

**Question.** Does Phi help an LLM that plans over its own reasoning, once the
reward stops being a proxy for length?

**Context.** The reference project's Bet 2 ran FMC over chains of thought and
got 87.5% against 83.3% for self-consistency, and called the verdict honest but
modest. Its own weak point is stated in its docstring: the reward is
`1/(1+|cot|)`, a proxy for "concise means confident". That is not a goal signal.

**What to build.**

1. A rebuilt FoT harness where R comes from a real judge (LLM-as-judge or a
   verifier on the final answer), not from length.
2. Phi for a reasoning state, as the number of distinct viable continuations:
   fan out m short continuations from a partial chain, drop the ones that
   contradict themselves or run out of context budget, take the perplexity of
   their distinct answer-keys. `phi_slack` maps to remaining context tokens.
3. A mock-LLM backend so the whole harness runs deterministically and free in
   CI. The real-model path must be a flag, not the default.

**Pre-registered predictions.**

- P1. On problems needing multiple steps, FoT+Phi beats FoT at equal token
  budget. Equal token budget, not equal N and M.
- P2. The gain concentrates on problems where the baseline commits early to a
  wrong branch. Report the split.
- P3. On single-step problems Phi is flat. It should be: no irreversibility,
  no cone narrowing (`docs/SPEC.md` section 5).

**Budget.** $25 cap, pre-registered. Do not run the paid arm without asking.

**Deliverables.** `scripts/fot_phi.py`, mock backend, unit tests, `REPORT.md`
with the harness validated end to end on the mock and the paid arm specified
down to the exact command.
