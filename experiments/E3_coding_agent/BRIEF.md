# E3 — Phi for a coding agent

**Question.** What is Phi when the environment is a repository and the actions
are tool calls?

**Context.** The reference project's `fractal-coding-loop` plugin already maps
FMC onto coding: state is a git SHA in a worktree, actions are strategy labels,
the simulator is an LLM sub-agent, reward is tests plus lint plus a judge. Its
survival term is `R_alive`, a binary multiplicative factor. Binary and
instantaneous: it notices death, it does not anticipate it.

**What to build.** A `phi_repo(state)` with three multiplicative components,
each measurable without an LLM call:

1. Viable next actions: how many of the K strategies still apply from this SHA
   without breaking the build. Depth-1, exhaustive, the `phi_viable_actions`
   analogue.
2. Budget slack: remaining context window, tokens, wall-clock, money.
   `phi_slack` applies unchanged.
3. Irreversibility: a penalty per option-destroying action. `push --force`,
   `rm`, a sent email, a DB migration, an amend on shared history. These are
   cone contractions and they are detectable statically from the diff and the
   tool call, no model needed.

**Pre-registered predictions.**

- P1. Phi is strictly lower on a SHA immediately preceding an irreversible
  action than on a comparable SHA without one.
- P2. On a task with a trap (an attractive shortcut that destroys options), the
  gamma > 0 planner avoids it while gamma = 0 takes it. This is E1 transposed,
  and the task must be built to contain the trap or the test is null.

**Scope.** Design, implementation and unit tests against a fixture repository.
**No live LLM run** in this rung: E3 delivers the instrument, E2 has the budget.

**Deliverables.** `scripts/phi_repo.py`, a fixture repo under `fixtures/`, unit
tests, `REPORT.md` covering how the three components compose and what the
failure modes are.
