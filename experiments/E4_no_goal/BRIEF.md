# E4 — competence with no goal

**Question.** With alpha = 0 and gamma = 1, does an agent with no goal behave
competently?

This is the rung that is worth publishing. Sergio's Common Sense mode
(alpha = 0, beta = 1) already showed an Atari agent that survives without
chasing reward. The claim here is stronger and sharper: with alpha = 0 the
*only* pressure is the reachable-future count, and that alone should produce
recognisable purposive behaviour. It is the highway argument made falsifiable.

**Pre-registered predictions.**

- P1. alpha = 0, gamma = 1 survives longer than a uniform random policy on
  TrapGrid, by a wide margin. If it does not, the layer does nothing.
- P2. It survives longer than alpha = 0, gamma = 0 (canonical Common Sense,
  diversity only). This is the comparison that separates Phi from beta, and it
  is the core claim of the whole project.
- P3. It does **not** reach the goal more often than random. It has no goal.
  A goal rate above random would mean Phi is smuggling in reward information
  through the environment geometry, which must be reported, not hidden.

**Runs.** TrapGrid, 50 seeds, four arms: random policy; alpha=0 gamma=0;
alpha=0 gamma=1; alpha=1 gamma=0. Then a second environment with different
geometry, so the result is not an artefact of one map. Build it: a room with
narrowing corridors, or a resource-depletion map. Keep both properties from
`docs/SPEC.md` section 5, irreversibility and a finite budget.

**Deliverables.** `scripts/run_e4.py`, the second environment in
`src/fmcphi/envs/`, `REPORT.md` with survival curves and CI95, and an explicit
verdict on P3 in particular.
