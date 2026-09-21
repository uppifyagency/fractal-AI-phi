# E5 — ablations

**Question.** Is the E1 and E4 effect caused by Phi as defined, or by anything
that perturbs the virtual reward?

Without this rung the whole ladder is unfalsifiable. A regulariser that happened
to add noise in the right place would produce the same headline numbers.

**Arms**, all on TrapGrid, 30 seeds, at the best gamma from E1:

1. **Shuffled Phi.** Compute the Phi vector, then permute it across walkers.
   Same marginal distribution, no state correspondence. Prediction: the effect
   disappears. This is the decisive arm.
2. **Constant Phi.** Replace it with its swarm mean. Isolates the variance
   contribution from the state-dependence.
3. **No survival weighting.** `weight_by_survival=False`. Predicted to be worse
   than weighted but better than gamma = 0, since the trap still returns Phi = 0
   exactly.
4. **Cheap proxy.** `phi_viable_actions` instead of `phi_cone`. Costs K steps
   rather than m*h. If it matches, the expensive estimator is not needed and
   that changes the cost story for E2 and E3.
5. **Random noise.** Replace Phi with uniform noise on the same support. The
   null.

**Pre-registered decision rule.** The layer is supported only if arm 1 and arm 5
both lose to the real Phi with non-overlapping CI95. If shuffled Phi matches
real Phi, report the negative result plainly and stop the ladder.

**Deliverables.** `scripts/run_e5.py`, `REPORT.md` with one table, CI95 on
every arm, and a one-line verdict at the top.
