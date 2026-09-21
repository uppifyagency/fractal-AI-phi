# E1 — gamma sweep on TrapGrid

**Question.** Does the Phi factor trade trap deaths for goal completions at
equal simulator budget, and at what gamma?

**Pre-registered predictions.**

- P1. Trap deaths at gamma >= 0.5 are strictly fewer than at gamma = 0. Pilot
  says 0 vs 10 out of 10; confirm with n = 30 seeds and bootstrap CI95.
- P2. The trade is against fuel exhaustion and step count, not against the goal
  rate. If the goal rate also drops, the layer is a brake, not a steering wheel,
  and that is a negative result worth reporting.
- P3. There is an interior optimum in gamma. Monotone improvement up to the
  largest gamma tested would mean the reward term is being swamped, which is a
  different finding.

**Runs.**

1. gamma in {0, 0.25, 0.5, 1.0, 2.0, 4.0}, 30 seeds each, alpha = beta = 1,
   N = 32, M = 10, TrapGrid(fuel=24), max_steps = 40.
2. Equal-budget arm: re-run gamma = 0 with N and M raised until `sim_steps`
   matches the gamma = 1 arm. This is the arm that matters. A win that
   disappears at equal budget is not a win.
3. Amortisation: phi_every in {1, 2, 3, 5} at gamma = 1. Report the cost/benefit
   curve.
4. Geometry: phi_m in {2, 4, 8}, phi_h in {1, 3, 5} at gamma = 1, 15 seeds.

**Deliverables.** `scripts/run_e1.py` (JSON per run into `results/`),
`REPORT.md` with a table per sweep, bootstrap CI95 on every mean, and an
explicit verdict per prediction.

**Open question inherited from the spec.** At gamma = 1 the pilot agent hovers
one cell above the goal, because the goal corner is option-poor. Quantify it:
how often does the episode end in `timeout` adjacent to the goal? If it is
common, propose (do not implement) a gamma schedule.
