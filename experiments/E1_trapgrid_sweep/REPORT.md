# E1 - gamma sweep on TrapGrid - REPORT

Revision 2, after adversarial review. The headline of revision 1 was wrong and
is retracted; see **Headline** and **Review response** below.

Environment `TrapGrid(fuel=24)`, `max_steps=40`, `alpha=beta=1`, `N=32`, `M=10`,
`phi_m=4`, `phi_h=3`, `phi_every=1`, `budget_attr=None` (planner defaults), seeds
`0..n-1`. Every mean carries a percentile bootstrap CI95 (10 000 resamples);
proportions also carry an exact Clopper-Pearson CI95 where the bootstrap
degenerates. Arms share seeds, so deltas are computed paired (common random
numbers) and the binary contrasts also carry an exact McNemar p.
Total compute: **1.2 min** for the four original sweeps, **~3 min** including
every addendum and the revision-2 arms. No seed count was cut.

Reproduce:

```
uv run python experiments/E1_trapgrid_sweep/scripts/run_e1.py --all
uv run python experiments/E1_trapgrid_sweep/scripts/analyze_e1.py
uv run python experiments/E1_trapgrid_sweep/scripts/phi_map_and_hover.py
uv run python experiments/E1_trapgrid_sweep/scripts/extra_contrasts.py
uv run python experiments/E1_trapgrid_sweep/scripts/decomposition.py    # revision 2
uv run python experiments/E1_trapgrid_sweep/scripts/review_checks.py    # revision 2
```

---

## Headline

The **Phi layer as shipped** (the exponent `relativize(Phi)^gamma` *and* the hard
mask `kill_dead`, which `planner.plan` switches on together whenever
`gamma != 0 and phi_every > 0`) turns 30/30 trap deaths into 16/30 goals and
1/30 trap deaths on TrapGrid, and the effect survives equal simulator budget
against a gamma = 0 control. That much replicates.

Three attributions that revision 1 made, and that the data does not support:

1. **It is not the causal-cone entropy term alone.** Factorially, at the same
   4160 sim/decision and the same 30 seeds: mask only = 19 goal / 11 trap,
   exponent only = 9 goal / 21 trap, both = 16 goal / 1 trap, neither = 0 goal /
   30 trap. The **mask carries most of the goal-rate gain** (+0.633
   [+0.467, +0.800] against gamma = 0, versus +0.300 [+0.133, +0.467] for the
   exponent) and the **exponent is what drives trap rate down** (mask only minus
   both = +0.333 [+0.167, +0.500] trap, McNemar p = 0.002), paid for in fuel
   deaths (+0.433 [+0.267, +0.600]).
2. **A free viability mask reproduces the win at 1/13 of the cost.** A gamma = 0
   arm that merely zeroes VR for walkers standing in a trap, costing **zero**
   extra simulator steps, scores 23 goal / 6 trap / 1 fuel at 320 sim/decision.
   Against the full Phi arm at 4160 sim/decision, its goal rate is +0.233
   [-0.033, +0.500] (p = 0.14) and its trap rate +0.167 [+0.000, +0.333]
   (p = 0.125): **indistinguishable on both axes at 13x less compute.**
3. **At equal budget, the free mask strictly dominates the Phi layer.** The same
   viability mask with N = 416 (4160 sim/decision, matched) scores **30/30 goal,
   0 trap, 0 fuel**, beating the Phi arm by +0.467 [+0.300, +0.633] on goal rate
   (p = 0.0001) and tying it on trap rate (-0.033 [-0.100, +0.000]).

So the honest abstract-level claim for E1 is: *on TrapGrid the Phi layer beats
canonical FMC decisively, but almost all of that gap is explained by the layer
telling FMC that traps are absorbing, which canonical FMC never consults. Once
the gamma = 0 baseline is given that same one-bit fact for free, it matches the
Phi layer at 1/13 the cost and beats it at equal cost.* E1 does not show that
causal-cone freedom is worth its price on this environment. The exponent does
have a distinct, measurable effect (it is the only thing that pushes trap deaths
to 1/30 and it buys that with fuel deaths), but that effect is dominated by the
cheap control on the metric the BRIEF pre-registered.

Scope of the negative result: TrapGrid exposes death as a predicate on the
*current* state, so a viability mask is available for free. In an environment
where the absorbing set is only detectable a few steps ahead, no such mask
exists and the cone rollouts would be the only way to see it. E1 does not test
that case, and this paragraph is scope, not a rescue: the pre-registered E1
claim as written stands falsified in its causal attribution.

---

## Verdicts

**P1 - SUPPORTED, with the attribution corrected.** Trap deaths at gamma >= 0.5
are strictly fewer than at gamma = 0, with every paired CI95 excluding zero and
every McNemar p < 0.001. gamma = 0: 30/30 trap deaths (Clopper-Pearson
[0.884, 1.000], n = 30). gamma = 0.5: 5/30, delta -0.833 [-0.967, -0.700].
gamma = 1: 1/30, delta -0.967 [-1.000, -0.900]. gamma = 2: 5/30, delta -0.833
[-0.933, -0.700]. gamma = 4: 2/30, delta -0.933 [-1.000, -0.833]. gamma = 0.25,
below the pre-registered threshold, also qualifies (7/30, delta -0.767
[-0.900, -0.600]). What the sweep varies is `gamma`, which switches the mask and
the exponent on together; sweep 1b separates them and finds the **exponent** is
the part that carries trap rate the last stretch to 1/30.

**P2 - UNTESTABLE against the pre-registered baseline (floor effect), and
FALSIFIED against the best gamma.** The pre-registered comparison was "the goal
rate does not also drop". The gamma = 0 goal rate is 0/30, Clopper-Pearson
[0.000, 0.116]: a rate cannot drop below the floor, so the prediction has no
content there and the correct label is *untestable*, not *supported*. Inside the
gamma > 0 range it is testable and it fails: goal rate peaks at 20/30
[0.472, 0.827] for gamma = 0.25-0.5, falls to 3/30 [0.021, 0.265] at gamma = 2
and 1/30 [0.001, 0.172] at gamma = 4, paired gamma = 2 minus gamma = 0.5 being
-0.567 [-0.767, -0.333]. **Above gamma ~ 1 the layer is a brake, exactly as P2
warned, and this is the negative result of the gamma sweep.**

**P3 - SUPPORTED on the goal-rate criterion only.** Goal rate peaks at
gamma = 0.25-0.5 (20/30; the two are tied, paired delta +0.000 [-0.267, +0.233])
and falls significantly by gamma = 2 (-0.567 [-0.767, -0.333] vs gamma = 0.5),
so monotone improvement to the largest gamma tested is ruled out and the optimum
is interior. Revision 1 also claimed a *strict interior minimum in trap rate at
gamma = 1*; that is **withdrawn**. It rests on 1 versus 5 episodes out of 30
(gamma = 2 minus gamma = 1, paired +0.133 [+0.033, +0.267], but McNemar exact
p = 0.125, 4 discordant pairs all in one direction), and gamma = 4 does not
continue the rise (+0.033 [+0.000, +0.100], p = 1.0). The goal-rate criterion
carries P3 on its own.

**Inherited open question - ANSWERED, the pre-registered event never happens.**
Timeouts are 0 in **all 23 result files and all 555 episodes** (21 distinct arms
and 495 episodes after removing byte-identical reruns; see the denominator note
below), so "ends in `timeout` adjacent to the goal" has observed frequency
0/555. This is a joint parameter-and-behaviour fact, not a construction: a
no-op costs 0.5 fuel and a move costs 1.0, so 40 stay-steps cost only 20 of the
24 fuel and a fully stalling agent *would* time out with 4.0 fuel left
(verified). No agent stalled that completely. The hover the spec describes is
real but is recorded as a **fuel** death: at gamma = 1, 6 of 14 non-goal
episodes end in the goal column x = 11 with y > 0, and at gamma = 2, 15 of 27.

---

## Sweep 1 - gamma

30 seeds per arm. gamma = 0 costs 320 simulator steps per decision; every
gamma > 0 arm is charged 4160, i.e. 13x. Sweep 2 removes that confound.
Degenerate rates carry the exact Clopper-Pearson interval, because the
percentile bootstrap returns a zero-width interval when every value is equal
(see Caveat 8).

| arm | n | sim_steps/decision | goal rate [CI95] | trap rate [CI95] | fuel rate [CI95] | timeout | steps [CI95] |
|---|---|---|---|---|---|---|---|
| gamma=0.0 | 30 | 320 | 0.000 [0.000, 0.116]* | 1.000 [0.884, 1.000]* | 0.000 [0.000, 0.116]* | 0 | 1.2 [1.0, 1.4] |
| gamma=0.25 | 30 | 4160 | 0.667 [0.500, 0.833] | 0.233 [0.100, 0.400] | 0.100 [0.000, 0.233] | 0 | 17.0 [14.1, 19.7] |
| gamma=0.5 | 30 | 4160 | 0.667 [0.500, 0.833] | 0.167 [0.033, 0.300] | 0.167 [0.033, 0.300] | 0 | 18.4 [15.2, 21.5] |
| gamma=1.0 | 30 | 4160 | 0.533 [0.367, 0.700] | 0.033 [0.000, 0.100] | 0.433 [0.267, 0.600] | 0 | 24.7 [22.2, 26.8] |
| gamma=2.0 | 30 | 4160 | 0.100 [0.000, 0.200] | 0.167 [0.033, 0.300] | 0.733 [0.567, 0.867] | 0 | 25.7 [21.9, 28.9] |
| gamma=4.0 | 30 | 4160 | 0.033 [0.000, 0.100] | 0.067 [0.000, 0.167] | 0.900 [0.767, 1.000] | 0 | 28.4 [25.4, 30.7] |

`*` = exact Clopper-Pearson, not bootstrap.

Paired deltas against gamma = 0 (same 30 seeds):

| arm | d trap rate [CI95] | d goal rate [CI95] | d fuel rate [CI95] | d steps [CI95] |
|---|---|---|---|---|
| gamma=0.25 | -0.767 [-0.900, -0.600] | +0.667 [+0.500, +0.833] | +0.100 [+0.000, +0.233] | +15.8 [+13.0, +18.4] |
| gamma=0.5 | -0.833 [-0.967, -0.700] | +0.667 [+0.500, +0.833] | +0.167 [+0.033, +0.300] | +17.2 [+13.9, +20.3] |
| gamma=1.0 | -0.967 [-1.000, -0.900] | +0.533 [+0.367, +0.700] | +0.433 [+0.267, +0.600] | +23.5 [+21.0, +25.6] |
| gamma=2.0 | -0.833 [-0.933, -0.700] | +0.100 [+0.000, +0.233] | +0.733 [+0.567, +0.867] | +24.5 [+20.6, +27.7] |
| gamma=4.0 | -0.933 [-1.000, -0.833] | +0.033 [+0.000, +0.100] | +0.900 [+0.767, +1.000] | +27.2 [+24.1, +29.5] |

A delta CI bound of exactly +0.000 here is a boundary artefact of the bootstrap
on an all-one-sided difference vector, not evidence of a null. For the two
contrasts where revision 1 leaned on such a bound, the exact McNemar p is given
in the verdicts above (gamma = 2 vs gamma = 0 on goal rate: 3 discordant pairs,
p = 0.25, i.e. **not** distinguishable from zero, which is the reading revision 1
gave it, but for the right reason).

Diagnostics. `mean Phi` is the per-tick swarm mean over the episode, `b_eff` the
effective branching factor over walker labels, `ESS` the effective sample size
out of N = 32.

| arm | n | mean Phi [CI95] | b_eff [CI95] | ESS [CI95] | final dist to goal [CI95] |
|---|---|---|---|---|---|
| gamma=0.0 | 30 | n/a (gamma=0) | 1.535 [1.383, 1.688] | 17.18 [16.74, 17.65] | 10.00 [10.00, 10.00] |
| gamma=0.25 | 30 | 1.888 [1.800, 1.969] | 1.925 [1.835, 2.015] | 15.91 [15.19, 16.72] | 2.00 [0.80, 3.40] |
| gamma=0.5 | 30 | 1.906 [1.812, 2.000] | 2.095 [1.965, 2.252] | 16.06 [15.22, 16.95] | 2.10 [0.87, 3.50] |
| gamma=1.0 | 30 | 1.870 [1.783, 1.960] | 2.162 [2.042, 2.343] | 17.36 [16.36, 18.34] | 1.57 [0.83, 2.43] |
| gamma=2.0 | 30 | 1.846 [1.796, 1.904] | 2.336 [2.238, 2.456] | 16.66 [15.41, 17.73] | 3.37 [2.37, 4.47] |
| gamma=4.0 | 30 | 1.904 [1.877, 1.931] | 2.389 [2.329, 2.461] | 15.48 [14.47, 16.26] | 4.87 [3.90, 5.83] |

### b_eff, measured at matched states

The episode-averaged `b_eff` column above **cannot** be read as a gamma effect:
the gamma = 0 mean averages 1.2 decisions all taken in the start cell, the
gamma = 4 mean averages 28 decisions spread over the board, so it confounds gamma
with episode length and with the visited-state distribution. Revision 1's
sentence "b_eff rises monotonically with gamma, 1.535 -> 2.389" is **withdrawn**.

Re-measured properly: 30 `plan()` calls per cell, matched seeds, matched state.

| probe state | gamma=0 | gamma=0.25 | gamma=1 | gamma=4 |
|---|---|---|---|---|
| start (0,0) fuel 24 | 1.564 [1.389, 1.750] | 2.156 [1.899, 2.416] | 2.093 [1.805, 2.409] | 2.102 [1.809, 2.405] |
| corridor (5,2) fuel 17 | 1.740 [1.586, 1.890] | 1.841 [1.550, 2.179] | 1.757 [1.557, 1.966] | 2.190 [2.001, 2.389] |
| pre-goal (11,1) fuel 12 | 2.119 [1.839, 2.413] | 2.240 [2.011, 2.488] | 2.515 [2.294, 2.733] | 2.186 [1.955, 2.428] |

What survives: **Phi raises b_eff over gamma = 0 at every probe state**, which is
still the opposite sign to the beta effect quoted in docs/SPEC.md section 1
(beta 0 -> 5 lowered b_eff 5.45 -> 1.89), and is still evidence that Phi is not
a re-skinned distance term. What does not survive: any claim of monotonicity in
gamma. At the start cell gamma = 0.25, 1 and 4 are flat within CI; in the
corridor only gamma = 4 separates from gamma = 0; at the pre-goal cell gamma = 1
is the peak and gamma = 4 falls back.

## Sweep 1b - factorial decomposition of the layer (revision 2)

`planner.plan` sets `use_phi = gamma != 0.0 and phi_every > 0` and only then
applies both `VR *= relativize(Phi)^gamma` and `kill_dead(vr, phis)`. Sweep 1
therefore never separated the two. This sweep does, at the identical
configuration (N = 32, M = 10, phi_m = 4, phi_h = 3), the identical 30 seeds and
the identical 4160 sim/decision in all four cells, so the comparison is at equal
budget by construction. `scripts/decomposition.py --parity-only` asserts that the
instrumented planner reproduces `fmcphi.planner.plan` bit-for-bit on the two
configurations that exist in both, and the "both" cell reproduces sweep 1's
gamma = 1 arm outcome-for-outcome.

| cell | exponent | mask | n | sim/dec | goal | trap | fuel | steps [CI95] |
|---|---|---|---|---|---|---|---|---|
| neither (Phi computed, discarded) | off | off | 30 | 4160 | 0/30 [0.000, 0.116] | 30/30 [0.884, 1.000] | 0/30 | 1.5 [1.1, 2.0] |
| exponent only (gamma=1, `kill_dead` off) | on | off | 30 | 4160 | 9/30 [0.147, 0.494] | 21/30 [0.506, 0.853] | 0/30 | 7.7 [4.6, 10.9] |
| mask only (gamma exponent 0, `kill_dead` on) | off | on | 30 | 4160 | 19/30 [0.439, 0.801] | 11/30 [0.199, 0.561] | 0/30 | 15.1 [12.5, 17.6] |
| both = the shipped layer at gamma=1 | on | on | 30 | 4160 | 16/30 [0.343, 0.717] | 1/30 [0.001, 0.172] | 13/30 | 24.7 [22.2, 26.8] |

Intervals are exact Clopper-Pearson. The "mask only" cell was run twice, once
with the exponent literally set to 0 and once through the library planner at
gamma = 1e-9 as the reviewer specified; the two agree outcome-for-outcome on all
30 seeds.

Paired deltas (same seeds, bootstrap CI95 and exact McNemar):

| contrast | d goal rate [CI95] (p) | d trap rate [CI95] (p) | d fuel rate [CI95] (p) |
|---|---|---|---|
| neither minus gamma=0 | +0.000 [+0.000, +0.000] (1.0) | +0.000 [+0.000, +0.000] (1.0) | +0.000 (1.0) |
| exponent only minus gamma=0 | +0.300 [+0.133, +0.467] (0.004) | -0.300 [-0.467, -0.133] (0.004) | +0.000 (1.0) |
| mask only minus gamma=0 | +0.633 [+0.467, +0.800] (<1e-4) | -0.633 [-0.800, -0.467] (<1e-4) | +0.000 (1.0) |
| both minus gamma=0 | +0.533 [+0.367, +0.700] (<1e-4) | -0.967 [-1.000, -0.900] (<1e-4) | +0.433 (2e-4) |
| mask only minus both | +0.100 [-0.100, +0.300] (0.51) | +0.333 [+0.167, +0.500] (0.002) | -0.433 (2e-4) |
| exponent only minus both | -0.233 [-0.467, +0.000] (0.09) | +0.667 [+0.500, +0.833] (<1e-4) | -0.433 (2e-4) |

Readings, in order of how much they matter:

1. **The mask alone beats the exponent alone on goal rate** (19/30 vs 9/30) and
   beats the shipped layer's point goal rate (19/30 vs 16/30, though that
   difference is not significant, p = 0.51).
2. **The exponent is what removes the remaining trap deaths.** Going from mask
   only to both takes trap from 11/30 to 1/30 (-0.333 paired, p = 0.002), and
   pays 13/30 fuel deaths for it. Nothing else in the sweep produces a fuel
   death at all: caution is the exponent's signature.
3. **Computing Phi and then discarding it changes nothing.** The "neither" cell
   consumes the same rng draws as the Phi arms (the cone rollouts draw from the
   same generator) and still returns 30/30 trap deaths, zero paired delta on
   every outcome. So the sweep-1 effect is not rng divergence, and the 13x
   simulator spend buys nothing unless one of the two factors consumes it.
4. Mechanism note on why the two masks are not the same object: `kill_dead`
   fires when `phi_cone` returns exactly 0, which happens either because the
   walker is genuinely non-viable **or** because all `m = 4` sampled
   continuations happened to die. The second case is estimator variance, not
   information. Measured over 400 draws per cell at fuel 12, that false-kill
   rate is 0 in the two top rows, 0.5 % to 3.0 % on the row adjacent to the trap
   row, and 3.0 % on the goal cell itself
   (`results/false_kill_rate.json`).

## Sweep 2 - equal simulator budget (the arm that matters)

gamma = 1 is charged `M * N * (1 + phi_m * phi_h)` = 13x the gamma = 0 cost at
the same (N, M). The 13x is given back to gamma = 0 in three different shapes.
The last two rows are the death-aware controls added in revision 2: they zero VR
for any walker standing in a trap, which is a predicate on the state the walker
is already in and therefore costs **zero** extra simulator steps. 30 seeds each.

| arm | n | sim_steps/decision | goal rate | trap rate | fuel rate | timeout | steps [CI95] |
|---|---|---|---|---|---|---|---|
| gamma=1 reference N=32 M=10 | 30 | 4160 | 16/30 [0.343, 0.717] | 1/30 [0.001, 0.172] | 13/30 [0.255, 0.626] | 0 | 24.7 [22.2, 26.8] |
| gamma=0 wide N=416 M=10 | 30 | 4160 | 0/30 [0.000, 0.116] | 30/30 [0.884, 1.000] | 0/30 | 0 | 1.0 [1.0, 1.0] |
| gamma=0 deep N=32 M=130 | 30 | 4160 | 0/30 [0.000, 0.116] | 30/30 [0.884, 1.000] | 0/30 | 0 | 1.4 [1.1, 1.7] |
| gamma=0 both N=116 M=36 | 30 | 4176 | 0/30 [0.000, 0.116] | 30/30 [0.884, 1.000] | 0/30 | 0 | 1.1 [1.0, 1.2] |
| **gamma=0 + viability mask N=32 M=10** | 30 | **320** | **23/30 [0.577, 0.901]** | 6/30 [0.077, 0.386] | 1/30 | 0 | 16.7 [14.1, 19.2] |
| **gamma=0 + viability mask N=416 M=10** | 30 | **4160** | **30/30 [0.884, 1.000]** | **0/30 [0.000, 0.116]** | 0/30 | 0 | 13.5 [13.3, 13.8] |

| gamma=0 control | sim/dec | d trap vs gamma=1 [CI95] (p) | d goal vs gamma=1 [CI95] (p) |
|---|---|---|---|
| wide N=416 M=10 | 4160 | +0.967 [+0.900, +1.000] (<1e-4) | -0.533 [-0.700, -0.367] (<1e-4) |
| deep N=32 M=130 | 4160 | +0.967 [+0.900, +1.000] (<1e-4) | -0.533 [-0.700, -0.367] (<1e-4) |
| both N=116 M=36 | 4176 | +0.967 [+0.900, +1.000] (<1e-4) | -0.533 [-0.700, -0.367] (<1e-4) |
| **+ viability mask N=32 M=10** | **320** | +0.167 [+0.000, +0.333] (0.125) | +0.233 [-0.033, +0.500] (0.14) |
| **+ viability mask N=416 M=10** | 4160 | -0.033 [-0.100, +0.000] (1.0) | **+0.467 [+0.300, +0.633] (1e-4)** |

**The win over *blind* FMC survives equal budget completely, and evaporates
against *death-aware* FMC.** Spending the same 4160 simulator steps on more
walkers, more inner iterations, or both leaves canonical FMC at 30/30 trap
deaths, because in the vendored core loop and in `plan()`'s gamma = 0 path
viability is never consulted during rollouts: a walker that enters a trap keeps
stepping, can leave the trap row, and its -Manhattan reward keeps improving, so
extra budget only resolves that same wrong objective more sharply. But the
obvious null hypothesis, *"you only needed to tell FMC that traps are
absorbing"*, costs ~0 extra simulator steps and was never run in revision 1. Run
now, it matches the Phi layer at 320 sim/decision and beats it at 4160.

Revision 1's sentence *"extra budget without the Phi factor sharpens the wrong
objective; it does not buy caution"* is retained only for the three blind
controls, where it is exactly what the data shows. Any implication that no cheap
gamma = 0 intervention could close the gap is **retracted**: one does, and it is
free.

Revision 1 also wrote that extra budget makes gamma = 0 *"die faster: mean steps
1.2 at N = 32 against 1.0 at N = 416"*. That contrast was unpaired and the two
bootstrap CIs overlapped. Paired, it is -0.20 steps [-0.40, -0.03], Wilcoxon
p = 0.063, and the other two controls go the other way or nowhere (deep +0.17
[-0.07, +0.47], p = 0.24; both -0.13 [-0.37, +0.07], p = 0.23). The claim is
**withdrawn**: at this n the three controls all die on move 1 or 2 and the
differences between them are noise.

## Sweep 3 - phi_every amortisation

gamma = 1, 30 seeds. Phi is recomputed on ticks `t` with `t % k == 0`, so the
per-decision charge is `M*N + ceil(M/k)*N*phi_m*phi_h`: 4160, 2240, 1856, 1088
for k = 1, 2, 3, 5.

| arm | n | sim_steps/decision | cost vs k=1 | goal rate [CI95] | trap rate [CI95] | fuel rate [CI95] | steps [CI95] |
|---|---|---|---|---|---|---|---|
| phi_every=1 | 30 | 4160 | 1.00x | 0.533 [0.367, 0.700] | 0.033 [0.000, 0.100] | 0.433 [0.267, 0.600] | 24.7 [22.2, 26.8] |
| phi_every=2 | 30 | 2240 | 0.54x | 0.667 [0.500, 0.833] | 0.100 [0.000, 0.200] | 0.233 [0.100, 0.400] | 22.8 [20.1, 25.2] |
| phi_every=3 | 30 | 1856 | 0.45x | 0.633 [0.467, 0.800] | 0.033 [0.000, 0.100] | 0.333 [0.167, 0.500] | 25.4 [23.8, 26.8] |
| phi_every=5 | 30 | 1088 | 0.26x | 0.667 [0.500, 0.833] | 0.000 [0.000, 0.116]* | 0.333 [0.167, 0.500] | 24.2 [22.6, 25.9] |

Paired against phi_every = 1:

| contrast | cost ratio | d goal rate [CI95] | d trap rate [CI95] |
|---|---|---|---|
| k=2 minus k=1 | 0.54x | +0.133 [-0.133, +0.400] | +0.067 [+0.000, +0.167] |
| k=3 minus k=1 | 0.45x | +0.100 [-0.167, +0.367] | +0.000 [-0.100, +0.100] |
| k=5 minus k=1 | 0.26x | +0.133 [-0.133, +0.400] | -0.033 [-0.100, +0.000] |

**Amortisation is free on this task.** At phi_every = 5 the layer costs 26 % of
the every-tick price and no outcome degrades: goal rate +0.133 [-0.133, +0.400]
and trap rate -0.033 [-0.100, +0.000], both consistent with zero, both
point-estimated in the favourable direction. A stale Phi acts like a slightly
weaker gamma, which is the same axis as sweep 1's interior optimum, so cost and
quality are not in tension here. Reported plainly: the CIs are wide at n = 30 and
no arm is *significantly* better; the claim is "no measurable loss at 3.8x
cheaper". Note that sweep 1b now supplies a cheaper explanation for why staleness
is harmless: most of the benefit at gamma = 1 is the mask, and a walker that is
in a trap on tick t was in a trap on tick t-1 too.

## Sweep 4 - cone geometry

gamma = 1, phi_every = 1, 15 seeds per cell (as pre-registered). CIs are
correspondingly wide; the pooled contrasts at the bottom carry n = 45.

| phi_m | phi_h | n | sim_steps/decision | goal rate [CI95] | trap rate [CI95] | fuel rate [CI95] | steps [CI95] |
|---|---|---|---|---|---|---|---|
| 2 | 1 | 15 | 960 | 0.800 [0.600, 1.000] | 0.200 [0.000, 0.400] | 0.000 [0.000, 0.000] | 18.0 [13.4, 22.1] |
| 2 | 3 | 15 | 2240 | 0.467 [0.200, 0.733] | 0.133 [0.000, 0.333] | 0.400 [0.133, 0.667] | 24.7 [20.7, 27.9] |
| 2 | 5 | 15 | 3520 | 0.067 [0.000, 0.200] | 0.200 [0.000, 0.400] | 0.733 [0.467, 0.933] | 27.1 [22.7, 29.9] |
| 4 | 1 | 15 | 1600 | 0.333 [0.133, 0.600] | 0.200 [0.000, 0.400] | 0.467 [0.200, 0.733] | 20.5 [14.8, 25.5] |
| 4 | 3 | 15 | 4160 | 0.600 [0.333, 0.800] | 0.000 [0.000, 0.000] | 0.400 [0.200, 0.667] | 24.1 [21.5, 26.8] |
| 4 | 5 | 15 | 6720 | 0.200 [0.000, 0.400] | 0.133 [0.000, 0.333] | 0.667 [0.400, 0.867] | 24.8 [19.7, 28.9] |
| 8 | 1 | 15 | 2880 | 0.400 [0.133, 0.667] | 0.333 [0.133, 0.600] | 0.267 [0.067, 0.533] | 18.9 [13.1, 24.3] |
| 8 | 3 | 15 | 8000 | 0.467 [0.200, 0.733] | 0.000 [0.000, 0.000] | 0.533 [0.267, 0.800] | 26.0 [23.8, 28.1] |
| 8 | 5 | 15 | 13120 | 0.333 [0.133, 0.600] | 0.133 [0.000, 0.333] | 0.533 [0.267, 0.800] | 24.8 [20.2, 28.7] |

The `phi_m=4 phi_h=3` row is the seed-0..14 prefix of sweep 1's gamma = 1 arm,
not an independent run.

Pooled main effects (45 episodes per level, paired by (m, seed) or (h, seed)):

| contrast | n per arm | d goal rate [CI95] |
|---|---|---|
| h=1 minus h=3 | 45 | +0.000 [-0.222, +0.222] |
| h=5 minus h=3 | 45 | **-0.311 [-0.489, -0.133]** |
| m=2 minus m=4 | 45 | +0.067 [-0.133, +0.267] |
| m=8 minus m=4 | 45 | +0.022 [-0.156, +0.178] |

**Depth hurts, fanout does not help.** Going from h = 3 to h = 5 costs goal rate
-0.311 [-0.489, -0.133] and roughly doubles sim cost. Fanout m has no detectable
effect over 2 -> 8 while cost scales linearly with it. Mechanism: with h = 5, a
random 5-step walk from almost anywhere in the safe corridor has a substantial
chance of falling into the trap row, so the survival factor drags Phi down almost
uniformly, the term saturates towards "everywhere is dangerous", and the agent
stalls until fuel runs out (fuel rate 0.667-0.733 at h = 5). Only trap deaths
stay near zero: at h = 5 the agent is not reckless, it is paralysed. The cheapest
cell tested, m = 2 h = 1 at 960 sim_steps/decision, has the highest point goal
rate of the whole Phi sweep, 0.800 [0.600, 1.000], with the caveat that n = 15
and its CI overlaps m = 4 h = 3. Read against sweep 1b, the m = 2 h = 1 cell is
also the cell closest to a pure viability mask: at h = 1 the cone barely looks
ahead at all, and what survives of Phi is mostly "am I in a trap right now".

## Phi map (null-test check and the goal-corner standoff)

`phi_cone(m=4, h=3)`, fuel = 12, 400 rng draws per cell. Range of the estimator
is [0, 4]. 'T' = trap, Phi = 0 by construction.

| y | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 2.82 | 3.12 | 3.26 | 3.29 | 3.25 | 3.30 | 3.22 | 3.18 | 3.19 | 3.25 | 3.13 | 2.80 |
| 2 | 3.09 | 3.02 | 2.99 | 3.05 | 2.91 | 2.97 | 2.85 | 2.89 | 2.97 | 2.93 | 2.97 | 3.03 |
| 1 | 2.39 | 1.74 | 1.59 | 1.56 | 1.45 | 1.49 | 1.47 | 1.53 | 1.50 | 1.51 | 1.73 | 2.45 |
| 0 | 1.13 | T | T | T | T | T | T | T | T | T | T | 1.16 |

Phi spans 1.13 to 3.30 across the free cells and is exactly 0 on the trap row, so
docs/SPEC.md section 5 is satisfied: the cone really narrows here and the term
does not vanish after relativize.

The goal cell (11, 0) scores 1.16 and is the second least free cell on the board
after the start corner, so the alpha/gamma standoff is a direct consequence of
the geometry. Revision 1 quoted "1.9x below open ground" against a mean of 2.23
that pooled rows y = 1 (1.514 over x in 2..9) and y = 2 (2.946 over the same
range), two very different regimes. Stated per row: the goal is **1.31x** below
the row the agent actually traverses on its way down (y = 1) and 2.54x below the
corridor row (y = 2). The sharp and relevant contrast is local and vertical:
the cell directly above the goal, (11, 1), scores **2.445 against the goal's
1.16, a factor 2.11**, and that single step down is the one the agent refuses to
take.

## The hover, quantified

Where non-goal episodes end, sweep 1:

| arm | non-goal n | ends in goal column x=11, y>0 | dist<=2 | dist 3-5 | dist>5 |
|---|---|---|---|---|---|
| gamma=0.25 | 10 | 1 | 4 | 1 | 5 |
| gamma=0.5 | 10 | 2 | 2 | 3 | 5 |
| gamma=1.0 | 14 | 6 | 4 | 9 | 1 |
| gamma=2.0 | 27 | 15 | 11 | 11 | 5 |
| gamma=4.0 | 29 | 5 | 4 | 13 | 12 |

At gamma = 1, 43 % (6/14) of the failures stall in the goal column without
descending the last cell or two; at gamma = 2 it is 56 % (15/27). At gamma = 4
the agent no longer even reaches the goal column (5/29): it is too expensive in
Phi to leave the top corridor at all, so failures move back out to distance > 5.

### Trap deaths at high gamma are not stall-induced

Revision 1 reported as a surprise that *"at very high gamma the agent stalls in
the corridor, runs low on fuel and then makes worse terminal moves, so excessive
caution reintroduces trap deaths"*. The per-episode records contradict it and it
is **withdrawn**. The 5 trap deaths at gamma = 2, as (steps, x, y, fuel), are
(28, 10, 0, 1.0), (2, 1, 0, 22.5), (1, 1, 0, 23.0), (1, 1, 0, 23.0),
(2, 1, 0, 22.5): four of five are move-1 or move-2 cliff walks at near-full
fuel, i.e. the same opening failure as gamma = 0, not a stall. Both gamma = 4
trap deaths are move-1 at fuel 23.0, and so is the single gamma = 1 trap death.
Exactly one episode in the entire sweep matches the described mechanism.

The real content of the gamma = 2 and gamma = 4 arms is in the *fuel* column
(22/30 and 27/30) and in the hover table: high gamma paralyses, it does not
recklessly re-enter traps.

### Proposed gamma schedule (proposal only, not implemented)

Three candidates, in increasing order of intrusiveness. E1 does not test any of
them; they belong to a follow-up, and after sweep 1b the honest framing is that
any such schedule must first beat the free viability mask, not just gamma = 0.

1. **Distance-gated decay.** `gamma(s) = gamma0 * min(1, d(s, goal) / d0)` with
   `gamma0 = 1`, `d0 = 4`. Keeps full Phi while crossing the trap row and
   releases it inside the goal basin, where option-poorness is the goal's own
   property and not a hazard. Cheapest to implement, one line in `plan`, no new
   state.
2. **Slack-gated decay.** `gamma(s) = gamma0 * (fuel / fuel0)`. Buying options is
   only rational while there is budget left to spend them; this makes the agent
   commit as fuel runs out and directly attacks the 43 % of failures that are
   stalls. Requires `budget_attr` to be wired, which sweep 1 left at the `None`
   default.
3. **Terminal exemption inside Phi.** Treat the goal as an absorbing *success* in
   `phi_cone` and score it as maximally free rather than as a dead end. This is a
   change to the estimator, not to the schedule, and it is the cleanest fix
   conceptually: right now a state one step from victory and a state one step
   from a cliff are both penalised for having no futures. The false-kill
   measurement in sweep 1b sharpens this: `kill_dead` already zeroes a walker
   standing on the goal cell 3.0 % of the time.

## Caveats

1. `budget_attr` was left at the planner default `None`, so `phi_slack` was never
   multiplied in and Phi here is pure `phi_cone`. The BRIEF specified no value.
   Candidate 2 of the schedule proposal is therefore untested, not merely
   unimplemented.
2. n = 30 (sweeps 1, 1b, 2, 3) and n = 15 (sweep 4) give wide CIs on
   proportions: half widths of roughly 0.17 and 0.27. Sweep 4's per-cell rankings
   should not be read as more than a direction; only the pooled h = 5 effect is
   significant.
3. Zero timeouts is a joint parameter-and-behaviour fact. A no-op costs 0.5 fuel
   and a move 1.0, so 40 stay-steps cost 20 of the 24 fuel and a fully stalling
   agent would time out with 4.0 left; no agent stalled that completely. Any
   future comparison of "stall" rates must use the fuel-death-in-goal-column
   measure defined above, or re-run with fuel raised well above max_steps.
4. The equal-budget controls match cost **per decision**. Total episode sim_steps
   are not matched and cannot be: the blind gamma = 0 arms die on move 1, so
   their episode totals are ~13x-30x smaller. Per-decision matching is the
   comparison that constrains the claim; it is also the conservative one, since
   it hands gamma = 0 more compute for every choice it makes.
5. `sim_steps` is an **upper bound** for the Phi arms, not an exact count.
   docs/SPEC.md section 4 invariant 2 and revision 1 both said it "counts every
   simulator call"; `plan()` in fact charges a flat `N * phi_m * phi_h` per Phi
   tick, while `phi_cone` breaks out of a rollout as soon as a continuation dies
   and returns 0 without simulating at all for a non-viable walker. Audited over
   5 decisions at gamma = 1: charged 20 800, actually executed 17 010 `env.step`
   calls, an **18.2 % overcharge** (gamma = 0 is exact, 1600 vs 1600). The error
   is conservative for every claim in this report, since it hands the gamma = 0
   controls real steps the Phi arm never spent, and it makes sweep 2's last row
   (the free viability mask winning at nominal parity) an understatement rather
   than an overstatement. The SPEC wording should be corrected; that file is out
   of this experiment's scope.
6. The pairing across arms is common-random-numbers, not true pairing: the same
   seed produces divergent trajectories once the policies differ. It reduces
   variance and is valid for the delta CIs, but the arms are not matched on any
   post-seed state.
7. One environment, one layout, one fuel level. Nothing here says the interior
   optimum sits near gamma = 0.5 on any other task, and in particular the free
   viability mask is only free because TrapGrid exposes death as a predicate on
   the current state. E5's ablations and E4's alpha = 0 arm are the tests of the
   first point; nothing in the current ladder tests the second.
8. `e1_lib.bootstrap_ci` short-circuits to a zero-width interval when all values
   are equal. That is a known percentile-bootstrap failure, not a 95 % interval.
   Revision 1 reported degenerate arms as "1.000 [1.000, 1.000]" and
   "0.000 [0.000, 0.000]", asserting a certainty the data does not have; those
   cells now carry exact Clopper-Pearson intervals (30/30 -> [0.884, 1.000],
   0/30 -> [0.000, 0.116]) and every binary contrast carries an exact McNemar p
   alongside the bootstrap delta. No conclusion changed, but two of revision 1's
   readings of a "+0.000" bound were reading a boundary artefact.
9. Denominators. The results directory holds **23 sweep JSON files and 555
   episodes** (6 + 4 + 4 + 9 arms; 180 + 120 + 120 + 135 episodes). Three of
   those files are byte-identical reruns of one another
   (`sweep1_gamma_1.json` == `sweep2_gamma1_reference.json` ==
   `sweep3_phi_every_1.json`), so there are **21 distinct arms / 495 distinct
   episodes**; `sweep4_m4_h3.json` is additionally the seed-0..14 prefix of
   `sweep1_gamma_1.json`, not an independent 15-seed run. Revision 2 adds 8 more
   arms and 240 more episodes in `decomp_*.json`. Revision 1's "19 arms and 465
   episodes" was wrong under every counting; the substance (0 timeouts anywhere)
   holds across all 795 episodes on disk.
10. The instrumented runners in `scripts/e1_lib.py` and
    `scripts/decomposition.py` duplicate `fmcphi.planner.run_episode` and
    `fmcphi.planner.plan` in order to record final positions and to separate the
    two Phi factors. `run_e1.py --parity` and `decomposition.py --parity-only`
    assert bit-identical agreement with the library on every configuration that
    exists in both, before any sweep runs.

## Review response

The adversarial review returned "revise" with two blocking issues and ten
overclaims. Point by point.

**Blocking 1 - the report attributed the effect to the exponent, but gamma > 0
also enables `kill_dead`.** Acted on in full. Added sweep 1b, the 2x2 factorial
(mask x exponent) at 30 seeds and 4160 sim/decision in every cell, including both
of the arms the reviewer specified (gamma = 1e-9 with the mask on, and gamma = 1
with the mask path disabled) plus a fourth "neither" cell that computes Phi and
discards it, which rules out rng divergence. The reviewer's numbers replicate
exactly: 19/11, 9/21, 30 trap. The headline is rewritten to say the Phi *layer*
(mask + exponent) produces the win, that the mask carries most of the goal-rate
gain and the exponent is what drives trap rate to 1/30 at the cost of fuel
deaths, and the abstract-level claim now names the mask explicitly. The causal
cone entropy term is no longer presented as the sole cause anywhere.

**Blocking 2 - no death-aware gamma = 0 control existed.** Acted on in full, and
it falsified more than the reviewer asked. Two arms added to sweep 2: gamma = 0
with VR zeroed for non-viable walkers at the baseline 320 sim/decision (23 goal /
6 trap / 1 fuel, statistically indistinguishable from the full Phi arm on goal
and on trap at 1/13 of the cost) and the same mask at the matched 4160
sim/decision (30/30 goal, 0 trap, which beats the Phi arm on goal rate by +0.467
[+0.300, +0.633], McNemar p = 1e-4). The sentence the reviewer flagged is
restated: it is kept for the three *blind* controls, where it is literally what
the data shows, and the implication that no cheap gamma = 0 intervention closes
the gap is retracted with the measured numbers. The headline now leads with this.

**Overclaim: "0 timeouts in all 19 arms and all 465 episodes".** Corrected.
Recounted programmatically: 23 files / 555 episodes, 21 distinct arms / 495
episodes after removing byte-identical reruns. Caveat 9 states both countings and
identifies the duplicate files. I differ from the reviewer on one detail:
`sweep4_m4_h3.json` has n = 15 and is a *prefix* of `sweep1_gamma_1.json`, not a
byte-identical rerun of it, so the dedupe is 21/495 and not 20/480.

**Overclaim: "zero timeouts is a parameter fact by construction".** Corrected in
the verdict and in Caveat 3, with the arithmetic (40 stays cost 20 of 24 fuel)
and a direct simulation confirming a permanently stalling agent survives all 40
steps and times out.

**Overclaim: b_eff rises monotonically with gamma.** Withdrawn and replaced. New
matched-state measurement, 30 `plan()` calls per cell per gamma at three fixed
states. Phi raises b_eff over gamma = 0 everywhere, but it is not monotone in
gamma, and the episode-averaged column is explicitly labelled as confounded with
episode length and visited-state distribution.

**Overclaim: "makes it die faster, 1.2 vs 1.0 steps".** Withdrawn. Paired delta
-0.20 [-0.40, -0.03], Wilcoxon p = 0.063, and the other two controls point the
other way. Reported as noise.

**Overclaim: P2 labelled SUPPORTED.** Relabelled to "untestable against the
pre-registered baseline (floor effect), falsified against the best gamma", in the
verdict heading and in the structured summary, not only in the prose two lines
below.

**Overclaim: the high-gamma stall-then-die surprise.** Withdrawn, with the five
gamma = 2 trap episodes listed as (steps, x, y, fuel) and the note that exactly
one episode out of the whole sweep matches the described mechanism.

**Overclaim: "strict interior minimum in trap rate at gamma = 1".** Withdrawn.
The exact McNemar p for gamma = 2 vs gamma = 1 is 0.125 on 4 discordant pairs and
gamma = 4 does not continue the rise. P3 is now carried by the goal-rate
criterion alone, which is solid.

**Overclaim: goal cell "1.9x below open ground".** Corrected. The pooled
open-ground mean mixed y = 1 (1.514) and y = 2 (2.946); the per-row figure is
1.31x against y = 1, and the report now quotes the stronger and more relevant
vertical contrast, (11,1) = 2.445 versus the goal at 1.16, a factor 2.11.

**Overclaim: zero-width bootstrap intervals on degenerate arms.** Corrected.
Exact Clopper-Pearson intervals for every degenerate proportion, exact McNemar p
next to every binary paired delta, and Caveat 8 names the `bootstrap_ci`
short-circuit as the cause. `e1_lib.bootstrap_ci` itself was left unmodified so
that the existing result files remain reproducible from it; the exact intervals
are computed in `scripts/review_checks.py` and stored in
`results/review_checks.json`.

**Overclaim: `sim_steps` "counts every simulator call".** Corrected in Caveat 5
with an audit: 20 800 charged against 17 010 executed at gamma = 1, an 18.2 %
overcharge; exact at gamma = 0. The wording is now "upper bound", and the
direction of the error is stated (conservative, and it makes the new sweep-2
result an understatement). docs/SPEC.md section 4 carries the same wrong wording
but is out of this experiment's scope to edit.

Nothing in the review was declined.

## Files

- `scripts/e1_lib.py` - instrumented runner, parity check, bootstrap helpers
- `scripts/run_e1.py` - the four sweeps, one JSON per arm
- `scripts/analyze_e1.py` - tables and paired deltas
- `scripts/phi_map_and_hover.py` - Phi map, hover breakdown, P3 contrasts
- `scripts/extra_contrasts.py` - sweep 3 and 4 paired contrasts
- `scripts/decomposition.py` - **revision 2**: mask/exponent factorial, the
  death-aware gamma = 0 controls, and the planner parity assertion for both
- `scripts/review_checks.py` - **revision 2**: denominators, timeout
  reachability, matched-state b_eff, Clopper-Pearson, McNemar, trap forensics,
  `sim_steps` audit
- `results/sweep{1..4}_*.json` - 23 files, 21 distinct arms, per-episode records
- `results/decomp_*.json` - 8 revision-2 arms, 240 episodes
- `results/review_checks.json`, `results/false_kill_rate.json`
- `results/index_all_sweeps.json`, `analysis_deltas.json`, `phi_map.json`,
  `hover_and_p3.json`, `extra_contrasts.json`
