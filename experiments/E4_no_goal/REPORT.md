# E4 — competence with no goal

Run 2026-09-21, **revision 2** after adversarial review. All numbers regenerated
from scratch; nothing below is carried over from revision 1. Raw runs in
`results/*.json`, tables by `scripts/analyze_e4.py` into `results/tables.md`,
occupancy maps by `scripts/behaviour_e4.py`, simulator-cost audit by
`scripts/cost_audit_e4.py`. A "Review response" section at the end lists what
changed and why.

Two maps: `TrapGrid(fuel=24)` and `FunnelRooms(fuel=20)`
(`src/fmcphi/envs/e4_funnel_rooms.py`, owned by E4, registered nowhere).
50 seeds per FMC arm, 500 seeds per policy baseline, `max_steps=60`,
`N=32, M=10, beta=1, phi_m=4, phi_h=3, phi_every=1`, Phi composite with the
fuel-slack factor. Compute wall clock 133 s for the main sweep, 12 s for the
added `a1g0_eqN` arm, 22 s for the diagnostics and cost audit: 2 min 47 s total,
inside the 12-minute budget, no seed count was cut.

**The one change that moved every number: death is now absorbing.** In revision 1
both simulators transitioned normally out of a trap, a pit and out of negative
fuel; death existed only as a predicate in `viable()`. Every gamma = 0 arm
therefore rolled its swarm *through* its own death, because `planner.plan` gates
all viability handling behind `gamma != 0`. That is a defect in the measurement,
not in the agents, and it inflated the whole experiment. `FunnelRooms.step` now
returns a non-viable state unchanged; `TrapGrid` is shared with E1 and E5 and is
not E4's to edit, so it is wrapped by `common.AbsorbingTrapGrid`, which every E4
arm on that map uses. `run_e4.py --selftest` asserts that death is a fixed point
under every action for both causes on both maps.

---

## Verdicts

- **P1 — supported, both maps, and it was never in doubt.** alpha=0 gamma=1
  survives far longer than a uniform random policy. TrapGrid **+21.64 steps
  CI95 [20.53, 22.72]** (31.38 vs 9.74, n=50 vs n=500); Funnel **+11.08 CI95
  [10.26, 11.90]** (27.54 vs 16.46). Hazard death falls from 0.864 to **0.000**
  (TrapGrid, 50/50 episodes end in fuel exhaustion) and from 0.676 to 0.060
  (Funnel).

- **P2 — supported at equal budget on both maps, but the effect is about a
  quarter of what revision 1 reported, and only about half of it is the
  causal-cone term.** At equal charged budget (4160 sim_steps per real step),
  a0g1 minus a0g0_eq is **+6.98 steps CI95 [5.26, 8.92]** on TrapGrid and
  **+5.94 CI95 [5.10, 6.94]** on Funnel. Revision 1 reported +24.62 and +15.64;
  the difference is the absorbing-death fix, which raised the gamma=0 control
  from 5.52 to 24.40 and from 11.68 to 21.60. Hazard difference is
  -0.200 CI95 [-0.320, -0.100] on TrapGrid and **-0.060 CI95 [-0.180, 0.040] on
  Funnel, which includes zero**: on Funnel the survival gain is in steps, not in
  hazard avoidance.

  **Decomposition.** `planner.plan` gates two mechanisms behind one flag: the
  graded factor `relativize(Phi)^gamma`, and `kill_dead`, which hard-zeroes
  every walker with Phi exactly 0. The `killdead_only` arm (gamma = 1e-12, so
  `p_hat**gamma` is within 9.4e-13 of 1 while `kill_dead` stays live and Phi is
  still paid for) separates them, at the same 4160 budget:

  | map | a0g1 minus a0g0_eq | of which: hard viability kill | of which: graded causal cone |
  |---|---|---|---|
  | TrapGrid | +6.98 [5.26, 8.92] | **+3.80 [2.10, 5.74]** (54 %) | **+3.18 [2.38, 3.96]** (46 %) |
  | Funnel | +5.94 [5.10, 6.94] | **+2.86 [2.02, 3.88]** (48 %) | **+3.08 [2.38, 3.76]** (52 %) |

  Both components are individually positive with CIs excluding zero on both
  maps, so the entropy term does contribute something a pure viability filter
  does not. But roughly half of P2 is a hard constraint, not an entropy term,
  and the honest statement of P2 is: *at equal budget the Phi layer buys about 6
  extra steps over canonical Common Sense, half of that from refusing dead
  walkers and half from preferring option-rich ones.* On hazard death the cone
  term's own contribution does not clear zero on either map
  (TrapGrid -0.060 [-0.140, 0.000], Funnel -0.040 [-0.140, 0.060]).

- **P3 — supported, both maps.** The goal rate of alpha=0 gamma=1 is **0/50 on
  TrapGrid and 0/50 on Funnel**, identical to the 0/500 of uniform random and
  the 0/500 of safe random. Over 20 extra logged episodes per map the agent
  visits the goal cell **0 times**; on Funnel it spends **100.0 %** of its
  state-visits in the open first room and never crosses the first doorway. The
  sensitive version is discussed below and does not overturn this.

---

## The result the brief did not ask for, and the one that matters next

**A depth-1 viability filter costing 5 simulator calls per real step matches or
nearly matches the Phi agent at survival.** `safe_random` (uniform over the
actions that leave the next state viable) survives 31.75 CI95 [31.52, 31.99] on
TrapGrid against Phi's 31.38: the difference is **-0.37 CI95 [-0.97, 0.23]**,
i.e. indistinguishable. On Funnel Phi wins by **+2.59 CI95 [2.09, 3.08]**, a
real but small margin. Phi's hazard rate is lower on TrapGrid
(-0.052 [-0.072, -0.034]) and identical on Funnel (+0.000 [-0.060, 0.076]).

The cost ratio is the point. `safe_random` probes one step per action, so it
costs |A| = 5 simulator calls per real step (revision 1 reported 0 for it, which
contradicted SPEC section 4 invariant 2 and is now counted and tabled). The Phi
agent is charged 4160 and measurably spends 3232 (TrapGrid) / 3111 (Funnel).
That is a factor of ~640 in exchange for a difference that is zero on one map
and 2.6 steps on the other. On these two maps the Phi layer's survival
competence is not separable from "do not step into a hole", which is a depth-1
test. A deeper cone or a map where depth-1 safety is insufficient is the obvious
next run, and it is now the main open question of this experiment.

---

## Tables

Full tables including every arm are in `results/tables.md`. The load-bearing
rows (steps and rates with bootstrap CI95, 10k resamples):

### TrapGrid (fuel=24), absorbing death

| arm | n | steps survived (CI95) | hazard death (CI95) | goal rate (CI95) | sim_steps/real step |
|---|---|---|---|---|---|
| uniform random | 500 | 9.74 [8.83, 10.70] | 0.864 [0.834, 0.892] | 0.000 [0.000, 0.000] | 0 |
| safe random (depth-1 filter) | 500 | 31.75 [31.52, 31.99] | 0.052 [0.034, 0.072] | 0.000 [0.000, 0.000] | 5 |
| a=0 g=0 Common Sense | 50 | 24.84 [23.46, 25.82] | 0.160 [0.060, 0.260] | 0.000 [0.000, 0.000] | 320 |
| a=0 g=0 @ equal budget | 50 | 24.40 [22.52, 25.96] | 0.200 [0.100, 0.320] | 0.000 [0.000, 0.000] | 4160 |
| a=1 g=0 canonical FMC | 50 | 2.14 [1.28, 3.26] | 0.920 [0.840, 0.980] | 0.080 [0.020, 0.160] | 320 |
| a=1 g=0 @ equal budget, deeper (M x13) | 50 | 2.86 [1.54, 4.42] | 0.900 [0.820, 0.980] | 0.100 [0.020, 0.180] | 4160 |
| a=1 g=0 @ equal budget, wider (N x13) | 50 | 7.10 [5.42, 8.78] | 0.500 [0.360, 0.640] | 0.500 [0.360, 0.640] | 4160 |
| *kill_dead only (Phi exponent inert)* | 50 | *28.20 [27.68, 28.74]* | *0.060 [0.000, 0.140]* | *0.000 [0.000, 0.000]* | 4160 |
| **a=0 g=1 (E4 agent)** | 50 | **31.38 [30.82, 31.94]** | **0.000 [0.000, 0.000]** | **0.000 [0.000, 0.000]** | 4160 |
| a=0 g=1, no fuel slack | 50 | 29.64 [29.24, 30.04] | 0.020 [0.000, 0.060] | 0.000 [0.000, 0.000] | 4160 |
| a=1 g=1 (SPEC 6 pilot) | 50 | 27.34 [26.04, 28.56] | 0.020 [0.000, 0.060] | 0.420 [0.280, 0.560] | 4160 |

### FunnelRooms (fuel=20), absorbing death

| arm | n | steps survived (CI95) | hazard death (CI95) | goal rate (CI95) | sim_steps/real step |
|---|---|---|---|---|---|
| uniform random | 500 | 16.46 [15.81, 17.14] | 0.676 [0.634, 0.718] | 0.000 [0.000, 0.000] | 0 |
| safe random (depth-1 filter) | 500 | 24.95 [24.81, 25.10] | 0.060 [0.042, 0.082] | 0.000 [0.000, 0.000] | 5 |
| a=0 g=0 Common Sense | 50 | 21.58 [21.32, 21.84] | 0.080 [0.020, 0.160] | 0.000 [0.000, 0.000] | 320 |
| a=0 g=0 @ equal budget | 50 | 21.60 [20.68, 22.26] | 0.120 [0.040, 0.220] | 0.000 [0.000, 0.000] | 4160 |
| a=1 g=0 canonical FMC | 50 | 16.94 [15.92, 17.84] | 0.040 [0.000, 0.100] | 0.920 [0.840, 0.980] | 320 |
| a=1 g=0 @ equal budget, deeper (M x13) | 50 | 17.92 [16.90, 18.90] | 0.040 [0.000, 0.100] | 0.880 [0.780, 0.960] | 4160 |
| a=1 g=0 @ equal budget, wider (N x13) | 50 | 14.00 [14.00, 14.00] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 4160 |
| *kill_dead only (Phi exponent inert)* | 50 | *24.46 [23.96, 24.96]* | *0.100 [0.020, 0.200]* | *0.000 [0.000, 0.000]* | 4160 |
| **a=0 g=1 (E4 agent)** | 50 | **27.54 [27.04, 28.02]** | **0.060 [0.000, 0.140]** | **0.000 [0.000, 0.000]** | 4160 |
| a=0 g=1, no fuel slack | 50 | 26.22 [25.68, 26.74] | 0.140 [0.060, 0.240] | 0.000 [0.000, 0.000] | 4160 |
| a=1 g=1 (SPEC 6 pilot) | 50 | 26.38 [25.68, 27.06] | 0.080 [0.020, 0.160] | 0.060 [0.000, 0.140] | 4160 |

### Survival curves (fraction of episodes still alive at step t)

TrapGrid:

| arm | t=5 | t=10 | t=15 | t=20 | t=25 | t=30 | t=35 |
|---|---|---|---|---|---|---|---|
| uniform random (500) | 0.50 | 0.32 | 0.26 | 0.22 | 0.18 | 0.10 | 0.00 |
| safe random (500) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.78 | 0.16 |
| a=0 g=0 (50) | 0.98 | 0.96 | 0.96 | 0.96 | 0.88 | 0.00 | 0.00 |
| a=0 g=0 eq-budget (50) | 0.98 | 0.92 | 0.90 | 0.90 | 0.86 | 0.02 | 0.00 |
| a=1 g=0 (50) | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 |
| kill_dead only (50) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.20 | 0.00 |
| **a=0 g=1 (50)** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **0.78** | **0.04** |

FunnelRooms:

| arm | t=5 | t=10 | t=15 | t=20 | t=25 | t=30 | t=35 |
|---|---|---|---|---|---|---|---|
| uniform random (500) | 0.96 | 0.73 | 0.55 | 0.42 | 0.22 | 0.00 | 0.00 |
| safe random (500) | 1.00 | 1.00 | 1.00 | 1.00 | 0.57 | 0.00 | 0.00 |
| a=0 g=0 (50) | 1.00 | 1.00 | 1.00 | 1.00 | 0.02 | 0.00 | 0.00 |
| a=0 g=0 eq-budget (50) | 0.98 | 0.98 | 0.98 | 0.96 | 0.04 | 0.00 | 0.00 |
| a=1 g=0 (50) | 0.96 | 0.96 | 0.96 | 0.96 | 0.92 | 0.92 | 0.92 |
| kill_dead only (50) | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 0.00 | 0.00 |
| **a=0 g=1 (50)** | **1.00** | **1.00** | **1.00** | **1.00** | **0.96** | **0.08** | **0.00** |

The a=1 g=0 plateaus are not survival: they are the episodes that ended at the
goal and are counted as alive thereafter (0.08 on TrapGrid, 0.92 on Funnel).

### P3, the sensitive version

A zero goal rate is easy to hit when random is also zero, so the same question
was asked with two continuous statistics that do have power. These were computed
after seeing that the goal rates were all zero, so they are a supplement and not
a pre-registered verdict.

| map | arm | n | min distance to goal reached (CI95) | frac of steps in goal region (CI95) |
|---|---|---|---|---|
| TrapGrid | uniform random | 500 | 9.75 [9.67, 9.83] | 0.000 [0.000, 0.000] |
| TrapGrid | safe random | 500 | 9.51 [9.37, 9.65] | 0.000 [0.000, 0.000] |
| TrapGrid | **a=0 g=1** | 50 | **8.86 [8.38, 9.32]** | **0.000 [0.000, 0.000]** |
| TrapGrid | a=1 g=1 | 50 | 0.90 [0.64, 1.18] | 0.408 [0.372, 0.442] |
| Funnel | uniform random | 500 | 12.65 [12.55, 12.74] | 0.000 [0.000, 0.000] |
| Funnel | safe random | 500 | 12.17 [12.06, 12.28] | 0.000 [0.000, 0.000] |
| Funnel | **a=0 g=1** | 50 | **13.44 [13.24, 13.62]** | **0.000 [0.000, 0.000]** |
| Funnel | a=1 g=0 | 50 | 0.48 [0.00, 1.20] | 0.255 [0.234, 0.274] |

On TrapGrid the no-goal agent gets marginally closer to the goal than *both*
baselines, and **both contrasts exclude zero**: against uniform random
**-0.89 CI95 [-1.37, -0.42]**, and against the length-matched safe random
**-0.65 CI95 [-1.13, -0.17]**. Safe random is the better-controlled comparison,
because it survives 31.75 steps against uniform random's 9.74 and so has
comparable opportunity to wander; uniform random is dead by step 10 and cannot
get far from anywhere. Both TrapGrid contrasts, including the length-matched
one, would look like a leak in isolation.

**The Funnel sign reversal is what rules it out.** On Funnel the same agent ends
up *further* from the goal than uniform random (**+0.79 CI95 [0.58, 1.00]**) and
than safe random (**+1.27 CI95 [1.05, 1.49]**), and spends 0.000 of its steps in
the goal region against a=1 g=0's 0.255. A genuine reward leak through the
geometry would not change sign when the geometry changes. The TrapGrid drift is
an anti-wall effect: Phi is lowest in corners, the start is a corner, so the
swarm slides off it, and on TrapGrid "off the start corner" happens to point at
the goal while on Funnel it points away from the funnel mouth. This is exactly
why the brief asked for a second geometry. Verdict on P3 stands: supported.

### Where the no-goal agent actually goes

FunnelRooms, a=0 g=1, 20 seeds, 571 state-visits, **0 visits to the goal cell**,
region occupancy share `{A: 1.0}`:

```
y=6  -           # # # # # # # # #
y=5  :   X       # # # # # # # # #
y=4  +     X     #     X   # # # #
y=3  @ .                         G
y=2  #  .  X     #     X   # # # #
y=1  :   X       # # # # # # # # #
y=0  -  .        # # # # # # # # #
```

Same map, a=1 g=0 (canonical FMC, goal-seeking), for contrast: it threads the
corridor, occupancy `{A: 0.396, door_AB: 0.062, B: 0.264, door_BC: 0.070,
C: 0.154, goal: 0.054}`, 20 goal-cell visits.

TrapGrid, a=0 g=1: row occupancy `{y=0: 0.040, y=1: 0.056, y=2: 0.191,
y=3: 0.713}`. It climbs off the trap row and loiters in the safe rows near the
start. Canonical a=1 g=0 spends 0.50 of its 84 state-visits on the trap row and
is dead by step 2.14 on average.

---

## Unpredicted findings

1. **A death model in the simulator is worth more than the entropy term.**
   Making death absorbing, three lines and nothing else changed, same seeds,
   same budget, moved canonical Common Sense from 7.62 [5.36, 10.12] to
   **24.84 [23.46, 25.82]** steps on TrapGrid (hazard 0.860 to 0.160) and from
   11.44 [9.90, 13.08] to **21.58 [21.32, 21.84]** on Funnel (hazard 0.860 to
   0.080), and canonical goal-seeking FMC on Funnel from a 0.500 goal rate to
   **0.920**. That single fix is larger than every treatment effect measured in
   this experiment. Revision 1 of this report concluded from the broken version
   that "beta alone is an anti-survival term"; **that claim is withdrawn.** With
   absorbing death, a0g0 beats uniform random on both maps (24.84 vs 9.74 and
   21.58 vs 16.46) at 1/13 of the Phi agent's cost. The finding measured a
   missing death model, not beta.
2. **On Funnel, adding gamma=1 to a goal-seeking agent destroys goal
   attainment:** a=1 g=0 reaches the goal 0.920 [0.840, 0.980] of the time,
   a=1 g=1 only 0.060 [0.000, 0.140]. On TrapGrid the same change goes the other
   way, 0.080 [0.020, 0.160] to 0.420 [0.280, 0.560]. The alpha/gamma standoff
   flagged as an open question in `docs/SPEC.md` section 6 is not a TrapGrid
   quirk about one option-poor corner; it is general, and its sign depends on
   whether the goal sits behind a Phi minimum. FunnelRooms puts the goal at the
   bottom of a monotone Phi funnel and gamma vetoes the whole approach. This
   finding got *stronger* under the fix: the Funnel gap widened from
   0.500 -> 0.060 to 0.920 -> 0.060. E1's gamma schedule is not optional.
3. **Neither ablation identifies a single carrier; the gap splits roughly in
   half.** Dropping `phi_slack` costs 1.74 steps on TrapGrid
   (29.64 [29.24, 30.04] vs 31.38 [30.82, 31.94]) and 1.32 on Funnel
   (26.22 [25.68, 26.74] vs 27.54 [27.04, 28.02]), so slack is not carrying the
   result. That is all the noslack arm shows; it says nothing about the cone.
   The `killdead_only` control is what says something about the cone, and it says
   the cone is worth about half: +3.18 [2.38, 3.96] of TrapGrid's +6.98 and
   +3.08 [2.38, 3.76] of Funnel's +5.94 over a0g0_eq. Revision 1's claim that
   "the cone term is carrying the result" is **withdrawn**: it was inferred from
   the noslack ablation, which cannot support it.
4. **Extra search budget *does* rescue the "bad objective", if it is spent on
   walkers rather than on depth. Revision 1's finding 4 is falsified.** Revision
   1 concluded "extra budget does not rescue a bad objective; the failure is the
   objective, not the search". Two equal-budget variants of canonical FMC, both
   at the same 4160 charged sim_steps per real step as the Phi agent, say
   otherwise:

   | map | a=1 g=0 (320) | deeper, M x13 | wider, N x13 |
   |---|---|---|---|
   | TrapGrid, goal rate | 0.080 [0.020, 0.160] | 0.100 [0.020, 0.180] | **0.500 [0.360, 0.640]** |
   | TrapGrid, hazard | 0.920 [0.840, 0.980] | 0.900 [0.820, 0.980] | **0.500 [0.360, 0.640]** |
   | Funnel, goal rate | 0.920 [0.840, 0.980] | 0.880 [0.780, 0.960] | **1.000 [1.000, 1.000]** |
   | Funnel, hazard | 0.040 [0.000, 0.100] | 0.040 [0.000, 0.100] | **0.000 [0.000, 0.000]** |

   The wider arm solves Funnel outright: 50/50 seeds reach the goal in exactly
   14.00 steps, the length of the optimal route, with a degenerate CI because
   every seed takes it. On TrapGrid it turns a 0.08 goal rate into 0.50 and
   halves the trap-death rate. Revision 1 tested only the deeper variant
   (`M x13`) and generalised from it; the reviewer's own N=416 check was run on
   the pre-fix, non-absorbing simulator, where it also showed nothing. The
   correct statement is narrower and more interesting: **on these maps the
   marginal value of swarm width is large and the marginal value of rollout
   depth is nil**, and at 13x width canonical FMC needs no Phi term to survive
   Funnel at all. That is a finding against the layer, and it is the strongest
   one in this report after the absorbing-death fix itself.

---

## Caveats

1. **Survival on these maps is a saturating metric.** Fuel is the binding
   constraint: an agent that only idles lives 48 steps on TrapGrid and 40 on
   Funnel. Fuel exhaustion ends 50/50 TrapGrid and 47/50 Funnel episodes of the
   Phi arm, i.e. 1.00 and 0.94 of *all* episodes and 1.00 and 1.00 *conditional
   on not dying in a hazard*. Revision 1 quoted "0.92 and 0.94 of episodes" as
   if conditional; they were unconditional. "Steps survived" measures how much
   fuel was spent moving, not an unbounded survival horizon. Hazard-death rate
   is the cleaner metric and is reported alongside everywhere.
2. **P3's headline test has low power by construction.** Random reaches the goal
   0/500 times on both maps, so "not more often than random" cannot be violated
   by a small effect. The min-distance and goal-region statistics were added to
   give the prediction real power. They were computed after seeing that the goal
   rates were all zero; they are not pre-registered, they are reported as a
   supplement, and both maps' contrasts against both baselines are quoted above,
   including the two on TrapGrid whose sign is unfavourable to the conclusion.
3. **n=50 per FMC arm, as briefed.** Baselines got n=500 because they cost
   almost nothing. Differences between a 50-seed arm and a 500-seed arm use
   independent bootstrap resampling; seeds are not paired across arms (each
   arm's seed k drives a different sequence of planner calls), so all CIs are
   unpaired and therefore conservative.
4. **Equal-budget matching is exact per real step in the *charged* count, and
   the charge over-bills the treatment.** Phi is charged `1 + phi_m*phi_h = 13`
   times a gamma=0 tick, so the `_eq` arms use `M=130` against `M=10`, and every
   `_eq`, `killdead_only` and gamma=1 arm reports exactly 4160 charged
   sim_steps per real step. But `phi_cone` returns 0 on a non-viable state
   without a single simulator call and breaks out of a continuation as soon as
   it dies, so the flat charge is nominal. `scripts/cost_audit_e4.py` counts
   every `env.step` for real over 10 seeds: the Phi arm spends **0.777** of its
   charge on TrapGrid (3232 actual against 4160 charged per real step) and
   **0.748** on Funnel (3111 against 4160), while `a0g0_eq` spends exactly
   1.000 of its charge. The direction is conservative, the treatment is billed
   for ~29 % more than it uses, so P2 at equal *actual* budget would be larger,
   not smaller. Revision 1 said the 4160 figures were "verified in the tables";
   the tables only show the charge. Total *episode* budget still differs because
   episode lengths differ, and matching it is impossible when the treatment
   changes the episode length.
5. **Both maps pass the section-5 null-test guard.** Phi over all free cells has
   sd 1.724 (TrapGrid, range 1.125 to 8.000) and sd 1.510 (Funnel, range 0.500
   to 6.727), measured by `run_e4.py --selftest`, which also asserts that
   `common.rollout_fmc` reproduces `fmcphi.planner.run_episode` exactly on three
   seeds and two configurations, that death is a fixed point under every action
   for both death causes on both maps, and that `GAMMA_INERT=1e-12` leaves
   `p_hat**gamma` within 9.4e-13 of 1. Neither environment is a null test.
6. **The `killdead_only` control is a control, not a clean decomposition.**
   `kill_dead` and the graded factor are not orthogonal: both read the same Phi
   vector, and zeroing a walker also changes which walkers the surviving ones
   are compared against under `relativize`. The two contributions add to the
   total here by construction of the contrast chain
   (a0g0_eq -> killdead_only -> a0g1), so calling them "54 % / 46 %" is a
   decomposition of *that path*, not an interaction-free attribution.
7. **One hyper-parameter point.** `N=32, M=10, phi_m=4, phi_h=3` throughout, to
   match the SPEC section 6 pilot. No sweep over `phi_h`; a deeper cone is the
   obvious thing that might separate Phi from the depth-1 safe-random baseline,
   and after this revision that is the single most important open question in
   E4.
8. **Files touched.** `src/fmcphi/envs/e4_funnel_rooms.py` (owned by E4: the
   absorbing guard and its docstring) and everything under
   `experiments/E4_no_goal/`. `src/fmcphi/core.py`, `src/fmcphi/planner.py`,
   `src/fmcphi/phi.py`, `src/fmcphi/envs/trapgrid.py`, `docs/SPEC.md`,
   `README.md`, the existing tests and the other experiment directories were not
   touched. No git command of any kind was run in this revision.
9. `uv run pytest -q` is green, 16/16, after the change.

## Reproduce

```
uv run python experiments/E4_no_goal/scripts/run_e4.py --selftest
uv run python experiments/E4_no_goal/scripts/run_e4.py --seeds 50 --baseline-seeds 500
uv run python experiments/E4_no_goal/scripts/analyze_e4.py
uv run python experiments/E4_no_goal/scripts/behaviour_e4.py
uv run python experiments/E4_no_goal/scripts/cost_audit_e4.py
```

---

## Review response

Verdict received: **revise**, three blocking issues and seven overclaims.

### Blocking issue 1 — non-absorbing death invalidates P2's magnitude

**Acted on, in full.** `FunnelRooms.step` now returns a non-viable state
unchanged (E4 owns that file, and its docstring already claimed irreversibility
as a SPEC section 5 property, so this was E4's defect). `TrapGrid` is shared
with E1 and E5 and is not E4's to edit, so the reviewer's alternative was taken:
`common.AbsorbingTrapGrid` is an explicit absorbing wrapper used by **every** E4
arm on that map, FMC and baseline alike. All seven named arms plus the two
baselines were re-run on both maps, 50 seeds, same seeds, same budgets. The
reviewer's replication numbers reproduced exactly (TrapGrid a0g0
7.62 -> 24.84 [23.46, 25.82], hazard 0.860 -> 0.160; Funnel a0g0
11.44 -> 21.58 [21.32, 21.84], hazard 0.860 -> 0.080; Funnel a1g0
11.60 -> 16.94 [15.92, 17.84], hazard 0.480 -> 0.040). P2 at equal budget is
restated as +6.98 [5.26, 8.92] and +5.94 [5.10, 6.94], down from +24.62 and
+15.64, and the Funnel hazard difference is now -0.060 [-0.180, 0.040],
overlapping zero, stated as such in the P2 verdict. Unpredicted finding 1 is
deleted and rewritten: the beta-is-anti-survival claim is withdrawn in the
report text, not silently dropped. A selftest assertion now guards the property
so it cannot regress.

### Blocking issue 2 — P2 confounds the graded Phi factor with `kill_dead`

**Acted on, in full.** A `killdead_only` arm was added (gamma = `GAMMA_INERT`
= 1e-12; a flag in `planner.plan` would have meant editing shared src that other
agents are working in, so the inert-exponent route was taken and the selftest
asserts `|p_hat**gamma - 1| < 1e-9`). It is in `ARM_ORDER`, in `ARM_LABEL`, in
both main tables, in both survival-curve tables and in four new entries of the
`contrasts` list in `scripts/analyze_e4.py`. The P2 verdict now attributes the
gap explicitly: TrapGrid +3.80 [2.10, 5.74] from the hard viability kill and
+3.18 [2.38, 3.96] from the graded causal cone; Funnel +2.86 [2.02, 3.88] and
+3.08 [2.38, 3.76]. "Phi is doing something beta does not do" is gone.
Unpredicted finding 3 is rewritten: the noslack ablation is now stated to show
only that slack is not the carrier, and the cone's own share comes from the new
control. Note that against the *equal-budget* control the cone's share is
larger than in the reviewer's own decomposition (46 % and 52 % rather than 15 %
and 32 %), because the reviewer measured against a0g0 at 320 sim_steps while
constraint 6 requires the 4160 control; both framings are now derivable from
`results/tables.md`. Caveat 6 states plainly that this is a decomposition along
one contrast path and not an interaction-free attribution.

### Blocking issue 3 — selective reporting in the P3 supplement

**Acted on, in full.** The TrapGrid a0g1-minus-safe_random min-distance contrast
is now quoted in the P3 section next to the uniform-random one
(-0.65 [-1.13, -0.17]; the value shifted from the reviewer's -0.69 because every
arm was re-run on the absorbing environment), with an explicit note that safe
random is the better-controlled, length-matched baseline. The sentence "That
single number, on one map, would look like a leak" is replaced by "Both TrapGrid
contrasts, including the length-matched one, would look like a leak in
isolation. The Funnel sign reversal is what rules it out." The conclusion on P3
is unchanged; only the completeness of the evidence shown changed.

### Overclaims

All seven were removed or substantiated:

1. "Phi is doing something beta does not do" — removed, replaced by the
   decomposition table.
2. "The cone term is carrying the result" — removed, replaced by the
   `killdead_only` contrast and an explicit statement of what the noslack arm
   can and cannot support.
3. "beta alone is an anti-survival term" — withdrawn in text, with the new
   numbers showing the opposite.
4. "The failure is the objective, not the search" — **falsified on both maps,
   not merely weakened.** The reviewer noted that the absorbing fix rescues
   a0g0 at the original budget, so "the objective" was the wrong diagnosis.
   Testing further, an `a1g0_eqN` arm was added (equal charged budget spent on
   swarm width, N x13 instead of M x13): canonical FMC then reaches the goal
   0.500 [0.360, 0.640] of the time on TrapGrid, up from 0.080, and
   1.000 [1.000, 1.000] on Funnel in exactly 14.00 steps, the optimal route.
   Revision 1 had tested only the deeper variant. Unpredicted finding 4 is
   rewritten as a finding *against* the layer.
5. safe_random's "zero simulator cost" — `rollout_policy` now counts its probe
   calls, it is tabled at 5 sim_steps per real step on both maps, and the
   qualitative point (5 against 4160) is restated with the corrected number.
6. Caveat 1's fuel fractions — corrected, both unconditional (1.00 and 0.94) and
   conditional (1.00 and 1.00) figures given, with the revision-1 error named.
7. Caveat 4's "verified" — replaced by a measurement. `scripts/cost_audit_e4.py`
   counts every simulator call: the Phi arm spends 0.777 (TrapGrid) and 0.748
   (Funnel) of its nominal charge, `a0g0_eq` spends 1.000, so the treatment is
   over-billed by ~29 % and the equal-budget comparison is conservative.

### Nothing was declined

There is no blocking issue or overclaim from this review that was left unactioned.
Two limits of what was done are worth stating rather than leaving implicit:
`src/fmcphi/planner.py` was not modified, so `killdead_only` is implemented with
an inert exponent instead of the `phi_kill_only` flag the reviewer suggested
(functionally equivalent to within 1e-12, asserted in the selftest); and
`src/fmcphi/envs/trapgrid.py` was not modified, so the absorbing guard on that
map lives in E4's own `common.AbsorbingTrapGrid` rather than in the environment
itself, which means **E1 and E5 are still running against a non-absorbing
TrapGrid and their gamma=0 arms are likely to carry the same defect.** That is
outside E4's ownership; it is flagged here for the orchestrator.
