# E5 — ablations on the Phi layer

**On TrapGrid the effect is carried by the `kill_dead` viability mask, not by
the causal-cone entropy. Replacing `phi_cone` with the binary indicator
`Phi = 1{env.viable(state)}` — no rollouts, no perplexity, no survival
weighting, zero `env.step` calls — matches or beats the real Phi everywhere we
measured it: goal 0.83 [0.70, 0.97] at equal N against real's 0.53 [0.37, 0.70]
(paired diff -0.30 [-0.50, -0.13], n=30, CI excludes zero), and 0.97
[0.90, 1.00] with 1/30 trap deaths at equal measured simulator budget. The
causal cone, the perplexity, the survival weighting and even the depth-1 viable
action count are not shown to buy anything over knowing whether the current
state is alive.**

Second submission, 2026-09-21, after an adversarial review returned "revise".
Everything below is re-run from scratch with the two missing controls added; the
`Review response` section at the end lists each blocking issue and what was done
about it. The previous headline ("the effect is carried by the
state-correspondence of Phi") is **withdrawn**: it was not what the arms tested.

Runs on `TrapGrid(width=12, height=4, fuel=24)`, `alpha=beta=1`, `M=10`,
`phi_m=4`, `phi_h=3`, `phi_every=1`, `budget_attr=None`, `max_steps=60`,
30 paired seeds (1000..1029), `gamma = 1.0` unless stated. `gamma` was fixed a
priori and **not** inherited from E1, which is a deviation from the BRIEF; the
gamma-matched re-run demanded by the review is Table 5.

## What the arms actually ablate

The decisive arms of the first submission do not isolate what they were
advertised to isolate. `phi_cone` returns **exactly 0** on a non-viable state,
and `kill_dead` zeroes the virtual reward of exactly those walkers. Permuting
the Phi vector (`shuffled`), replacing it with its strictly positive mean
(`constant`) or with a continuous uniform draw (`noise`) destroys or relocates
those zeros, so each of those arms removes the hard viability mask **and** the
state-correspondence of the graded factor at the same time. Two mechanisms,
one arm.

The two new arms separate them:

| arm | graded Phi factor in the VR | `kill_dead` mask | isolates |
|---|---|---|---|
| `shuffled_keepmask`, `noise_keepmask` | destroyed | intact, on the true Phi | the graded factor |
| `mask`, `mask_matched`, `mask_true_matched` | `1{viable}` (constant among survivors) | intact | the death mask |

For `mask` the graded factor is provably inert: all viable walkers get
`Phi = 1`, so among the walkers that survive `kill_dead` the relativized Phi is
a single constant, `clone_step` uses only the ratio `(VR_k - VR_i) / VR_i` and
is invariant to a uniform rescale, and therefore the arm is *exactly* canonical
FMC plus "a walker standing on a dead state always clones away". It has no
tunable exponent: measured goal rate is 0.83 [0.70, 0.97] at `gamma` = 0.25,
0.5 and 1.0 alike (Table 5), as the argument predicts.

## Verdicts, one line per pre-registered prediction

| id | prediction (BRIEF) | verdict |
|---|---|---|
| **P1** | arm 1 *shuffled Phi*: the effect disappears | **literally supported, but not diagnostic.** `shuffled` collapses to canonical FMC (goal 0.00 [0.00, 0.00] vs real 0.53 [0.37, 0.70]; trap 30/30). It is now shown that this is the mask being removed, not the state-correspondence: with the mask kept, `shuffled_keepmask` reaches goal 0.63 [0.47, 0.80], paired diff real minus arm -0.10 [-0.30, +0.10], n=30 |
| **P2** | arm 2 *constant Phi*: no effect expected | **supported, and a harness check only.** goal 0.00, trap 30/30, indistinguishable from `gamma=0`. `relativize` of a constant vector returns exactly ones, so the graded factor is the identity and the only thing the arm changes relative to canonical FMC is that no walker is ever killed |
| **P3** | arm 3 *no survival weighting*: worse than weighted, better than `gamma=0` | **falsified as stated.** Better than `gamma=0`: yes (0.60 [0.43, 0.77] vs 0.00). Worse than weighted: no. Paired diff real minus `nosurv` = -0.07 [-0.30, +0.17] (n=30), -0.08 [-0.21, +0.06] (n=100); the point estimate favours the *unweighted* arm |
| **P4** | arm 4 *cheap proxy*: if it matches, `phi_cone` is not needed | **supported: it matches.** At equal N, paired diff -0.10 [-0.33, +0.13]. At equal measured budget (`cheap_true_matched`, N=60), goal 0.77 [0.60, 0.90] vs real 0.53 [0.37, 0.70], paired diff **-0.23 [-0.47, +0.00], not distinguishable from zero at the pre-registered n=30**; the marginal CIs overlap. "Matches or exceeds" is what the primary analysis supports, "beats" is not. At the secondary n=100 the same contrast is -0.25 [-0.37, -0.13] |
| **P5** | arm 5 *random noise*: the null | **literally supported, but not diagnostic,** for the same reason as P1. `noise` gives goal 0.00, trap 100%. With the mask kept, `noise_keepmask` reaches 0.57 [0.40, 0.73], paired diff -0.03 [-0.30, +0.23] |
| **decision rule** | layer supported only if arms 1 and 5 both lose to real Phi with non-overlapping CI95 | **satisfied as written, and the rule is not sufficient.** Both arms lose at n=30 and n=100 and under the exact binomial bound. But both also remove the viability mask, and the mask alone (`mask`, 0 extra `env.step` calls) *beats* real Phi. The rule cannot distinguish "the causal cone works" from "zeroing dead walkers works", and the new arms show it is the second |

**New, unpredicted, and the main result:** a zero-cost, zero-rollout Phi beats
the real one. See Finding 1.

## Table 1 — outcome rates, n=30 paired seeds, mean [CI95, percentile bootstrap 10k]

Budget columns are per real decision. *Nominal* is what `planner.plan` and
`plan_e5` add up front (`M*N*(1 + m*h)`); *measured* is the number of
`env.step` calls actually made, counted by the `StepCounter` wrapper. They
differ because `phi_cone` returns 0 after zero steps on a non-viable state and
breaks its rollout early when a continuation dies.

| arm | mode | N | n | nominal/dec | measured/dec | meas/nom | goal | trap | fuel |
|---|---|---|---|---|---|---|---|---|---|
| `off` | gamma=0 | 32 | 30 | 320 | 320 | 1.000 | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] |
| `off_matched` | gamma=0 | 416 | 30 | 4160 | 4160 | 1.000 | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] |
| `off_true_matched` | gamma=0 | 329 | 30 | 3290 | 3290 | 1.000 | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] |
| **`real`** | `phi_cone`, survival-weighted | 32 | 30 | 4160 | **3291** | 0.791 | **0.53 [0.37, 0.70]** | 0.10 [0.00, 0.23] | 0.37 [0.20, 0.53] |
| `shuffled` (arm 1) | Phi permuted, mask destroyed | 32 | 30 | 4160 | 2564 | 0.616 | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] |
| `constant` (arm 2) | Phi = swarm mean, mask destroyed | 32 | 30 | 4160 | 2166 | 0.521 | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] |
| `noise` (arm 5) | uniform on Phi's support, mask destroyed | 32 | 30 | 4160 | 2245 | 0.540 | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] |
| **`shuffled_keepmask`** (arm A) | Phi permuted in the VR, mask intact | 32 | 30 | 4160 | 3314 | 0.797 | **0.63 [0.47, 0.80]** | 0.23 [0.10, 0.40] | 0.13 [0.03, 0.27] |
| **`noise_keepmask`** (arm A) | Phi noised in the VR, mask intact | 32 | 30 | 4160 | 3209 | 0.772 | **0.57 [0.40, 0.73]** | 0.17 [0.03, 0.30] | 0.27 [0.13, 0.43] |
| `nosurv` (arm 3) | `weight_by_survival=False` | 32 | 30 | 4160 | 3264 | 0.785 | 0.60 [0.43, 0.77] | 0.07 [0.00, 0.17] | 0.33 [0.17, 0.50] |
| `cheap` (arm 4) | `phi_viable_actions` | 32 | 30 | 1920 | 1696 | 0.883 | 0.63 [0.47, 0.80] | 0.10 [0.00, 0.20] | 0.27 [0.13, 0.43] |
| `cheap_matched` (arm 4) | equal *nominal* budget | 69 | 30 | 4140 | 3764 | 0.909 | 0.77 [0.60, 0.90] | 0.23 [0.10, 0.40] | 0.00 [0.00, 0.00] |
| `cheap_true_matched` (arm 4) | equal *measured* budget | 60 | 30 | 3600 | 3243 | 0.901 | 0.77 [0.60, 0.90] | 0.13 [0.03, 0.27] | 0.10 [0.00, 0.23] |
| **`mask`** (arm B) | `Phi = 1{viable}`, 0 extra steps | 32 | 30 | **320** | **320** | 1.000 | **0.83 [0.70, 0.97]** | 0.17 [0.03, 0.30] | 0.00 [0.00, 0.00] |
| **`mask_matched`** (arm B) | equal *nominal* budget | 416 | 30 | 4160 | 4160 | 1.000 | **1.00 [1.00, 1.00]** | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| **`mask_true_matched`** (arm B) | equal *measured* budget | 329 | 30 | 3290 | 3290 | 1.000 | **0.97 [0.90, 1.00]** | 0.03 [0.00, 0.10] | 0.00 [0.00, 0.00] |

Rates are per-seed 0/1 indicators. Six arms have a degenerate percentile
bootstrap CI because every seed gave the same outcome; the conservative exact
(Clopper-Pearson) interval is **[0.000, 0.116] for 0/30 and [0.884, 1.000] for
30/30**, and every comparison drawn below survives under it. Exact intervals
for all arms are printed by `scripts/review_stats.py`; the ones that matter:
`real` goal 16/30 exact [0.343, 0.717], `mask` goal 25/30 [0.653, 0.944],
`mask_true_matched` goal 29/30 [0.828, 0.999].

`mask` uses zero `env.step` calls for Phi, but it does call `env.viable`, the
same predicate `phi_cone` calls before its first rollout. The claim is "no
simulator rollouts", not "no environment knowledge at all".

## Table 2 — paired contrasts, n=30 (positive = the reference arm is higher)

Paired bootstrap over the 30 shared seeds, 10k resamples.

| arm | vs `real`: d goal | vs `real`: d trap | vs `mask_true_matched`: d goal |
|---|---|---|---|
| `off` / `off_matched` / `off_true_matched` | +0.53 [+0.37, +0.70] | -0.90 [-1.00, -0.77] | +0.97 [+0.90, +1.00] |
| `shuffled` (arm 1) | +0.53 [+0.37, +0.70] | -0.90 [-1.00, -0.77] | +0.97 [+0.90, +1.00] |
| `constant` (arm 2) | +0.53 [+0.37, +0.70] | -0.90 [-1.00, -0.77] | +0.97 [+0.90, +1.00] |
| `noise` (arm 5) | +0.53 [+0.37, +0.70] | -0.90 [-1.00, -0.77] | +0.97 [+0.90, +1.00] |
| `shuffled_keepmask` (arm A) | **-0.10 [-0.30, +0.10]** | -0.13 [-0.33, +0.07] | +0.33 [+0.13, +0.53] |
| `noise_keepmask` (arm A) | **-0.03 [-0.30, +0.23]** | -0.07 [-0.23, +0.10] | +0.40 [+0.20, +0.60] |
| `nosurv` (arm 3) | -0.07 [-0.30, +0.17] | +0.03 [-0.10, +0.17] | +0.37 [+0.17, +0.57] |
| `cheap` (arm 4) | -0.10 [-0.33, +0.13] | +0.00 [-0.17, +0.17] | +0.33 [+0.13, +0.53] |
| `cheap_matched` (arm 4) | -0.23 [-0.47, +0.00] | -0.13 [-0.33, +0.07] | +0.20 [+0.03, +0.37] |
| `cheap_true_matched` (arm 4) | -0.23 [-0.47, +0.00] | -0.03 [-0.20, +0.13] | +0.20 [+0.03, +0.37] |
| `mask` (arm B) | **-0.30 [-0.50, -0.13]** | -0.07 [-0.23, +0.10] | +0.13 [+0.00, +0.30] |
| `mask_matched` (arm B) | **-0.47 [-0.63, -0.30]** | +0.10 [+0.00, +0.23] | -0.03 [-0.10, +0.00] |
| `mask_true_matched` (arm B) | **-0.43 [-0.63, -0.23]** | +0.07 [-0.07, +0.20] | — |
| `real` | — | — | +0.43 [+0.23, +0.63] |

Three contrasts exclude zero at the pre-registered n=30, and all three are the
mask beating the cone. No contrast anywhere shows the cone beating anything
except the arms that had their mask removed.

## Table 3 — episode diagnostics, n=30

| arm | steps | mean b_eff | mean Phi (over its own decisions) | mean ESS (of N) |
|---|---|---|---|---|
| `off` | 1.73 [1.33, 2.23] | 1.65 [1.49, 1.81] | n/a | 17.35 [16.94, 17.72] |
| `off_matched` | 1.00 [1.00, 1.00] | 2.04 [1.91, 2.19] | n/a | 216.14 [214.71, 217.62] |
| `off_true_matched` | 1.00 [1.00, 1.00] | 2.05 [1.90, 2.21] | n/a | 171.27 [169.83, 172.74] |
| `real` | 23.43 [20.27, 26.17] | 2.15 [2.06, 2.25] | 1.878 [1.796, 1.964] over 703 decisions | 17.28 [16.36, 18.21] |
| `shuffled` | 2.37 [1.77, 3.07] | 2.57 [2.35, 2.79] | 1.204 over 71 decisions | 9.69 [9.09, 10.34] |
| `constant` | 1.43 [1.17, 1.73] | 1.63 [1.44, 1.82] | 0.956 over 43 decisions | 17.05 [16.63, 17.47] |
| `noise` | 1.50 [1.07, 2.27] | 1.78 [1.57, 2.00] | 1.918 over 45 decisions | 12.80 [11.98, 13.63] |
| `shuffled_keepmask` | 17.93 [14.90, 20.80] | 2.14 [2.05, 2.23] | 1.83 [1.75, 1.91] over 538 decisions | 13.42 [12.41, 14.53] |
| `noise_keepmask` | 20.70 [17.80, 23.27] | 2.22 [2.13, 2.32] | 1.78 [1.70, 1.85] over 621 decisions | 14.72 [13.56, 15.89] |
| `nosurv` | 22.80 [20.17, 25.13] | 2.17 [2.05, 2.32] | 2.09 [2.00, 2.18] | 16.57 [15.49, 17.64] |
| `cheap` | 21.40 [18.37, 24.30] | 2.07 [1.95, 2.23] | 3.91 [3.75, 4.06] | 17.09 [16.30, 17.88] |
| `cheap_matched` | 16.67 [13.47, 19.57] | 2.51 [2.35, 2.68] | 4.06 [3.97, 4.14] | 33.93 [33.10, 34.81] |
| `cheap_true_matched` | 18.87 [15.87, 21.63] | 2.35 [2.24, 2.49] | 4.00 [3.89, 4.10] | 30.12 [29.02, 31.30] |
| `mask` | 17.87 [15.30, 20.23] | 1.99 [1.90, 2.10] | 0.860 over 536 decisions | 15.74 [15.22, 16.34] |
| `mask_matched` | 13.27 [13.10, 13.47] | 2.52 [2.48, 2.56] | 0.877 | 197.52 [196.95, 198.14] |
| `mask_true_matched` | 12.97 [12.13, 13.50] | 2.59 [2.47, 2.77] | 0.876 | 155.52 [154.36, 156.46] |

**`mean Phi` is not comparable across arms.** Each arm's mean is taken over the
states that arm actually visited, and the collapsed arms die almost
immediately: `noise`'s 1.918 is averaged over 45 decisions in total, against
`real`'s 1.878 over 703. The similarity of those two numbers is not evidence of
a matched null; it only says the noise arm was drawn from the real Phi's
support at the handful of near-start states it saw before dying. `mask`'s 0.860
is the mean of a 0/1 indicator: about 14% of walkers stand on a dead state at
decision time.

## Table 4 — the cost accounting, measured

Every "equal budget" statement in this report is now measured, not nominal.

| arm | nominal/dec | measured/dec | measured / nominal |
|---|---|---|---|
| `real` | 4160 | 3291 | 0.791 |
| `shuffled` | 4160 | 2564 | 0.616 |
| `constant` | 4160 | 2166 | 0.521 |
| `noise` | 4160 | 2245 | 0.540 |
| `cheap` | 1920 | 1696 | 0.883 |
| `cheap_matched` | 4140 | 3764 | 0.909 |
| `cheap_true_matched` | 3600 | 3243 | 0.901 |
| `mask`, `mask_matched`, `mask_true_matched`, all `off` arms | = M*N | = M*N | 1.000 |

The first submission called `cheap_matched` (nominal 4140) equal-budget against
`real` (nominal 4160). Measured, that comparison gave the proxy 3764 calls
against 3291, about **14% more**, so the wording was wrong. `cheap_true_matched`
(N=60, measured 3243 against 3291) and `mask_true_matched` (N=329, measured
3290 against 3291) are the honest equal-budget arms, and both reproduce the
result: 0.77 [0.60, 0.90] and 0.97 [0.90, 1.00] against real's 0.53
[0.37, 0.70].

## Table 5 — gamma-matched comparison, n=30 (`scripts/gamma_check.py`)

The BRIEF pre-registered "at the best gamma from E1"; E5 ran at gamma = 1.0
because E1 had not reported. E1's REPORT.md puts the goal-rate optimum at
gamma = 0.25-0.5 (0.667 [0.500, 0.833]) and states that above gamma ~ 1 the
layer is a brake, so gamma = 1.0 is on the brake side of `phi_cone`'s own
optimum. Each estimator is therefore re-run at gamma in {0.25, 0.5, 1.0} and
compared at its own best gamma.

| arm | gamma=0.25 | gamma=0.5 | gamma=1.0 |
|---|---|---|---|
| `real` | 0.63 [0.47, 0.80] | **0.70 [0.53, 0.87]** | 0.53 [0.37, 0.70] |
| `cheap` (N=32) | **0.83 [0.70, 0.97]** | 0.77 [0.60, 0.90] | 0.63 [0.47, 0.80] |
| `cheap_matched` (N=69) | 0.83 [0.70, 0.97] | **0.87 [0.73, 0.97]** | 0.77 [0.60, 0.90] |
| `mask` (N=32) | **0.83 [0.70, 0.97]** | 0.83 [0.70, 0.97] | 0.83 [0.70, 0.97] |
| `mask_matched` (N=416) | **1.00 [1.00, 1.00]** | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |

Goal rate; bold is each row's best. Paired contrasts, each arm at its own best
gamma, real at gamma = 0.5:

| contrast | d goal rate [CI95] |
|---|---|
| `real`@0.5 minus `cheap`@0.25 | -0.13 [-0.37, +0.10] |
| `real`@0.5 minus `cheap_matched`@0.5 | -0.17 [-0.37, +0.03] |
| `real`@0.5 minus `mask`@0.25 | -0.13 [-0.37, +0.10] |
| `real`@0.5 minus `mask_matched`@0.25 | **-0.30 [-0.47, -0.13]** |

Gamma-matched, the cheap proxy and the mask at equal N are **indistinguishable
from real Phi at its own best gamma** (CIs contain zero), and the mask at equal
nominal budget still beats it with the CI excluding zero. Tuning gamma raises
real Phi from 0.53 to 0.70 and closes the gap at equal N; it does not reverse
any comparison. `mask` is gamma-invariant to three decimal places, as the
argument in "What the arms actually ablate" predicts.

## Table 6 — Phi is state-dependent, and collapses at low fuel

`phi_cone(m=4, h=3)`, 400 rng repeats per cell, probed at five fuel levels
(episodes start at fuel = 24 and run down to 0). Row 0 is the trap row.

fuel = 20 (unchanged from the first submission):

| y | x0 | x1 | x2 | x3 | x4 | x5 | x6 | x7 | x8 | x9 | x10 | x11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 2.86 | 3.11 | 3.27 | 3.25 | 3.16 | 3.22 | 3.26 | 3.23 | 3.23 | 3.23 | 3.09 | 2.86 |
| 2 | 2.95 | 2.94 | 3.00 | 2.94 | 2.94 | 3.01 | 2.93 | 2.88 | 2.95 | 2.97 | 2.95 | 3.08 |
| 1 | 2.45 | 1.67 | 1.49 | 1.46 | 1.56 | 1.50 | 1.53 | 1.50 | 1.55 | 1.48 | 1.71 | 2.31 |
| 0 | 1.18 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.16 |

Summary by fuel level. The two populations are now reported separately, which
the first submission conflated in one sentence:

| fuel | range over all 48 cells | range over the 38 alive cells | mean, sd (alive) | P(Phi = 0 exactly given an alive cell) | trap cells with Phi = 0 |
|---|---|---|---|---|---|
| 20 | [0.00, 3.27] | [1.16, 3.27] | 2.52, 0.736 | 0.007 | 10/10 |
| 8 | [0.00, 3.35] | [1.12, 3.35] | 2.54, 0.753 | 0.008 | 10/10 |
| 4 | [0.00, 3.30] | [1.11, 3.30] | 2.52, 0.733 | 0.007 | 10/10 |
| 3 | [0.00, 2.15] | [0.54, 2.15] | 1.35, 0.536 | 0.063 | 10/10 |
| 2 | [0.00, 0.22] | [0.00, 0.22] | 0.05, 0.063 | **0.830** | 10/10 |

Phi is not constant on this environment, so it does not vanish after
`relativize` and docs/SPEC.md section 5's null-test trap is discharged. But the
low-fuel regime, which the first submission never probed, is where the
0.32-0.37 fuel-death rate is decided, and there `phi_cone` stops being a graded
signal: at fuel = 2 a depth-3 rollout usually exhausts the budget, so **83% of
individual estimates on living cells are exactly 0** and `kill_dead` zeroes most
of the swarm at once. The binary mask does not have this failure mode, since a
state with fuel = 2 is still viable, and that is the most likely mechanism
behind the one result where real Phi is clearly worse: fuel deaths 0.37
[0.20, 0.53] for `real` against 0.00 [0.00, 0.00] for every `mask` arm. This is
a hypothesis consistent with the measurements, not a separately tested claim.

Per-cell values with CI95 at every fuel level: `results/E5_phi_landscape.json`.

## Table 7 — robustness at n=100 (secondary, not pre-registered)

Every verdict above is unchanged. The n=100 sweep was added after seeing n=30
in the first submission, so it is reported as secondary; **the primary analysis
is the n=30 one** and no claim in this report rests on n=100 alone.

| arm | goal | trap | fuel | paired d goal vs `real` |
|---|---|---|---|---|
| `off` | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | +0.57 [+0.47, +0.67] |
| `off_matched` | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | +0.57 [+0.47, +0.67] |
| `off_true_matched` | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | +0.57 [+0.47, +0.67] |
| `real` | 0.57 [0.47, 0.67] | 0.11 [0.05, 0.17] | 0.32 [0.23, 0.41] | — |
| `shuffled` | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | +0.57 [+0.47, +0.67] |
| `constant` | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | +0.57 [+0.47, +0.67] |
| `noise` | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | +0.57 [+0.47, +0.67] |
| `shuffled_keepmask` | 0.73 [0.64, 0.81] | 0.14 [0.08, 0.21] | 0.13 [0.07, 0.20] | -0.16 [-0.28, -0.04] |
| `noise_keepmask` | 0.65 [0.55, 0.74] | 0.18 [0.11, 0.26] | 0.17 [0.10, 0.25] | -0.08 [-0.21, +0.05] |
| `nosurv` | 0.65 [0.56, 0.74] | 0.15 [0.08, 0.22] | 0.20 [0.13, 0.28] | -0.08 [-0.21, +0.06] |
| `cheap` | 0.65 [0.55, 0.74] | 0.16 [0.09, 0.23] | 0.19 [0.12, 0.27] | -0.08 [-0.20, +0.04] |
| `cheap_matched` | 0.83 [0.75, 0.90] | 0.15 [0.08, 0.22] | 0.02 [0.00, 0.05] | -0.26 [-0.38, -0.14] |
| `cheap_true_matched` | 0.82 [0.74, 0.89] | 0.13 [0.07, 0.20] | 0.05 [0.01, 0.10] | -0.25 [-0.37, -0.13] |
| `mask` | 0.80 [0.72, 0.87] | 0.18 [0.11, 0.26] | 0.02 [0.00, 0.05] | -0.23 [-0.34, -0.12] |
| `mask_matched` | 0.99 [0.97, 1.00] | 0.01 [0.00, 0.03] | 0.00 [0.00, 0.00] | -0.42 [-0.52, -0.32] |
| `mask_true_matched` | 0.98 [0.95, 1.00] | 0.02 [0.00, 0.05] | 0.00 [0.00, 0.00] | -0.41 [-0.51, -0.31] |

At n=100 the `shuffled_keepmask` arm is *better* than real Phi on goal rate
(-0.16 [-0.28, -0.04], CI excludes zero). Destroying the state-correspondence of
the graded factor, while keeping the mask, does not hurt; if anything it helps.

## Findings

1. **A zero-rollout, zero-parameter Phi beats the real one.**
   `Phi = 1{env.viable(state)}` costs no `env.step` calls, has no `m`, no `h`,
   no `weight_by_survival`, and no effective `gamma` (Table 5). At equal N it
   reaches goal 0.83 [0.70, 0.97] against real's 0.53 [0.37, 0.70], paired diff
   -0.30 [-0.50, -0.13] with the CI excluding zero at the pre-registered n=30.
   At equal *measured* budget it reaches 0.97 [0.90, 1.00] with 1/30 trap
   deaths and 0 fuel deaths. **The Phi layer as implemented is supported only
   insofar as it supplies a viability mask; the causal-cone estimator is not
   justified by this data.** If E2 and E3 depend on `phi_cone` specifically,
   that dependency is unsupported and they should carry a `1{viable}` arm.
2. **The graded Phi factor contributes nothing measurable on TrapGrid.**
   Shuffling it (`shuffled_keepmask`, 0.63 [0.47, 0.80]) or replacing it with
   pure noise (`noise_keepmask`, 0.57 [0.40, 0.73]) while leaving `kill_dead`
   on the true Phi costs nothing relative to real Phi (paired -0.10
   [-0.30, +0.10] and -0.03 [-0.30, +0.23]). The whole collapse reported in the
   first submission was the mask being removed.
3. **The cheap proxy matches real Phi; "beats" is not established at the
   pre-registered n.** At equal measured budget the paired diff is -0.23
   [-0.47, +0.00] (n=30): the interval touches zero and the marginal CIs
   overlap. It is -0.25 [-0.37, -0.13] at the secondary n=100. The defensible
   statement is: *there is no evidence that `phi_cone` is better than
   `phi_viable_actions`, and some evidence that it is worse.*
4. **More compute alone does nothing for canonical FMC here.** `off_matched`
   (N=416) and `off_true_matched` (N=329) both die in the trap on move 1 in
   30/30 episodes, with a *lower* step count than `off` (1.00 vs 1.73). What
   the mask adds is not a proxy for search budget: `mask` gets the same 320
   steps per decision as `off` and reaches 0.83.
5. **Survival weighting is inert, and the reason generalises.** `nosurv`
   matches `real` (-0.07 [-0.30, +0.17]). The trap already returns Phi = 0
   through the `env.viable` gate at the top of `phi_cone`, and `kill_dead` acts
   on that; `p_surv` only reorders *living* states, which is the same graded
   factor finding 2 shows to be inert.
6. **Real Phi does not eliminate trap deaths, and it causes fuel deaths.**
   Against SPEC section 6's 10-seed pilot ("trap deaths go to zero"), we see
   0.10 [0.00, 0.23] at n=30 and 0.11 [0.05, 0.17] at n=100, and a fuel-death
   rate of 0.37 [0.20, 0.53] that every mask arm drives to 0.00. Table 6 gives
   the likely mechanism: below fuel ~3 the depth-3 rollouts exhaust the budget
   and `phi_cone` returns exactly 0 on 83% of living cells, so the mask it
   induces stops discriminating and starts killing the whole swarm.
7. **Shuffled Phi is not just useless, it is expensive noise.** Its mean ESS is
   9.69 [9.09, 10.34] against 17.28 [16.36, 18.21] for real, and it pays 2564
   measured steps per decision to deliver the `gamma=0` outcome.

## Caveats

1. One environment, one (N, M), one fuel level, one reward shape. TrapGrid's
   viability predicate is exact and free; in an environment where "is this
   state alive" is itself expensive or only estimable by rollout, the mask arm
   is not available and the comparison would look different. Nothing here says
   the ranking transfers. That is what E2 and E3 are for, and they should now
   include the mask arm.
2. `gamma = 1.0` for Tables 1-4 and 7 deviates from the BRIEF's "best gamma
   from E1". Table 5 runs the gamma-matched comparison the review asked for and
   no conclusion changes; real Phi's best gamma on this grid is 0.5 (0.70
   [0.53, 0.87]) and the mask is gamma-invariant.
3. `budget_attr=None` throughout, so `phi_slack` is never multiplied in. A
   `phi_cone * phi_slack` composite might fix the low-fuel collapse of Table 6;
   untested here, and it would be a new experiment, not a re-analysis.
4. The `mask` arm's advantage is measured only on goal rate and death rates.
   It plans shorter episodes (13.0 vs 23.4 steps) because it walks the safe
   corridor without hesitating; on a task where option-preservation mattered
   after reaching the goal it might well lose. TrapGrid cannot see that.
5. The ablation arms consume extra RNG draws, so they diverge from `real`'s
   random stream even on the same seed. Pairing is seed-blocked, not
   matched-trajectory.
6. `noise` samples uniformly on [min, max] of the real Phi vector recomputed
   each tick, so it shares Phi's support but not its marginal shape, and (per
   Table 3) it only ever sees the first 1.5 decisions of an episode.
   `shuffled` is the arm that preserves the marginal exactly.
7. `constant` is a null by construction: `relativize` returns *exactly ones*
   when the input has zero standard deviation, so the Phi factor is the
   identity, not a rescale. (The first submission said "uniform rescale"; the
   conclusion was unaffected but the reason was wrong.) Its agreement with
   `off` is a harness check.
8. Exact binomial intervals are reported alongside the degenerate [0.00, 0.00]
   bootstrap intervals. Do not quote the latter as zero-width.
9. Bootstrap seeds are now derived from `zlib.crc32` rather than `hash()`,
   which is salted per process. The first submission's CI endpoints were
   therefore not exactly reproducible; they moved by at most 0.03 (e.g. `real`
   trap [0.00, 0.23] vs [0.00, 0.20] on two runs of identical episode data).
   All point estimates and episode outcomes reproduce bit-for-bit.
10. Total compute for this submission: ~9 min single core, of which the
    reported set is ~4.1 min (45 s at n=30, 140 s at n=100, 51 s gamma sweep,
    5 s landscape). The rest is superseded runs: an earlier 14-arm n=30 sweep
    and two n=100 sweeps that were started and killed when the bootstrap-seed
    fix of caveat 9 landed. Nothing was truncated for time; no seed count was
    cut, and the 12-minute budget was not reached.
11. Process disclosure: no git command of any kind was run for this
    submission. Only files under `experiments/E5_ablation/` were written.

## Reproduce

```
uv run python experiments/E5_ablation/scripts/run_e5.py --seeds 30
uv run python experiments/E5_ablation/scripts/run_e5.py --seeds 100 --tag _n100
uv run python experiments/E5_ablation/scripts/gamma_check.py --seeds 30
uv run python experiments/E5_ablation/scripts/phi_landscape.py
uv run python experiments/E5_ablation/scripts/summarize_e5.py          # Tables 1-3
uv run python experiments/E5_ablation/scripts/review_stats.py          # exact CIs, contrasts
uv run python experiments/E5_ablation/scripts/review_stats.py _n100
```

`run_e5.py` asserts at startup that the mirrored loop in `scripts/e5_lib.py`
still reproduces `fmcphi.planner.plan`, for `mode=real, gamma=1` and
`mode=off, gamma=0`, on 8 seeds. The check compares the returned action,
`b_eff`, `ess`, `mean_phi`, `min_phi`, `dead_walkers`, `sim_steps`, the full
N-vector of walker labels after the last clone step, **and the post-call state
of the shared bit generator** (identical only if both loops drew exactly the
same random numbers in the same order). It does not compare the per-walker Phi
vector tick by tick, only its per-tick mean and min. Nothing in `src/` was
modified.

---

## Review response

Blocking issues, in the reviewer's order.

1. **"The decisive arms do not ablate what the report says they ablate."**
   *Accepted in full; the reviewer's control reproduces.* Added
   `shuffled_keepmask` and `noise_keepmask` (ablate the graded factor in the
   virtual reward, keep `kill_dead` on the unpermuted Phi) and `mask` /
   `mask_matched` / `mask_true_matched` (`Phi = 1{viable}`, zero simulator
   steps). Our numbers match the reviewer's: `shuffled_keepmask` 0.63
   [0.47, 0.80], `noise_keepmask` 0.57 [0.40, 0.73], real 0.53 [0.37, 0.70],
   n=30, seeds 1000-1029. P1, P2 and P5 are re-verdicted as "literally
   supported, not diagnostic", the decision rule is re-verdicted as "satisfied
   as written and not sufficient", and the headline now says the effect is
   carried by the `kill_dead` viability mask and that the causal-cone entropy
   is not shown to add anything over a binary viability indicator.
2. **"A zero-cost, zero-information Phi beats the real Phi."** *Accepted in
   full.* The `1{viable}` arm is in Table 1 at N=32 (0.83 [0.70, 0.97], paired
   -0.30 [-0.50, -0.13]) and at both equal-nominal (N=416, 1.00, trap 0.00) and
   equal-measured (N=329, 0.97 [0.90, 1.00], trap 0.03) budget. The verdict now
   reads: the Phi layer is supported only insofar as it supplies a viability
   mask; the causal-cone estimator is not justified by this data; if E2/E3
   depend on `phi_cone` that dependency is unsupported (Finding 1).
3. **"The equal-budget comparison is nominal, not measured."** *Accepted, and
   fixed by measuring rather than by rewording.* Added `StepCounter`, a
   delegating env wrapper that counts real `env.step` calls; every arm now
   reports nominal and measured cost per decision and their ratio (Table 4).
   Our measured ratios match the reviewer's instrumentation (real 0.791 vs
   0.793, `cheap_matched` 0.909 vs 0.910, `off` arms 1.000). Added
   `cheap_true_matched` (N=60, measured 3243 vs real's 3291) and
   `off_true_matched` (N=329); the direction survives, as the reviewer
   predicted, and "at equal simulator budget" now appears only where the budget
   is equal as measured.
4. **"'The cheap proxy beats real Phi' is asserted where the primary statistic
   does not support it."** *Accepted.* P4 now reads "supported: it matches",
   with the paired diff quoted as -0.23 [-0.47, +0.00], "not distinguishable
   from zero at the pre-registered n". n=30 remains primary; n=100 stays
   secondary and is labelled as decided after seeing n=30 (Table 7). The word
   "beats" is now used only for the mask arms, whose n=30 paired CIs exclude
   zero.
5. **"gamma = 1.0 is not the gamma the BRIEF pre-registered."** *Accepted; ran
   the sweep rather than only rewording.* `gamma_check.py` runs real, cheap and
   mask at gamma in {0.25, 0.5, 1.0}, n=30 (Table 5). Real's best is gamma=0.5
   (0.70 [0.53, 0.87]), which reproduces E1's optimum and closes the gap to
   `cheap`@0.25 and `mask`@0.25 to -0.13 [-0.37, +0.10], indistinguishable,
   while `mask_matched`@0.25 still wins at -0.30 [-0.47, -0.13]. The sentence
   "`phi_cone` should be justified there, not assumed" is retained in Finding 1
   but now rests on the gamma-matched comparison, not on the gamma=1 one.

Overclaims, each removed or substantiated:

- Headline "state-correspondence, not perturbation of the virtual reward":
  **withdrawn and replaced** with the mask decomposition.
- Finding 5 (survival weighting): kept, but the conclusion it implies is now
  drawn explicitly, that the `env.viable` gate and not the cone is what is
  working (Findings 1, 2, 5).
- P4 "at equal budget it beats real": **downgraded** to "matches" with the
  touching-zero CI quoted.
- "At equal simulator budget" / "4140 vs 4160": **replaced** by measured counts
  and by two new true-matched arms.
- `cheap_matched` "with zero fuel deaths": now stated per n. 0.00 [0.00, 0.00]
  at n=30, 0.02 [0.00, 0.05] at n=100 (Tables 1 and 7).
- P5 "mean 1.92 vs real 1.91": **removed.** Table 3 now gives 1.878
  [1.796, 1.964] for real at n=30 and annotates every mean Phi with the number
  of decisions it is averaged over (real 703, noise 45), with an explicit note
  that the two are different state distributions and not a matched null.
- "asserts ... reproduce ... exactly" / "bit-identical": `parity_check` was
  **rewritten** to compare all returned diagnostics, the full label vector and
  the post-call RNG state over 8 seeds, and the report states exactly what it
  does and does not cover.
- Caveat 7 "uniform rescale": **corrected** to "`relativize` returns exactly
  ones when std == 0, so the Phi factor is the identity".
- Table 4 "range 0.00 to 3.27, sd 0.736 over the 38 non-trap cells": **split**
  into the two populations (all 48 cells [0.00, 3.27]; the 38 alive cells
  [1.16, 3.27], mean 2.52, sd 0.736).
- Table 4 probed at fuel=20 only: **fixed.** The landscape is now probed at
  fuel 20, 8, 4, 3 and 2, and the low-fuel collapse (83% of living-cell
  estimates exactly 0 at fuel=2) is reported as the likely mechanism behind
  real Phi's fuel deaths (Table 6, Finding 6).

Nothing in the review was declined. One item is only partially closed:
`parity_check` still does not compare the per-walker Phi vector tick by tick,
only its per-tick mean and min plus the RNG stream; the report says so instead
of claiming bit-identity.
