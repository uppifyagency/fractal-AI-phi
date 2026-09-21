# SYNTHESIS — what the ladder decided

2026-09-21, after E1 to E5 (each with an adversarial reviewer; four of five were
sent back and re-run) and the E6 instrument build.

## The verdict in one paragraph

The hypothesis of this repository was that FMC is missing a causal-cone freedom
factor, $\Phi$, and that adding it as a third exponent would produce
self-preservation as a measurable term. **The effect is real and the mechanism
is not the one proposed.** Everything the layer achieves on TrapGrid is achieved
by a binary viability mask that costs zero simulator calls, and the graded cone
entropy contributes nothing that survives ablation. The expensive half of the
idea is the useless half.

## The decisive table (E5, TrapGrid, 30 paired seeds)

| arm | what it is | measured sim/decision | goal rate |
|---|---|---|---|
| `off_true_matched` | canonical FMC, equal budget | 3290 | 0.00 [0.00, 0.00] |
| **`real`** | $\Phi$ as specified: cone, survival-weighted | 3291 | 0.53 [0.37, 0.70] |
| `shuffled_keepmask` | the same $\Phi$ vector, **permuted across walkers** | 3314 | 0.63 [0.47, 0.80] |
| `noise_keepmask` | $\Phi$ replaced by uniform noise | 3209 | 0.57 [0.40, 0.73] |
| `cheap_true_matched` | depth-1 action count instead of the cone | 3243 | 0.77 [0.60, 0.90] |
| **`mask_true_matched`** | $\Phi = \mathbb{1}\{\text{viable}\}$, **0 extra steps** | 3290 | **0.97 [0.90, 1.00]** |
| `mask_matched` | the same mask, equal nominal budget | 4160 | **1.00 [1.00, 1.00]** |

Read the third and fourth rows first. Permuting $\Phi$ across walkers destroys
every correspondence between a walker and its own future, and the goal rate does
not move: paired diff against `real` is $-0.10$ [$-0.30$, $+0.10$]. Replacing it
with noise: $-0.03$ [$-0.30$, $+0.23$]. **The graded factor is carrying no
information about the state it is attached to.**

Then read the last two. A binary indicator of "is this walker standing on a dead
state", which requires no rollout, no perplexity and no survival weighting,
reaches 30/30 at the same budget the full layer spends to reach 16/30.

The original pre-registered decision rule (the layer is supported only if
shuffled and noise both lose to real $\Phi$) was **satisfied and was
insufficient**: both of those arms also removed the viability mask, so the rule
could not separate "the cone works" from "zeroing dead walkers works". E5's
reviewer caught this and the two `keepmask` arms were added to separate them.
That correction is the single most valuable thing the ladder produced.

## Rung by rung

| rung | pre-registered claim | outcome |
|---|---|---|
| E1 | $\Phi$ trades trap deaths for goals at equal budget | replicates, but factorially the **mask** carries the goal-rate gain and the exponent only lowers trap rate, paid in fuel deaths. A free viability mask at equal budget scores 30/30, beating the full layer by $+0.467$ [$+0.300$, $+0.633$], $p=0.0001$ |
| E2 | FoT+$\Phi$ beats FoT at equal token budget | **falsified**, $-7.3$ pp [$-10.0$, $-4.6$], $n=960$/arm, mock backend. P2 falsified. P3 supported by construction. A post-hoc amortised variant ($\texttt{phi\_every}=22$) wins $+4.4$ pp and is a hypothesis, not a rescue |
| E3 | $\Phi$ is lower before an irreversible action | **sign not robust**: it flips across two choices the spec never fixed (the state-identity function, and the unit of analysis). P2 (trap avoidance) supported: $0.74\to0.13$ |
| E4 | a goal-less agent survives on $\Phi$ alone | supported on both maps at equal budget, but about a quarter of the first estimate after a **measurement defect of mine** was fixed: death was not absorbing, so $\gamma=0$ arms rolled through their own death. P3 supported: goal rate 0/50, exactly random, so $\Phi$ smuggles no reward information |
| E5 | ablations isolate the cone | **decisive, above** |
| E6 | the E3 force-push surprise on a real agent | not run. Power analysis: 169 episodes/arm needed, 20 affordable. Redesign needs 54 decisions/arm. Then the hand-validation showed $\Phi_{\text{cone}}$ is **inert** on the repo encoding (identical for every candidate, so constant after relativize), which would guarantee a null on the one arm that tests the claim |

## The three results worth keeping

1. **The binary viability mask.** Free, and it dominates everything measured
   here. If FMC has a missing term, this is it, and it is one line:
   zero the virtual reward of a walker standing on a dead state. It is not a
   causal entropic force; it is a hard constraint that canonical FMC omits.
2. **E3's unpredicted trade.** The layer cut the visible trap 21x and raised
   force-push, the irreversibility it cannot see, by 2.8x ($+0.085$
   [$+0.051$, $+0.120$], $n=100$). E6's hand-validation then showed the *composite*
   $\Phi$, which does include an irreversibility classifier, refuses the
   force-push. Read together: an option-preserving planner is only as safe as
   the irreversibilities its $\Phi$ can see, and it actively seeks out the ones
   it cannot. That is a safety property, and it survives the collapse of the
   main hypothesis.
3. **E4's P3.** A $\Phi$-only agent with $\alpha=0$ reaches the goal exactly as
   often as random: 0/50 on both maps. Whatever the layer does, it is not
   leaking reward information through the geometry. The one claim that was
   supposed to be the publishable one is clean, and the term it rests on turned
   out to be unnecessary.

## What this says about the original motivation

The motivating picture was the motorway: you stay alive by continuously
correcting your trajectory so that tomorrow stays reachable, and the ratio
$\alpha/\gamma$ is how much of that reachability you spend to arrive sooner.
The ladder does not refute the picture. It refutes the **estimator**. Counting
reachable futures by Monte-Carlo fan-out turned out to be an expensive way to
approximate a much cruder fact that already decides the outcome: whether you are
on the road at all.

Whether a graded freedom term is worth anything in an environment where the
crude fact is not already decisive remains open, and E6 is where it would be
tested. It needs a $\Phi$ with genuine variance across candidate states, which
the current repository encoding does not provide.
