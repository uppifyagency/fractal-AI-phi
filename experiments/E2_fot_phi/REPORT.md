# E2: Fractal-of-Thought with Phi

Mock backend only. No paid API call was made, no model was downloaded, no key
was set. The paid arm is specified down to the exact command in section 7 and
has never been executed.

Total run wall clock: 195 s (131 s for R1-R5, 64 s for R7). 26/26 unit tests
green (18 mock + 8 paid-path), plus the repo's own 16/16 still green.

---

## 1. Verdicts

**P1. On problems needing multiple steps, FoT+Phi beats FoT at equal token
budget.**
**FALSIFIED**, and not narrowly. The pre-registered arm (gamma = 1, Phi
recomputed every tick) scores **6.6% [5.0, 8.2]** against the equal-budget
gamma = 0 baseline's **13.9% [11.7, 16.0]**, a difference of
**-7.3 pp [-10.0, -4.6]**, p < 0.001, n = 960 episodes per arm across 80
problems and 12 seeds. Spending the Phi surcharge on cone estimates instead of
on rollout depth loses roughly half the baseline's accuracy.

A post-hoc variant does win: amortising Phi (`phi_every = 22`, which buys 22
rollout cycles instead of 4 at the same price) scores **18.2% [15.8, 20.7]**,
**+4.4 pp [+1.0, +7.6]**, p = 0.010. It was designed after seeing P1 fail, it
replicates on a held-out problem block, and it survives a budget control
(section 5). It is a hypothesis for E5, not a rescue of P1.

**P2. The gain concentrates on problems where the baseline commits early to a
wrong branch.**
**FALSIFIED**. For the best Phi arm the gain is identical on both splits:
+4.3 pp on the early-commit split, +4.4 pp on the other, interaction
**-0.1 pp [-7.2, +6.8]**, p = 0.965, n = 576 and 384 episodes. The pre-registered
arm has no gain on either split to concentrate. Its interaction is significant
(+9.5 pp [+3.9, +15.3], p = 0.002) but in the sense that Phi *loses less* where
the baseline commits early (-3.5 pp) than where it does not (-13.0 pp), which is
not the prediction.

**P3. On single-step problems Phi is flat.**
**SUPPORTED**, exactly rather than approximately. Across 2560 sampled decisions
on the single-step tier the within-decision spread of Phi over the three
continuations a decision ranges over is **0.0000**, zero in **100.0%** of
decisions, and `relativize(Phi)` is exactly the all-ones vector, so
`Phi_hat ** gamma` is the identity factor for any gamma. Behaviourally, turning
gamma on at equal N and M moves accuracy by **-0.4 pp [-6.0, +5.0]**, p = 0.856,
n = 480 per arm. On the multi-step tier the same measurement gives spread 0.5032
and a -3.2 pp [-5.6, -0.7] effect, so the single tier is a null by construction
and the multi tier is not.

---

## 2. What was built

`ReasoningEnv` (`scripts/mock_llm.py`) is an `fmcphi` Environment whose states
are partial chains of thought. It satisfies the two conditions `docs/SPEC.md`
section 5 requires of any test bed for this layer: a finite context budget and
irreversible commitment to a wrong line of reasoning. The harness
(`scripts/fot_phi.py`) calls the shared, tested `fmcphi.planner.plan` once per
committed reasoning step, so the gamma = 0 arm is canonical FMC bit for bit
(`test_gamma_zero_is_canonical_fmc` asserts `plan(gamma=0, seed=k).action ==
fmcphi.core.plan(..., seed=k)`).

| brief requirement | where |
|---|---|
| R from a real judge, not a length proxy | `ReasoningEnv.reward`: a process verifier scoring resolved sub-goals, checkable slips (`flagged`) and self-contradiction. Never reads ground truth; `test_judge_and_geometry_are_ground_truth_free` proves it by changing the answer key and asserting every planner-facing quantity is byte-identical |
| Phi over reasoning states as distinct viable continuations | `fmcphi.phi.phi_composite` unchanged, with `env.key(s)` = the answer-key the chain currently points to, so the perplexity term counts distinct reachable answers |
| `phi_slack` maps to remaining context tokens | `budget_attr="tokens"`, `budget_max=tokens0` |
| mock backend, deterministic, free, CI-able | `mock_llm.py`, all hashes seeded; `test_episode_is_deterministic` |
| real-model path is a flag, not the default | `--backend anthropic`, gated behind `--confirm-paid` plus a resolvable credential plus the $25 cap check; `--dry-run-paid` renders prompts and prices the run offline |

### The load-bearing modelling assumption, stated plainly

The mock has to encode *some* relationship between cone shape and correctness or
there is nothing for Phi to find. It encodes it through two **independent**
draws per wrong branch:

- `detectable` (p = 0.6): the verifier catches the slip and R penalises it. This
  is the only signal the gamma = 0 baseline runs on, and it is why the baseline
  is well above chance.
- `narrow` (p = `phi_signal`): the branch's cone collapses. This is the only
  signal Phi adds.

Because the draws are independent (`test_narrow_and_detectable_are_independent_draws`),
Phi's incremental value can come only from branches that are narrow *and*
undetectable. If the two were one draw the experiment would be circular. The
`phi_signal` sweep in section 6 is the control that keeps it honest.

A second deliberate choice: a wrong branch is **not** shorter than the correct
one. Both reach a forced answer at the same depth, so the verifier sees
identical progress on both. If wrong branches were shorter, R alone would solve
the task and the experiment would be a length proxy again, which is the exact
defect the brief says to remove.

---

## 3. Budget matching

Task constraint 6 is enforced arithmetically, not by eyeball. `plan` charges

    sim_steps = N*M + [gamma != 0] * N * ceil(M / phi_every) * m * h

so with N = 16, M = 4, m = 3, h = 2 the reference budget is
16 * 4 * (1 + 6) = **448 sim_steps per decision = 80 640 tokens per decision** at
180 tokens per continuation. Every arm below except `fot_nm` is charged exactly
448; `assert_equal_budget` and `test_arms_are_budget_matched_and_the_planner_agrees`
check it against the planner's own counter. The Phi ladder is the set of
`(M, phi_every)` pairs that solve the budget equation exactly, which turns the
amortisation knob of `docs/SPEC.md` section 4 into a one-dimensional sweep of how
to spend a fixed budget between rollout depth and cone-estimate freshness.

| arm | gamma | M | phi_every | Phi refreshes | sim_steps/decision |
|---|---|---|---|---|---|
| `fot_nm` | 0 | 4 | - | - | 64 |
| `fot_budget` | 0 | 28 | - | - | 448 |
| `fot_phi` (pre-registered) | 1 | 4 | 1 | 4 | 448 |
| `fot_phi_M10k4` | 1 | 10 | 4 | 3 | 448 |
| `fot_phi_M16k8` | 1 | 16 | 8 | 2 | 448 |
| `fot_phi_M22k22` | 1 | 22 | 22 | 1 | 448 |

---

## 4. Main result

80 problems (R1: pid 0-39, R5: pid 5000-5039, a disjoint held-out block), 12
seeds each, `phi_signal = 0.75`, `p_trap = 0.6`, `p_detect = 0.6`,
`n_stages = 3`, context budget 8.

### R1 + R5 pooled

| arm | n | accuracy [CI95] | vs `fot_budget` [CI95] | p | sim_steps/dec | tokens/episode [CI95] | steps [CI95] |
|---|---|---|---|---|---|---|---|
| `fot_nm` | 960 | 9.8% [8.0, 11.7] | -4.1 pp [-7.0, -1.2] | 0.007 | 64 | 35 064 [34 740, 35 388] | 3.04 [3.02, 3.07] |
| `fot_budget` | 960 | 13.9% [11.7, 16.0] | - | - | 448 | 266 028 [262 416, 269 640] | 3.30 [3.26, 3.34] |
| **`fot_phi`** | 960 | **6.6% [5.0, 8.2]** | **-7.3 pp [-10.0, -4.6]** | **0.000** | 448 | 221 760 [218 568, 225 036] | 2.75 [2.71, 2.79] |
| `fot_phi_M10k4` | 960 | 15.8% [13.5, 18.2] | +2.0 pp [-1.1, +5.1] | 0.233 | 448 | 246 624 [244 944, 248 472] | 3.06 [3.04, 3.08] |
| `fot_phi_M16k8` | 960 | 16.7% [14.4, 19.1] | +2.8 pp [-0.4, +6.0] | 0.089 | 448 | 256 536 [253 680, 259 476] | 3.18 [3.15, 3.22] |
| `fot_phi_M22k22` | 960 | 18.2% [15.8, 20.7] | +4.4 pp [+1.0, +7.6] | 0.010 | 448 | 296 100 [291 312, 300 972] | 3.67 [3.61, 3.73] |

The two blocks separately (n = 480 per arm per block): `fot_phi` is -7.9 pp
[-11.9, -4.0] on R1 and -6.7 pp [-10.4, -3.1] on the held-out R5, so the
falsification of P1 replicates. `fot_phi_M22k22` is +4.6 pp [-0.2, +9.4] on R1
and +4.2 pp [-0.2, +8.5] on R5, each individually straddling zero and only
clearing it pooled.

### Why the pre-registered arm loses

`fot_phi` answers fastest of all arms (2.75 steps, 222k tokens per episode) and
is least accurate. Phi at a node on the correct line of reasoning is 3.64, and
at a committed wrong line 1.19 (section 6), so the factor does discriminate; but
with only M = 4 rollout cycles the swarm's lookahead is too shallow to see which
branch the goal reward will eventually favour, and a gamma that rewards keeping
options open makes it hesitate. This is the same alpha/gamma standoff
`docs/SPEC.md` section 6 records for TrapGrid, where the goal corner is
option-poor. Here the "corner" is a finished chain: an answered state scores
Phi = 0.83, the lowest of any node type, so gamma actively penalises finishing.

### P2 split (R1 + R5 pooled)

Split defined from `fot_budget` records only: a problem is in
`baseline_commits_early` iff the chain is off the correct line after the first
committed step on a strict majority of seeds. 48 of 80 problems qualify.

| split | arm | n | accuracy [CI95] | vs `fot_budget` [CI95] | p |
|---|---|---|---|---|---|
| commits early | `fot_budget` | 576 | 8.7% [6.4, 11.1] | - | - |
| commits early | `fot_phi` | 576 | 5.2% [3.5, 7.1] | -3.5 pp [-6.4, -0.5] | 0.017 |
| commits early | `fot_phi_M16k8` | 576 | 12.2% [9.5, 14.9] | +3.5 pp [+0.0, +6.9] | 0.058 |
| commits early | `fot_phi_M22k22` | 576 | 13.0% [10.4, 15.8] | +4.3 pp [+0.7, +8.0] | 0.021 |
| stays on line | `fot_budget` | 384 | 21.6% [17.4, 25.8] | - | - |
| stays on line | `fot_phi` | 384 | 8.6% [6.0, 11.5] | -13.0 pp [-18.0, -8.1] | 0.000 |
| stays on line | `fot_phi_M16k8` | 384 | 23.4% [19.3, 27.6] | +1.8 pp [-4.2, +7.6] | 0.575 |
| stays on line | `fot_phi_M22k22` | 384 | 26.0% [21.9, 30.5] | +4.4 pp [-1.6, +10.4] | 0.167 |

Interaction (the quantity P2 is actually about, since two separate intervals
cannot settle a difference between them):

| arm | (gain on early split) - (gain on other split) [CI95] | p |
|---|---|---|
| `fot_phi` | +9.5 pp [+3.9, +15.3] | 0.002 |
| `fot_phi_M10k4` | -0.6 pp [-7.4, +6.2] | 0.858 |
| `fot_phi_M16k8` | +1.6 pp [-5.2, +8.5] | 0.647 |
| `fot_phi_M22k22` | -0.1 pp [-7.2, +6.8] | 0.965 |

---

## 5. Is the amortised gain just a bigger budget? No.

Per-decision budget is matched exactly, but the Phi arm takes more real
reasoning steps before answering (3.67 vs 3.30), so it spends about 11% more
*total* tokens per episode. R7 buys the gamma = 0 baseline 25% and 50% more
rollout depth than the reference, which is strictly more than that gap, on the
same 80 problems and 12 seeds.

| arm | n | accuracy [CI95] | vs `fot_budget` [CI95] | p | sim_steps/dec | tokens/episode |
|---|---|---|---|---|---|---|
| `fot_budget` (M=28) | 960 | 13.9% [11.7, 16.0] | - | - | 448 | 266 028 |
| `fot_budget_M35` | 960 | 14.4% [12.2, 16.6] | +0.5 pp [-2.6, +3.6] | 0.780 | 560 | 331 065 |
| `fot_budget_M42` | 960 | 13.4% [11.4, 15.6] | -0.4 pp [-3.5, +2.6] | 0.764 | 672 | 395 640 |
| `fot_phi_M22k22` | 960 | 18.2% [15.8, 20.7] | +4.4 pp [+1.0, +7.6] | 0.010 | 448 | 296 100 |

The gamma = 0 baseline is saturated: 50% more depth buys it nothing
(-0.4 pp [-3.5, +2.6]). The Phi arm reaches 18.2% on 25% *fewer* total tokens
than `fot_budget_M42`. Whatever the amortised arm is doing, it is not buying
accuracy with budget.

---

## 6. Mechanism and the falsification control

### Phi by node type (`phi_composite`, m = 6, h = 2)

| node type | n | mean Phi | std |
|---|---|---|---|
| on the correct line (`b0`) | 1651 | 3.642 | 1.185 |
| committed wrong line, cone narrowed (`narrow`) | 860 | 1.187 | 0.462 |
| wrong line that still looks option-rich (`decoy`) | 238 | 3.244 | 1.202 |
| finished chain (`answered`) | 451 | 0.832 | 0.068 |

Phi separates a committed wrong line from an open correct one (1.19 vs 3.64) and
is correctly fooled by decoys (3.24), which is the designed imperfection.

### Phi spread within one decision

| tier | decisions sampled | mean within-decision std(Phi) | std(relativize(Phi)) | decisions with exactly zero spread |
|---|---|---|---|---|
| multi | 2549 | 0.5032 | 0.3881 | 38.5% |
| single | 2560 | **0.0000** | **0.0000** | **100.0%** |

### R3: accuracy gain against how much information Phi carries

`phi_signal` is the fraction of wrong branches whose cone actually narrows. At 0
the Phi term is well defined and non-constant but carries no information about
correctness. 20 problems, 8 seeds, n = 160 per cell.

| phi_signal | `fot_phi` vs base | `fot_phi_M16k8` vs base | `fot_phi_M22k22` vs base |
|---|---|---|---|
| 0.00 | -9.4 pp [-15.0, -3.8] | -1.2 pp [-8.1, +5.6] | -4.4 pp [-10.6, +1.9] |
| 0.25 | -7.5 pp [-13.1, -1.9] | +0.6 pp [-6.2, +7.5] | +0.6 pp [-6.3, +7.5] |
| 0.50 | -6.9 pp [-13.1, -0.6] | +2.5 pp [-5.0, +10.0] | +1.2 pp [-6.2, +8.8] |
| 0.75 | -11.2 pp [-18.1, -4.4] | -2.5 pp [-10.6, +5.6] | +5.0 pp [-3.8, +13.8] |
| 1.00 | -13.8 pp [-21.2, -6.2] | -1.2 pp [-10.0, +7.5] | +1.3 pp [-7.5, +10.6] |

The one thing this establishes cleanly is the negative control: at
`phi_signal = 0` no Phi arm beats the baseline, and the best arm is -4.4 pp. So
the amortised arm's pooled +4.4 pp is not a budget artifact *and* not a
structural artifact of running fewer cycles. What it does **not** establish is a
clean dose-response: the point estimates rise from 0 to 0.75 and then fall back
at 1.00, and every cell after 0 straddles zero at n = 160. Reporting that as a
monotone trend would be overreading it.

---

## 7. The paid arm

Never run. `scripts/llm_backend.py` is the specification plus an offline
dry-run; its network path has never executed and is marked unvalidated in the
module docstring. What is validated is the offline path: the planner runs
end to end on `LLMReasoningEnv`, `phi_composite` computes on it, node calls are
cached by path, the cost estimator is monotone in depth, the refusal gate fires
without `--confirm-paid`, and the $25 cap gate fires when the estimate exceeds
it (8 tests in `scripts/test_e2_llm_dryrun.py`).

The affordability trick is node caching. `phi_cone` costs m*h generations per
walker per Phi tick, which is unaffordable ported naively, but the continuation
tree is a property of the node, not of the walker standing on it. One call per
distinct `(question, path)` returns the three ranked continuations, the
verifier's verdict, the progress estimate and the model's current best guess, so
the bill is bounded by distinct nodes within `--max-depth`, not by N*M*m*h.
Seeds and arms share the cache and do not multiply it.

**Step 1, free, run this first:**

```bash
cd /Users/vladvrinceanu/fractal-AI-phi && \
  uv run python experiments/E2_fot_phi/scripts/fot_phi.py \
    --backend anthropic --dry-run-paid \
    --problems 10 --seeds 3 --max-depth 3 --model claude-opus-5
```

It prints the cost upper bound and the exact prompt that would be sent, and
states `NO NETWORK CALL WAS MADE.` At those settings the bound is **400 API
calls, $4.00** on `claude-opus-5` at $5 / $25 per MTok, assuming 700 input and
260 output tokens per call.

**Step 2, one problem, paid, to check the bound empirically:**

```bash
cd /Users/vladvrinceanu/fractal-AI-phi && export ANTHROPIC_API_KEY=... && \
  uv run --with anthropic python experiments/E2_fot_phi/scripts/fot_phi.py \
    --backend anthropic --confirm-paid \
    --problems 1 --seeds 1 --max-depth 3 --model claude-opus-5 \
    --out P1_paid_pilot
```

**Step 3, the set:** the same command with `--problems 10 --seeds 3`.

Three things a human must fix before step 3, all flagged in the module:

1. `run_paid` currently returns 3 with a refusal message on the last line rather
   than executing the grid. That line exists so the untested network path cannot
   run by accident. Delete it after reading the module.
2. `LLMReasoningEnv.observe` uses a hash of the path as a stand-in for the
   chain embedding. The prior art used all-MiniLM L2 distance between chains of
   thought, and the beta term is meaningless without a real embedding. This is
   the one substitution that changes the numbers rather than the bill.
3. The problem set is not wired in. Point it at a multi-hop set with verifiable
   answers; `/tmp/fmc-study/fmc-core/bench/llm/problems.py` `HARD` is 12
   suitable items.

`claude-opus-5` is the default. `claude-haiku-4-5` at $1 / $5 per MTok is a
fifth of the cost if the $25 cap binds before the problem count does; that is a
quality tradeoff and therefore a human's call, not the harness's.

---

## 8. Caveats, worst first

1. **The mock is not an LLM, and P1 and P2 were tested only against it.** The
   mock encodes a geometry in which committing to a wrong line of reasoning
   narrows the cone. Whether real chain-of-thought has that geometry is exactly
   what the paid arm exists to find out, and it is untested. Every number here
   is a statement about the *mechanism* (does a multiplicative Phi factor in the
   virtual reward convert a cone-narrowing signal into accuracy, at matched
   budget, with a verifier that cannot separate the branches) and not about
   language models.
2. **The amortised ladder is post-hoc.** `fot_phi_M22k22` was added after seeing
   the pre-registered arm fail. It replicates on a held-out block and survives
   the budget control and the `phi_signal = 0` control, but it has not been
   pre-registered anywhere and its per-block intervals both straddle zero. It is
   an E5 hypothesis. Treating it as E2's result would be exactly the
   goalpost-moving the task forbids.
3. **`relativize` is scale free, so "Phi is flat" has to mean exactly flat.** In
   the single tier the within-decision spread is 0 to the bit and
   `relativize(Phi)` is exactly ones. Had it been 1e-10 instead of 0,
   `relativize` would have z-scored it back up to full strength and gamma would
   have acted on numerical noise. An earlier version of the single tier did
   exactly this (context cost of 1 against a budget of 1e9), and it was fixed by
   giving the single tier no context cost at all. Anyone porting this layer to a
   new environment should measure `std(relativize(Phi))`, not `std(Phi)`.
4. **Phi is not flat over a mixed cloud even in the single tier.** Finishing is
   absorbing, so a finished chain scores Phi = 1 and an open one more. That
   residual cannot move a decision, because a decision ranges over siblings, but
   "Phi is flat" read as "Phi is constant everywhere" is false even in the null
   tier.
5. **Accuracies are low in absolute terms** (9% to 20%). The mock was tuned so
   that the verifier leaves real residual error for Phi to attack; the wide
   headroom keeps the arms separable but the CIs are correspondingly wide at
   n = 960.
6. **`n_stages`, `p_detect`, `p_trap` and `tokens0` were chosen by hand** to put
   the baseline above chance and below ceiling, before the arms were compared.
   No sweep over them was run, and they are not pre-registered.
7. **The gamma-only contrast in section 1 is not budget matched.** `fot_nm` and
   `fot_phi` share N and M, and the Phi arm additionally pays the m*h surcharge.
   It is reported only to isolate what gamma does, never as a verdict on P1.
8. **No gamma sweep.** Only gamma = 0 and gamma = 1 were run. The standoff
   described in section 4 suggests gamma < 1 or a gamma schedule is the obvious
   next knob, which is E1's open question, not this one's.

---

## 9. Reproducing

```bash
cd /Users/vladvrinceanu/fractal-AI-phi
uv run pytest experiments/E2_fot_phi/scripts/ -q                   # 26 tests
uv run python experiments/E2_fot_phi/scripts/run_all.py            # R1-R5, 131 s
uv run python experiments/E2_fot_phi/scripts/run_budget_control.py # R7, 64 s
uv run python experiments/E2_fot_phi/scripts/analyze.py            # tables + R6
```

| file | what |
|---|---|
| `scripts/mock_llm.py` | the mock backend and `ReasoningEnv` |
| `scripts/fot_phi.py` | arms, budget ladder, episode loop, CLI, diagnostics |
| `scripts/llm_backend.py` | the paid arm, specified and dry-runnable, never run |
| `scripts/stats.py` | bootstrap CI95, difference, difference-in-differences |
| `scripts/run_all.py` | R1 to R5 |
| `scripts/run_budget_control.py` | R7 |
| `scripts/analyze.py` | every table above, writes `R6_summary_stats.json` |
| `scripts/test_e2_mock.py` | 18 tests: protocol, parity, budget, determinism, flatness |
| `scripts/test_e2_llm_dryrun.py` | 8 tests: offline paid path and its gates |
| `results/R1_multi_main.json` | main grid, 40 problems x 12 seeds x 6 arms |
| `results/R2_single_control.json` | single-step tier, P3 |
| `results/R3_phi_signal_sweep.json` | falsification control |
| `results/R4_diagnostics.json` | Phi by node type, Phi spread per decision |
| `results/R5_heldout.json` | held-out problem block, pid 5000-5039 |
| `results/R6_summary_stats.json` | every number in this report, machine readable |
| `results/R7_budget_control.json` | the baseline handed 25% and 50% more depth |
