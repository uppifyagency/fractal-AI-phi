# SPEC — the Phi layer

Status: v0.1, 2026-09-21. Foundation implemented and tested (16/16 green).

## 1. The gap this closes

Canonical Fractal Monte Carlo weights each walker by

$$\mathrm{VR}^{(i)} = \big(\widehat{R}^{(i)}\big)^{\alpha}\cdot\big(\widehat{D}^{(i)}\big)^{\beta}$$

where $\widehat{R}$ is the relativized reward and $\widehat{D}$ the relativized
distance to a random partner walker.

$D$ measures dispersion **between walkers**: diversity of hypotheses inside the
swarm. The quantity Wissner-Gross & Freer (2013) call the causal entropic force,
$F = T_c\nabla_X S_c$, is a different object: the entropy of futures reachable
**from one state**. FMC has never contained the second one.

That the two are not interchangeable is visible in the reference project's own
rocket sweep, reported there as a counter-intuitive caveat under Theorem 3:

| $\beta$ | $b_{\text{eff}}$ |
|---|---|
| 0.0 | $5.45\pm0.82$ |
| 1.0 | $3.52\pm0.94$ |
| 5.0 | $1.89\pm0.69$ |

Raising $\beta$ *lowers* the effective branching factor. A term that preserved
options would raise it. $\beta$ rewards the isolated walker, so past the anti
collapse threshold it acts as one more selector.

## 2. Definition

$$\boxed{\;\Phi(x) \;=\; p_{\text{surv}}(x)\cdot\exp\Big(H\big(\{\,\text{distinct viable states reachable from }x\text{ in }h\text{ steps}\,\}\big)\Big)\;}$$

estimated by fanning out $m$ random continuations of depth $h$ from $x$,
discarding the ones that hit an absorbing state, and taking the perplexity of
the surviving endpoint distribution scaled by the surviving fraction.

This is Definition 6 of the reference canon (effective branching factor)
evaluated **forward from a single state** instead of backward over surviving
swarm labels.

The survival weighting is not cosmetic. A path that dies contributes no further
branching, so in causal path entropy it is mass removed, not one more endpoint.
Without the weighting, a state one step from a cliff scores as free as a state
in open ground. `weight_by_survival=False` keeps the unweighted form available
as an ablation arm.

## 3. The three-factor virtual reward

$$\boxed{\;\mathrm{VR}^{(i)} = \big(\widehat{R}^{(i)}\big)^{\alpha}\cdot\big(\widehat{D}^{(i)}\big)^{\beta}\cdot\big(\widehat{\Phi}^{(i)}\big)^{\gamma}\;}$$

Multiplicative, not additive, following the reference paper section 2.2.2 and
Sergio's explicit rejection of Pareto fronts. Under an additive composition a
large enough goal reward can buy a state with no futures.

Roles:

| Knob | Pulls toward | Horizon |
|---|---|---|
| $\alpha$ | the goal | long |
| $\gamma$ | states with more reachable futures | short |
| $\beta$ | swarm dispersion | per tick |

$\alpha/\gamma$ is the risk dial: how much option-space the agent will spend to
arrive sooner. $\alpha=0,\gamma=1$ is an agent with no goal at all that should
still behave competently. That is E4, and it is the claim worth publishing.

Walkers with $\Phi = 0$ exactly are zeroed by `kill_dead` rather than merely
down-weighted, keeping the hard-constraint semantics of a multiplicative reward
while leaving "dead" and "cornered" distinguishable in the diagnostics.

## 4. Implementation contract

```
fmcphi.core                vendored verbatim from fmc-core (MIT). Do not edit.
fmcphi.phi                 phi_cone, phi_viable_actions, phi_slack,
                           phi_composite, virtual_reward_phi, kill_dead
fmcphi.planner.plan        one decision -> PlanResult (action, b_eff, ess,
                           mean_phi, min_phi, dead_walkers, sim_steps)
fmcphi.planner.run_episode outer loop -> EpisodeResult (outcome, steps,
                           total_reward, sim_steps, mean_b_eff, mean_phi)
fmcphi.envs.base           Environment protocol + viable(s) + key(s)
fmcphi.envs.trapgrid       TrapGrid, the E1/E4/E5 environment
```

Two invariants, both under test:

1. `plan(..., gamma=0.0, seed=k).action == fmcphi.core.plan(..., seed=k)`.
   Any measured delta must come from $\Phi$ and nothing else.
2. `sim_steps` counts every simulator call including the $\Phi$ rollouts, so
   arms are comparable at equal budget, not at equal $(N, M)$.

Cost: $\Phi$ multiplies the per-tick cost by $m\cdot h$. `phi_every=k` recomputes
it every $k$ ticks and reuses a stale value in between. Measured amortisation is
E1's secondary deliverable.

## 5. The null-test trap

The reference project's Round-1 falsified "FMC beats greedy" on synthetic tasks,
then found in post-mortem that the landscape violated 3 of the 5 literature
conditions under which ensembles beat greedy. It was a null test by construction.

$\Phi$ has the same exposure. In an environment with no irreversibility and no
finite budget, the cone never narrows, $\Phi$ is constant, and the term vanishes
after relativize. **An environment without real death cannot test this layer.**
`TrapGrid` has both: absorbing traps and a fuel budget, with a short lethal route
and a long safe one, so neither pure goal-seeking nor pure caution wins alone.

## 6. Pilot result already in hand

`TrapGrid(fuel=24)`, $N=32$, $M=10$, $\alpha=\beta=1$, 10 seeds:

| arm | goal | trap | fuel | mean steps |
|---|---|---|---|---|
| $\gamma=0$ | 0 | **10** | 0 | 1.4 |
| $\gamma=1$ | **6** | **0** | 4 | 24.8 |

Pure goal-seeking walks off the cliff on the first move. With $\Phi$ on, the
swarm climbs to the safe corridor, crosses, and descends at the far end. Trap
deaths go to zero and are traded for fuel exhaustion, which is the predicted
trade and not an incidental one.

Open observation for E1 to resolve: at $\gamma=1$ the agent hesitates one cell
above the goal, because the goal corner is option-poor ($\Phi\approx2.1$ against
$\approx8.6$ in open ground). The $\alpha/\gamma$ standoff is real and needs
either a sweep or a $\gamma$ schedule. This is a finding, not a bug.
