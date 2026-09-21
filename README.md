# fractal-AI-phi

> Adding the missing factor to Fractal Monte Carlo: **the entropy of futures
> reachable from a state**, not the dispersion of walkers in a swarm.
>
> **Outcome: the factor as proposed does not survive ablation.** What works is a
> free binary viability constraint, not the graded causal cone. See
> [`docs/SYNTHESIS.md`](docs/SYNTHESIS.md).

Canonical FMC weights each walker by $\mathrm{VR} = \widehat{R}^{\alpha}\widehat{D}^{\beta}$.
The $D$ term measures how far apart the *hypotheses* are. The causal entropic
force of Wissner-Gross & Freer (2013), $F = T_c\nabla_X S_c$, measures something
else entirely: how many *futures* are still open from where you are standing.

FMC has never contained the second one. This repository adds it:

$$\mathrm{VR}^{(i)} = \big(\widehat{R}^{(i)}\big)^{\alpha}\cdot\big(\widehat{D}^{(i)}\big)^{\beta}\cdot\big(\widehat{\Phi}^{(i)}\big)^{\gamma}$$

$$\Phi(x) = p_{\text{surv}}(x)\cdot\exp\Big(H\big(\text{distinct viable states reachable from }x\text{ in }h\text{ steps}\big)\Big)$$

$\alpha$ pulls toward the goal. $\gamma$ keeps futures open. Their ratio is the
risk dial. On a motorway you could die at any second, and you stay alive by
correcting the trajectory continuously so that tomorrow remains reachable:
$\gamma$ is the correction, $\alpha$ is getting home.

## Why $\beta$ cannot do this job

The reference project's own rocket sweep, reported there as a counter-intuitive
caveat:

| $\beta$ | $b_{\text{eff}}$ |
|---|---|
| 0.0 | $5.45\pm0.82$ |
| 1.0 | $3.52\pm0.94$ |
| 5.0 | $1.89\pm0.69$ |

Raising $\beta$ **lowers** the branching factor. A term that preserved options
would raise it. Past the anti-collapse threshold, $\beta$ is one more selector:
it rewards the isolated walker, not the state with a future.

## Result: the hypothesis did not survive its own ablations

The layer works on TrapGrid. The proposed mechanism does not. Full detail in
[`docs/SYNTHESIS.md`](docs/SYNTHESIS.md); the decisive rows, 30 paired seeds at
matched measured budget:

| arm | measured sim/decision | goal rate |
|---|---|---|
| canonical FMC | 3290 | 0.00 [0.00, 0.00] |
| $\Phi$ as specified (cone, survival-weighted) | 3291 | 0.53 [0.37, 0.70] |
| the same $\Phi$ vector **permuted across walkers** | 3314 | 0.63 [0.47, 0.80] |
| $\Phi$ replaced by uniform **noise** | 3209 | 0.57 [0.40, 0.73] |
| $\Phi = \mathbb{1}\{\text{viable}\}$, **zero extra simulator calls** | 3290 | **0.97 [0.90, 1.00]** |

Permuting $\Phi$ across walkers destroys every correspondence between a walker
and its own future and the goal rate does not move (paired diff $-0.10$
[$-0.30$, $+0.10$]). A binary "is this walker standing on a dead state"
indicator, needing no rollout, no perplexity and no survival weighting, reaches
30/30 at the budget the full layer spends to reach 16/30.

**The causal cone is not doing the work. A hard viability constraint is, and
canonical FMC simply omits it.**

Two findings survive the collapse:

- An option-preserving planner is only as safe as the irreversibilities its
  $\Phi$ can see, and it actively buys the ones it cannot: in E3 the layer cut
  the visible trap 21x while raising force-push 2.8x ($+0.085$ [$+0.051$,
  $+0.120$], $n=100$).
- A goal-less agent ($\alpha = 0$) reaches the goal exactly as often as random,
  0/50 on two maps, so the layer leaks no reward information through the
  geometry.

Reproduce:

```bash
uv sync && uv run pytest -q          # 16/16
uv run python experiments/E5_ablation/scripts/run_e5.py
```

## Layout

```
src/fmcphi/core.py        vendored verbatim from fmc-core (MIT). Do not edit.
src/fmcphi/phi.py         the Phi estimators and the three-factor virtual reward
src/fmcphi/planner.py     instrumented plan() and run_episode()
src/fmcphi/envs/          Environment protocol + TrapGrid
docs/SPEC.md              the formal statement, and the null-test trap to avoid
experiments/              the ladder, E1 to E5, one brief each
tests/                    16 tests, including exact parity with the reference
                          implementation at gamma = 0
```

## Status

E1 to E5 run, each reviewed adversarially; four of the five were sent back and
re-run. E6 is pre-registered with its instrument built and deliberately not run:
the power analysis needs 169 episodes per arm for the direct replication, and the
hand-validation showed the cone term is inert on the repository encoding, which
would guarantee a meaningless null.

Read [`docs/SYNTHESIS.md`](docs/SYNTHESIS.md) first, then
[`experiments/README.md`](experiments/README.md).

## Credits

Fractal Monte Carlo is the work of Sergio Hernández-Cerezo and Guillem
Duran-Ballester, *Fractal AI: A Fragile Theory of Intelligence*, arXiv:1803.05049.
`src/fmcphi/core.py` is vendored unchanged from the `fmc-core` reference
implementation in [uppifyagency/fractal-AI-FMC](https://github.com/uppifyagency/fractal-AI-FMC) (MIT).
The $\Phi$ layer, `TrapGrid` and the experiment ladder are new here.

MIT.
