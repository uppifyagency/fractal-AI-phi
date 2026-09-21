# fractal-AI-phi

> Adding the missing factor to Fractal Monte Carlo: **the entropy of futures
> reachable from a state**, not the dispersion of walkers in a swarm.

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

## Pilot result

`TrapGrid`: a cliff-walk where the short route to the goal runs along a row of
absorbing traps and the safe route costs two extra moves out of a finite fuel
budget. 10 seeds, $N=32$, $M=10$, $\alpha=\beta=1$:

| arm | goal | trap | out of fuel | mean steps |
|---|---|---|---|---|
| $\gamma = 0$ (canonical FMC) | 0 | **10** | 0 | 1.4 |
| $\gamma = 1$ (Phi on) | **6** | **0** | 4 | 24.8 |

Pure goal-seeking walks off the cliff on the first move. With $\Phi$ on, the
swarm climbs to the safe corridor, crosses it, and comes down at the far end.
Trap deaths go to zero, traded for fuel exhaustion. That is the predicted trade.

Reproduce:

```bash
uv sync && uv run pytest -q          # 16/16
uv run python -m experiments.E1_trapgrid_sweep.scripts.run_e1
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

Foundation done and tested. The five experiments are specified and
pre-registered, not yet run. Read `experiments/README.md` first.

## Credits

Fractal Monte Carlo is the work of Sergio Hernández-Cerezo and Guillem
Duran-Ballester, *Fractal AI: A Fragile Theory of Intelligence*, arXiv:1803.05049.
`src/fmcphi/core.py` is vendored unchanged from the `fmc-core` reference
implementation in [uppifyagency/fractal-AI-FMC](https://github.com/uppifyagency/fractal-AI-FMC) (MIT).
The $\Phi$ layer, `TrapGrid` and the experiment ladder are new here.

MIT.
