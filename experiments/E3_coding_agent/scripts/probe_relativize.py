#!/usr/bin/env python3
"""Why component 3 is inert: relativize eats a common-mode penalty.

`virtual_reward_phi` applies `relativize` to the Phi vector before raising it
to gamma. `relativize` z-scores, so any factor that multiplies *every* walker's
Phi by roughly the same amount is removed. The irreversibility penalty is
exactly such a factor once most walkers in the swarm have already committed an
irreversible operation during their M-step rollout.

This probe measures, at a single planning tick, (a) what fraction of walkers
carry irrev > 0, and (b) how much of the penalty survives relativize, as the
ratio of the coefficient of variation of the Phi vector with and without it.

This is docs/SPEC.md section 5's null-test trap operating one level down: not
on the whole layer, but on one component of it.

Writes results/p4_relativize_probe.json.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fmcphi.core import relativize                     # noqa: E402
from fmcphi.phi import phi_composite                   # noqa: E402
from repo_env import RepoEnv                           # noqa: E402
from stats import summarize                            # noqa: E402

E3 = HERE.parent
RESULTS = E3 / "results"

N, M, PHI_M, PHI_H = 24, 8, 3, 2
REPS = 60
# tokens=12 is the well-posed budget: M=8 rollout steps at ~1.17 tokens each is
# affordable, so the swarm is not already dead at the tick where Phi is read.
# tokens=6 is reported too, and is the degenerate regime where M outlives the
# budget and almost every walker is dead by tick M.
TOKEN_LEVELS = (6.0, 12.0)
TICKS = (2, 4, 6, 8)


def roll(env, rng, t):
    """Roll N walkers forward t steps from the start, then measure Phi."""
    states = [env.clone_state(env.reset()) for _ in range(N)]
    for _ in range(t):
        for i in range(N):
            states[i] = env.step(states[i], env.sample_action(states[i], rng))
    phi_pen = np.array([phi_composite(env, s, rng, m=PHI_M, h=PHI_H,
                                      budget_attr="slack_irrev", budget_max=1.0)
                        for s in states])
    phi_raw = np.array([phi_composite(env, s, rng, m=PHI_M, h=PHI_H,
                                      budget_attr="slack", budget_max=1.0)
                        for s in states])
    frac_irrev = float(np.mean([s.irrev > 0 for s in states]))
    frac_dead = float(np.mean([not env.viable(s) for s in states]))
    return phi_pen, phi_raw, frac_irrev, frac_dead


def cv(v):
    v = np.asarray(v, dtype=np.float64)
    m = v.mean()
    return float(v.std() / m) if m > 0 else 0.0


def main():
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {"experiment": "E3 / relativize probe", "N": N, "M": M, "reps": REPS,
           "token_levels": list(TOKEN_LEVELS), "ticks": list(TICKS),
           "by_lam": {}}

    print(f"{'tokens':>7} {'tick':>5} {'lam':>5}  {'dead':>18} {'irrev>0':>18}"
          f"  {'CV ratio pen/raw':>22}")
    for tokens in TOKEN_LEVELS:
        for t in TICKS:
            for lam in (0.7, 5.0, 20.0):
                env = RepoEnv(tokens=tokens, lam=lam)
                fracs, deads, ratios, spread_pen, spread_raw = [], [], [], [], []
                for r in range(REPS):
                    rng = np.random.default_rng(5000 + r)
                    phi_pen, phi_raw, frac, dead = roll(env, rng, t)
                    fracs.append(frac)
                    deads.append(dead)
                    cp, cr = cv(relativize(phi_pen)), cv(relativize(phi_raw))
                    spread_pen.append(cp)
                    spread_raw.append(cr)
                    ratios.append(cp / cr if cr > 0 else float("nan"))
                good = [r for r in ratios if not math.isnan(r)]
                key = f"tokens={tokens:g}/tick={t}/lam={lam:g}"
                out["by_lam"][key] = {
                    "frac_walkers_dead": summarize(deads, seed=0),
                    "frac_walkers_with_irrev": summarize(fracs, seed=1),
                    "cv_relativized_phi_with_penalty": summarize(spread_pen, seed=2),
                    "cv_relativized_phi_without_penalty": summarize(spread_raw, seed=3),
                    "cv_ratio": summarize(good, seed=4) if good else None,
                    "n_reps_with_defined_ratio": len(good),
                }
                b = out["by_lam"][key]
                cvr = (f"{b['cv_ratio']['mean']:.3f} "
                       f"[{b['cv_ratio']['ci95'][0]:.3f},"
                       f" {b['cv_ratio']['ci95'][1]:.3f}]"
                       if b["cv_ratio"] else "undefined (Phi degenerate)")
                print(f"{tokens:7g} {t:5d} {lam:5g}  "
                      f"{b['frac_walkers_dead']['mean']:18.3f} "
                      f"{b['frac_walkers_with_irrev']['mean']:18.3f}  {cvr:>22}")

    out["wall_clock_s"] = round(time.time() - t0, 2)
    (RESULTS / "p4_relativize_probe.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwall clock {out['wall_clock_s']}s -> {RESULTS/'p4_relativize_probe.json'}")


if __name__ == "__main__":
    main()
