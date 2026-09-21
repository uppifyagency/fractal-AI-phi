"""Diagnostic for E5: is Phi actually state-dependent on TrapGrid?

The shuffled and constant arms only mean something if the real Phi vector
carries state information in the first place. This also discharges the
docs/SPEC.md section 5 null-test trap: if Phi were constant across the grid the
term would vanish after relativize and the layer could not be tested here.

Probed at several fuel levels, not just one: episodes start at fuel = 24 and
run down to 0, and with h = 3 the rollouts themselves burn fuel, so at low fuel
every continuation dies and Phi collapses toward 0 everywhere. That low-fuel
regime is where the 0.32-0.37 fuel-death rate is decided, so it is reported
here instead of being extrapolated from the fuel = 20 slice.

Writes results/E5_phi_landscape.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fmcphi.envs.trapgrid import TrapGrid, State  # noqa: E402
from fmcphi.phi import phi_cone, phi_viable_actions  # noqa: E402
from e5_lib import boot_ci  # noqa: E402

REPS = 400
RESULTS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "results"))


FUEL_LEVELS = (20.0, 8.0, 4.0, 3.0, 2.0)


def probe(env, rng, fuel):
    cells = {}
    for y in range(env.height):
        for x in range(env.width):
            s = State(x=x, y=y, fuel=fuel)
            vals = [phi_cone(env, s, rng, m=4, h=3) for _ in range(REPS)]
            m, lo, hi = boot_ci(vals)
            cells[f"{x},{y}"] = dict(
                x=x, y=y, trap=env.is_trap(x, y), goal=(x, y) == env.goal,
                phi_cone_mean=m, phi_cone_lo=lo, phi_cone_hi=hi,
                phi_cone_sd=float(np.std(vals)),
                # fraction of individual estimates that are exactly 0, i.e. the
                # rate at which kill_dead would zero a walker standing here
                p_phi_zero=float(np.mean([v == 0.0 for v in vals])),
                phi_viable_actions=phi_viable_actions(env, s),
            )
    return cells


def stats(cells):
    """Report the two populations separately: all cells, and alive cells only."""
    all_means = [c["phi_cone_mean"] for c in cells.values()]
    alive = [c["phi_cone_mean"] for c in cells.values()
             if not c["trap"] and c["phi_cone_mean"] is not None]
    alive_cells = [c for c in cells.values() if not c["trap"]]
    alive = [c["phi_cone_mean"] for c in alive_cells]
    return dict(
        n_cells=len(cells), n_alive_cells=len(alive_cells),
        phi_min_over_all_cells=float(np.min(all_means)),
        phi_max_over_all_cells=float(np.max(all_means)),
        phi_min_over_alive_cells=float(np.min(alive)),
        phi_max_over_alive_cells=float(np.max(alive)),
        phi_mean_over_alive_cells=float(np.mean(alive)),
        phi_sd_over_alive_cells=float(np.std(alive)),
        n_alive_cells_with_phi_zero=sum(1 for v in alive if v == 0.0),
        mean_p_phi_zero_over_alive_cells=float(np.mean(
            [c["p_phi_zero"] for c in alive_cells])),
        n_trap_cells_with_phi_zero=sum(
            1 for c in cells.values() if c["trap"] and c["phi_cone_mean"] == 0.0),
        n_trap_cells=sum(1 for c in cells.values() if c["trap"]),
    )


def main():
    env = TrapGrid(fuel=24.0)
    rng = np.random.default_rng(0)
    levels = {}
    for fuel in FUEL_LEVELS:
        cells = probe(env, rng, fuel)
        st = stats(cells)
        levels[str(fuel)] = dict(fuel=fuel, cells=cells, **st)
        print(f"\n--- fuel = {fuel} ---")
        print("row y  " + "  ".join(f"x{x:<4d}" for x in range(env.width)))
        for y in reversed(range(env.height)):
            row = "  ".join(f"{cells[f'{x},{y}']['phi_cone_mean']:.2f} "
                            for x in range(env.width))
            print(f"  {y}    {row}")
        print(f"all {st['n_cells']} cells: [{st['phi_min_over_all_cells']:.2f}, "
              f"{st['phi_max_over_all_cells']:.2f}]   "
              f"{st['n_alive_cells']} alive cells: "
              f"[{st['phi_min_over_alive_cells']:.2f}, "
              f"{st['phi_max_over_alive_cells']:.2f}] "
              f"mean {st['phi_mean_over_alive_cells']:.2f} "
              f"sd {st['phi_sd_over_alive_cells']:.3f}; "
              f"P(Phi==0 | alive cell) {st['mean_p_phi_zero_over_alive_cells']:.3f}; "
              f"trap cells with Phi exactly 0: "
              f"{st['n_trap_cells_with_phi_zero']}/{st['n_trap_cells']}")

    out = dict(reps_per_cell=REPS, phi_m=4, phi_h=3,
               fuel_levels=list(FUEL_LEVELS), levels=levels,
               # back-compatible top-level block for the fuel=20 slice
               fuel_at_probe=20.0, **{k: v for k, v in levels["20.0"].items()
                                      if k != "fuel"})
    with open(os.path.join(RESULTS, "E5_phi_landscape.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
