"""E2 run driver. Free, deterministic, mock backend only.

    cd /Users/vladvrinceanu/fractal-AI-phi && uv run python \
      experiments/E2_fot_phi/scripts/run_all.py

Writes one JSON per run into ../results/. Nothing here calls a paid API.

Runs
----
R1 multi_main       the pre-registered comparison, 6 arms, all budget matched
R2 single_control   P3: the same 6 arms on single-step problems
R3 phi_signal_sweep the falsification control: how the gain depends on whether
                    Phi carries any information about correctness at all
R4 diagnostics      Phi by node type, and Phi spread within a decision
R5 heldout          replication of R1 on a disjoint block of problem ids, run
                    because the amortised arm was added after looking at R1
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import fot_phi                                                  # noqa: E402
from fot_phi import (TOKENS_PER_CALL, assert_equal_budget,       # noqa: E402
                     make_arms, phi_action_spread, phi_by_node_type,
                     run_grid, save)

N, M, PHI_M, PHI_H = 16, 4, 3, 2
PHI_SIGNAL, P_TRAP, P_DETECT = 0.75, 0.6, 0.6
N_STAGES, TOKENS0 = 3, 8.0


def _cfg(**kw):
    base = dict(N=N, M=M, phi_m=PHI_M, phi_h=PHI_H, phi_signal=PHI_SIGNAL,
                p_trap=P_TRAP, p_detect=P_DETECT, n_stages=N_STAGES,
                tokens0=TOKENS0)
    base.update(kw)
    return base


def grid_run(name: str, tier: str, n_problems: int, n_seeds: int,
             phi_signal: float = PHI_SIGNAL, pid_offset: int = 0) -> dict:
    arms = make_arms(N, M, PHI_M, PHI_H)
    budget = assert_equal_budget(arms)
    print(f"\n[{name}] tier={tier} problems={n_problems} seeds={n_seeds} "
          f"phi_signal={phi_signal} pid_offset={pid_offset}")
    for a in arms:
        print(f"    {a.name:16s} gamma={a.gamma} M={a.M:>2d} "
              f"phi_every={a.phi_every:>2d} sim_steps/decision="
              f"{a.budget_per_plan()}")
    t0 = time.time()
    recs = run_grid(tier, arms, n_problems, n_seeds, phi_signal, P_TRAP,
                    P_DETECT, N_STAGES, TOKENS0, pid_offset=pid_offset)
    payload = {
        "run": name,
        "backend": "mock",
        "config": _cfg(tier=tier, problems=n_problems, seeds=n_seeds,
                       phi_signal=phi_signal, pid_offset=pid_offset),
        "arms": [asdict(a) for a in arms],
        "tokens_per_call": TOKENS_PER_CALL,
        "matched_budget_sim_steps_per_decision": budget,
        "wall_clock_s": round(time.time() - t0, 1),
        "records": recs,
    }
    save(name, payload)
    return payload


def main() -> int:
    t_all = time.time()

    grid_run("R1_multi_main", "multi", n_problems=40, n_seeds=12)
    grid_run("R2_single_control", "single", n_problems=40, n_seeds=12)

    sweep = {}
    for ps in (0.0, 0.25, 0.5, 0.75, 1.0):
        p = grid_run(f"_sweep_{ps}", "multi", n_problems=20, n_seeds=8,
                     phi_signal=ps)
        sweep[str(ps)] = p["records"]
    save("R3_phi_signal_sweep", {
        "run": "R3_phi_signal_sweep",
        "backend": "mock",
        "config": _cfg(tier="multi", problems=20, seeds=8,
                       phi_signal_values=[0.0, 0.25, 0.5, 0.75, 1.0]),
        "arms": [asdict(a) for a in make_arms(N, M, PHI_M, PHI_H)],
        "tokens_per_call": TOKENS_PER_CALL,
        "records_by_phi_signal": sweep,
    })
    for ps in (0.0, 0.25, 0.5, 0.75, 1.0):
        (fot_phi.RESULTS / f"_sweep_{ps}.json").unlink(missing_ok=True)

    print("\n[R4] diagnostics")
    save("R4_diagnostics", {
        "run": "R4_diagnostics",
        "backend": "mock",
        "config": _cfg(),
        "phi_by_node_type": phi_by_node_type(
            n_problems=40, n_samples=80, phi_m=6, phi_h=PHI_H,
            phi_signal=PHI_SIGNAL, p_trap=P_TRAP, p_detect=P_DETECT,
            n_stages=N_STAGES, tokens0=TOKENS0),
        "phi_action_spread_multi": phi_action_spread(
            "multi", 40, n_decisions=64, phi_m=6, phi_h=PHI_H,
            phi_signal=PHI_SIGNAL, p_trap=P_TRAP, p_detect=P_DETECT,
            n_stages=N_STAGES, tokens0=TOKENS0),
        "phi_action_spread_single": phi_action_spread(
            "single", 40, n_decisions=64, phi_m=6, phi_h=PHI_H),
    })

    grid_run("R5_heldout", "multi", n_problems=40, n_seeds=12, pid_offset=5000)

    print(f"\ntotal wall clock {time.time() - t_all:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
