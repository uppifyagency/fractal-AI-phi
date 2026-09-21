"""R7: is the amortised Phi arm's gain just more total tokens?

Per-decision budget is matched exactly in R1/R5 (448 sim_steps for every arm but
`fot_nm`), but total tokens per *episode* are not, because the arms take
different numbers of real reasoning steps before they answer. The best Phi arm
spends about 11% more tokens per episode than `fot_budget` for that reason.

This run buys the gamma = 0 baseline strictly more depth than that, M = 35 and
M = 42 against the reference M = 28, and puts it against the Phi arm. If a
baseline that is handed 25-50% more budget still does not reach the Phi arm, the
gain is not a budget artifact. If it does, it is, and P1 stays falsified in both
its pre-registered and its exploratory form.

    cd /Users/vladvrinceanu/fractal-AI-phi && uv run python \
      experiments/E2_fot_phi/scripts/run_budget_control.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from fot_phi import TOKENS_PER_CALL, Arm, run_grid, save     # noqa: E402

N, M, PHI_M, PHI_H = 16, 4, 3, 2
PHI_SIGNAL, P_TRAP, P_DETECT = 0.75, 0.6, 0.6
N_STAGES, TOKENS0 = 3, 8.0


def main() -> int:
    arms = [
        Arm("fot_budget", 0.0, N, 28, PHI_M, PHI_H, 1),
        Arm("fot_budget_M35", 0.0, N, 35, PHI_M, PHI_H, 1),
        Arm("fot_budget_M42", 0.0, N, 42, PHI_M, PHI_H, 1),
        Arm("fot_phi_M22k22", 1.0, N, 22, PHI_M, PHI_H, 22),
    ]
    for a in arms:
        print(f"    {a.name:16s} gamma={a.gamma} M={a.M:>2d} "
              f"phi_every={a.phi_every:>2d} "
              f"sim_steps/decision={a.budget_per_plan()}")
    t0 = time.time()
    recs = []
    for off in (0, 5000):                    # both problem blocks, pooled
        recs += run_grid("multi", arms, 40, 12, PHI_SIGNAL, P_TRAP, P_DETECT,
                         N_STAGES, TOKENS0, pid_offset=off)
    save("R7_budget_control", {
        "run": "R7_budget_control",
        "backend": "mock",
        "config": {"N": N, "problems_per_block": 40, "seeds": 12,
                   "pid_blocks": [0, 5000], "phi_signal": PHI_SIGNAL,
                   "p_trap": P_TRAP, "p_detect": P_DETECT,
                   "n_stages": N_STAGES, "tokens0": TOKENS0},
        "arms": [asdict(a) for a in arms],
        "tokens_per_call": TOKENS_PER_CALL,
        "wall_clock_s": round(time.time() - t0, 1),
        "records": recs,
    })
    print(f"wall clock {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
