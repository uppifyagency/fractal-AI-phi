"""Power analysis for the real-agent replication of E3's force-push finding.

E3 (simulated action space, n=100/arm) measured the per-episode force-push
share going 0.048 [0.026, 0.073] -> 0.133 [0.106, 0.162] when the Phi layer is
switched on. This asks the only question that matters before spending anything:
how many real-agent episodes would it take to see that again?
"""

import numpy as np

RNG = np.random.default_rng(20260921)
P0, P1 = 0.048, 0.133          # E3 point estimates, gamma=0 and gamma=1
ALPHA = 0.05
REPS = 20000


def power_two_proportions(n, p0, p1, alpha=ALPHA, reps=REPS):
    """Empirical power of a two-sided Fisher-style z test on two proportions."""
    a = RNG.binomial(n, p0, reps) / n
    b = RNG.binomial(n, p1, reps) / n
    pool = (a + b) / 2
    se = np.sqrt(2 * pool * (1 - pool) / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.abs(b - a) / se
    z = np.nan_to_num(z, nan=0.0, posinf=0.0)
    crit = 1.959963984540054
    return float((z > crit).mean())


def min_detectable(n, p0, target=0.80):
    """Smallest p1 > p0 this n can detect at `target` power."""
    lo, hi = p0, 0.99
    for _ in range(40):
        mid = (lo + hi) / 2
        if power_two_proportions(n, p0, mid) < target:
            lo = mid
        else:
            hi = mid
    return hi


print("Replicating E3's effect (0.048 -> 0.133) with real agents")
print(f"{'n/arm':>7} {'power':>8}   {'min detectable p1':>18}")
for n in (10, 15, 20, 25, 30, 40, 60, 100, 150, 200, 300):
    pw = power_two_proportions(n, P0, P1)
    mdp = min_detectable(n, P0)
    print(f"{n:>7} {pw:>8.3f}   {mdp:>18.3f}")

print()
for target in (0.80, 0.90):
    n = next(n for n in range(10, 3000) if power_two_proportions(n, P0, P1) >= target)
    print(f"n needed per arm for {target:.0%} power at the E3 effect size: {n}")


# ---------------------------------------------------------------------------
# The redesign: measure per DECISION, not per episode.
# ---------------------------------------------------------------------------
# A spontaneous force-push is a rare event (5-13%), and rare events need huge n.
# But the same mechanism can be probed as a CHOICE: put a history-rewriting
# option on the menu at every decision and measure how often it is selected.
# Base rates near 0.5 need far fewer observations, and one episode yields many
# decisions, so the unit of analysis multiplies.

print()
print("Redesign: selection rate of the history-rewriting option, per DECISION")
print(f"{'n/arm':>7}  {'0.20->0.45':>11} {'0.20->0.50':>11} {'0.25->0.55':>11}")
for n in (20, 30, 40, 50, 75, 100, 150, 200):
    row = [power_two_proportions(n, a, b) for a, b in
           ((0.20, 0.45), (0.20, 0.50), (0.25, 0.55))]
    print(f"{n:>7}  {row[0]:>11.3f} {row[1]:>11.3f} {row[2]:>11.3f}")

print()
for pair in ((0.20, 0.45), (0.20, 0.50)):
    n = next(n for n in range(5, 3000) if power_two_proportions(n, *pair) >= 0.80)
    print(f"n decisions per arm for 80% power, {pair[0]} -> {pair[1]}: {n}")
