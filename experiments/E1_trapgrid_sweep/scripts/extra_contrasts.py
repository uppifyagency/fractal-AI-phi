"""Paired contrasts for sweep 3 (phi_every) and sweep 4 (cone geometry)."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_e1 import load, paired_delta  # noqa: E402
from e1_lib import write_json  # noqa: E402

out = {}
print("### Sweep 3 paired contrasts vs phi_every=1 (same 30 seeds)\n")
print("| contrast | sim/dec ratio | d goal rate [CI95] | d trap rate [CI95] |")
print("|---|---|---|---|")
ref = load("sweep3_phi_every_1.json")
for k in (2, 3, 5):
    a = load(f"sweep3_phi_every_{k}.json")
    dg = paired_delta(ref["episodes"], a["episodes"], lambda e: 1.0*(e["outcome"] == "goal"))
    dt = paired_delta(ref["episodes"], a["episodes"], lambda e: 1.0*(e["outcome"] == "trap"))
    ratio = a["sim_steps_per_decision"]["mean"] / ref["sim_steps_per_decision"]["mean"]
    out[f"phi_every{k}_vs_1"] = {"goal": dg, "trap": dt, "cost_ratio": ratio}
    print(f"| phi_every={k} minus phi_every=1 | {ratio:.2f}x | "
          f"{dg[0]:+.3f} [{dg[1]:+.3f}, {dg[2]:+.3f}] | {dt[0]:+.3f} [{dt[1]:+.3f}, {dt[2]:+.3f}] |")

print("\n### Sweep 4 paired contrasts vs phi_m=4 phi_h=3 (same 15 seeds)\n")
print("| contrast | sim/dec | d goal rate [CI95] |")
print("|---|---|---|")
ref4 = load("sweep4_m4_h3.json")
for m in (2, 4, 8):
    for h in (1, 3, 5):
        if (m, h) == (4, 3):
            continue
        a = load(f"sweep4_m{m}_h{h}.json")
        dg = paired_delta(ref4["episodes"], a["episodes"], lambda e: 1.0*(e["outcome"] == "goal"))
        out[f"m{m}h{h}_vs_m4h3"] = {"goal": dg,
                                    "sim_per_dec": a["sim_steps_per_decision"]["mean"]}
        print(f"| m={m} h={h} minus m=4 h=3 | {a['sim_steps_per_decision']['mean']:.0f} | "
              f"{dg[0]:+.3f} [{dg[1]:+.3f}, {dg[2]:+.3f}] |")

print("\n### Depth effect pooled over m (h=1 and h=5 vs h=3), 45 episodes per level\n")
pool = {h: [e for m in (2, 4, 8) for e in load(f"sweep4_m{m}_h{h}.json")["episodes"]]
        for h in (1, 3, 5)}
print("| contrast | n per arm | d goal rate [CI95] |")
print("|---|---|---|")
for h in (1, 5):
    dg = paired_delta(pool[3], pool[h], lambda e: 1.0*(e["outcome"] == "goal"))
    out[f"pooled_h{h}_vs_h3"] = dg
    print(f"| h={h} minus h=3 | 45 | {dg[0]:+.3f} [{dg[1]:+.3f}, {dg[2]:+.3f}] |")
for m in (2, 8):
    poolm = {mm: [e for h in (1, 3, 5) for e in load(f"sweep4_m{mm}_h{h}.json")["episodes"]]
             for mm in (2, 4, 8)}
    dg = paired_delta(poolm[4], poolm[m], lambda e: 1.0*(e["outcome"] == "goal"))
    out[f"pooled_m{m}_vs_m4"] = dg
    print(f"| m={m} minus m=4 | 45 | {dg[0]:+.3f} [{dg[1]:+.3f}, {dg[2]:+.3f}] |")

write_json("extra_contrasts.json", out)
