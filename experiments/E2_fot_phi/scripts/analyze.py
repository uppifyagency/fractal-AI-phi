"""E2 analysis. Reads ../results/*.json, prints the REPORT tables, writes
../results/R6_summary_stats.json.

    cd /Users/vladvrinceanu/fractal-AI-phi && uv run python \
      experiments/E2_fot_phi/scripts/analyze.py

Every mean carries a bootstrap CI95 and every row states its n (task
constraint 5). Every comparison is between arms charged the same sim_steps per
decision (task constraint 6); the equal-(N, M) arm is printed but never used
for a verdict.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
RESULTS = _HERE.parent / "results"

from stats import boot_did, boot_diff, boot_mean, fmt, fmt_diff  # noqa: E402

BASE = "fot_budget"          # the equal-budget gamma = 0 reference
PREREG = "fot_phi"           # the pre-registered Phi arm: gamma = 1 every tick


def load(name: str) -> Dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def arms_of(recs: List[Dict]) -> List[str]:
    seen, out = set(), []
    for r in recs:
        if r["arm"] not in seen:
            seen.add(r["arm"])
            out.append(r["arm"])
    return out


def by_arm(recs: List[Dict], arm: str) -> List[Dict]:
    return [r for r in recs if r["arm"] == arm]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def accuracy_table(recs: List[Dict], title: str) -> Dict:
    rows, out = [], {}
    base = [float(r["correct"]) for r in by_arm(recs, BASE)]
    print(f"\n### {title}\n")
    print("| arm | n | accuracy [CI95] | vs fot_budget [CI95] | p | "
          "sim_steps/decision | tokens/episode [CI95] | steps [CI95] |")
    print("|---|---|---|---|---|---|---|---|")
    for arm in arms_of(recs):
        rr = by_arm(recs, arm)
        acc = boot_mean([float(r["correct"]) for r in rr], seed=1)
        tok = boot_mean([r["tokens"] for r in rr], seed=2)
        stp = boot_mean([r["steps"] for r in rr], seed=3)
        spd = int(round(np.mean([r["sim_steps"] / r["steps"] for r in rr])))
        d = boot_diff([float(r["correct"]) for r in rr], base, seed=4)
        out[arm] = {"n": acc["n"], "accuracy": acc, "tokens_per_episode": tok,
                    "steps": stp, "sim_steps_per_decision": spd,
                    "vs_fot_budget": d}
        dtxt = "-" if arm == BASE else fmt_diff(d, pct=True)
        ptxt = "-" if arm == BASE else f"{d['p_two_sided']:.3f}"
        print(f"| `{arm}` | {acc['n']} | {fmt(acc, pct=True)} | {dtxt} | "
              f"{ptxt} | {spd} | {tok['mean']:.0f} "
              f"[{tok['lo']:.0f}, {tok['hi']:.0f}] | {fmt(stp, digits=2)} |")
        rows.append(arm)
    return out


def commit_split(recs: List[Dict]) -> Dict:
    """P2: split problems by whether the equal-budget baseline commits early.

    Operational definition, fixed before looking at the Phi arms: a problem is
    in the `baseline_commits_early` split iff, in the `fot_budget` arm, the
    chain is off the correct line of reasoning after the *first* committed step
    on a strict majority of seeds. The split is computed from baseline records
    only, so it cannot be tuned by the Phi arms' behaviour.
    """
    base = by_arm(recs, BASE)
    frac = {}
    for r in base:
        frac.setdefault(r["pid"], []).append(float(bool(r["wrong_after_step1"])))
    early = {pid for pid, v in frac.items() if float(np.mean(v)) > 0.5}
    out = {"n_problems_total": len(frac), "n_problems_early": len(early),
           "splits": {}}
    print("\n### P2 split: does the gain concentrate where the baseline "
          "commits early to a wrong branch?\n")
    print(f"Split computed from the `{BASE}` arm only. "
          f"{len(early)} of {len(frac)} problems are in "
          f"`baseline_commits_early`.\n")
    print("| split | arm | n | accuracy [CI95] | vs fot_budget [CI95] | p |")
    print("|---|---|---|---|---|---|")
    for label, pids in (("baseline_commits_early", early),
                        ("baseline_stays_on_line", set(frac) - early)):
        b = [float(r["correct"]) for r in base if r["pid"] in pids]
        out["splits"][label] = {"n_problems": len(pids), "arms": {}}
        for arm in arms_of(recs):
            rr = [r for r in by_arm(recs, arm) if r["pid"] in pids]
            if not rr:
                continue
            acc = boot_mean([float(r["correct"]) for r in rr], seed=5)
            d = boot_diff([float(r["correct"]) for r in rr], b, seed=6)
            out["splits"][label]["arms"][arm] = {"accuracy": acc,
                                                 "vs_fot_budget": d}
            dtxt = "-" if arm == BASE else fmt_diff(d, pct=True)
            ptxt = "-" if arm == BASE else f"{d['p_two_sided']:.3f}"
            print(f"| {label} | `{arm}` | {acc['n']} | {fmt(acc, pct=True)} | "
                  f"{dtxt} | {ptxt} |")

    print("\nInteraction test. P2 is a claim about the *difference between the "
          "two gains*, so it\nneeds its own interval; two separate CIs cannot "
          "settle it.\n")
    print("| arm | (gain on early split) - (gain on other split) [CI95] | p |")
    print("|---|---|---|")
    e_pids, o_pids = early, set(frac) - early
    b_e = [float(r["correct"]) for r in base if r["pid"] in e_pids]
    b_o = [float(r["correct"]) for r in base if r["pid"] in o_pids]
    out["interaction"] = {}
    for arm in arms_of(recs):
        if arm in (BASE, "fot_nm"):
            continue
        rr = by_arm(recs, arm)
        a_e = [float(r["correct"]) for r in rr if r["pid"] in e_pids]
        a_o = [float(r["correct"]) for r in rr if r["pid"] in o_pids]
        did = boot_did(a_e, b_e, a_o, b_o, seed=9)
        out["interaction"][arm] = did
        print(f"| `{arm}` | {100 * did['did']:+.1f} pp "
              f"[{100 * did['lo']:+.1f}, {100 * did['hi']:+.1f}] | "
              f"{did['p_two_sided']:.3f} |")
    return out


def gamma_only_contrast(recs: List[Dict], title: str) -> Dict:
    """gamma = 1 against gamma = 0 at the *same* N and M.

    `fot_nm` and `fot_phi` both run M cycles with N walkers; the only intended
    difference is the Phi factor (the Phi arm additionally pays the m*h rollout
    surcharge, so this contrast is NOT budget matched, and it is reported only
    to isolate what gamma does, never as a verdict on P1).

    This is the decisive contrast for P3: on a tier where Phi is provably flat,
    turning gamma on must not move accuracy.
    """
    a0 = [float(r["correct"]) for r in by_arm(recs, "fot_nm")]
    a1 = [float(r["correct"]) for r in by_arm(recs, PREREG)]
    d = boot_diff(a1, a0, seed=11)
    m0, m1 = boot_mean(a0, seed=12), boot_mean(a1, seed=13)
    print(f"\n### {title}: gamma isolated at equal N and M\n")
    print("| arm | gamma | n | accuracy [CI95] |")
    print("|---|---|---|---|")
    print(f"| `fot_nm` | 0 | {m0['n']} | {fmt(m0, pct=True)} |")
    print(f"| `fot_phi` | 1 | {m1['n']} | {fmt(m1, pct=True)} |")
    print(f"\nEffect of turning gamma on: **{fmt_diff(d, pct=True)}**, "
          f"p = {d['p_two_sided']:.3f}.")
    return {"gamma0": m0, "gamma1": m1, "effect_of_gamma": d}


def sweep_table(payload: Dict) -> Dict:
    print("\n### R3 falsification control: accuracy gain vs how much "
          "information Phi carries\n")
    print("`phi_signal` is the fraction of wrong branches whose cone actually "
          "narrows. At 0 the Phi\nterm is well defined and non-constant but "
          "uncorrelated with correctness, so any surviving\n'gain' is a "
          "budget artifact, not the Phi layer.\n")
    print("| phi_signal | arm | n | accuracy [CI95] | vs fot_budget [CI95] | p |")
    print("|---|---|---|---|---|---|")
    out = {}
    for ps, recs in payload["records_by_phi_signal"].items():
        base = [float(r["correct"]) for r in by_arm(recs, BASE)]
        out[ps] = {}
        for arm in arms_of(recs):
            rr = by_arm(recs, arm)
            if arm == "fot_nm":
                continue
            acc = boot_mean([float(r["correct"]) for r in rr], seed=7)
            d = boot_diff([float(r["correct"]) for r in rr], base, seed=8)
            out[ps][arm] = {"accuracy": acc, "vs_fot_budget": d}
            dtxt = "-" if arm == BASE else fmt_diff(d, pct=True)
            ptxt = "-" if arm == BASE else f"{d['p_two_sided']:.3f}"
            print(f"| {ps} | `{arm}` | {acc['n']} | {fmt(acc, pct=True)} | "
                  f"{dtxt} | {ptxt} |")
    return out


def diagnostics_table(payload: Dict) -> Dict:
    print("\n### R4 mechanism diagnostics\n")
    print("Phi by what the reasoning node actually is "
          "(phi_composite, m=6, h=2):\n")
    print("| node type | n | mean Phi | std |")
    print("|---|---|---|---|")
    for k, v in payload["phi_by_node_type"].items():
        if v["mean"] is None:
            continue
        print(f"| {k} | {v['n']} | {v['mean']:.3f} | {v['std']:.3f} |")
    print("\nSpread of Phi across the three sibling continuations of one "
          "decision. This is what a\ngamma > 0 planner can act on; an absolute "
          "level it cannot.\n")
    print("| tier | decisions sampled | mean within-decision std(Phi) | "
          "std(relativize(Phi)) | decisions with exactly zero spread |")
    print("|---|---|---|---|---|")
    for key in ("phi_action_spread_multi", "phi_action_spread_single"):
        d = payload[key]
        print(f"| {d['tier']} | {d['n_decisions_sampled']} | "
              f"{d['phi_std_within_decision']:.4f} | "
              f"{d['relativized_phi_std_within_decision']:.4f} | "
              f"{100 * d['frac_decisions_with_zero_spread']:.1f}% |")
    return {"phi_by_node_type": payload["phi_by_node_type"],
            "spread_multi": payload["phi_action_spread_multi"],
            "spread_single": payload["phi_action_spread_single"]}


# ---------------------------------------------------------------------------

def main() -> int:
    summary: Dict = {}
    r1 = load("R1_multi_main")
    r2 = load("R2_single_control")
    r3 = load("R3_phi_signal_sweep")
    r4 = load("R4_diagnostics")
    r5 = load("R5_heldout")

    print("# E2 numbers\n")
    print(f"Matched budget: "
          f"{r1['matched_budget_sim_steps_per_decision']} sim_steps per "
          f"decision = "
          f"{r1['matched_budget_sim_steps_per_decision'] * r1['tokens_per_call']}"
          f" tokens per decision, for every arm except `fot_nm`.")

    summary["R1_multi_main"] = accuracy_table(
        r1["records"], "R1 multi-step problems (the main comparison)")
    summary["R1_p2_split"] = commit_split(r1["records"])
    summary["R5_heldout"] = accuracy_table(
        r5["records"], "R5 held-out problem block (pid 5000-5039), same config")
    summary["R2_single_control"] = accuracy_table(
        r2["records"], "R2 single-step problems (P3 control)")
    summary["gamma_only_multi"] = gamma_only_contrast(
        r1["records"] + r5["records"], "Multi-step (R1 + R5 pooled)")
    summary["gamma_only_single"] = gamma_only_contrast(
        r2["records"], "Single-step (R2)")
    summary["R3_phi_signal_sweep"] = sweep_table(r3)
    summary["R4_diagnostics"] = diagnostics_table(r4)

    # Pooled R1 + R5 for the headline verdicts.
    pooled = r1["records"] + r5["records"]
    summary["R1_R5_pooled"] = accuracy_table(
        pooled, "R1 + R5 pooled (80 problems, 12 seeds)")
    summary["R1_R5_pooled_p2_split"] = commit_split(pooled)

    r7 = load("R7_budget_control")
    summary["R7_budget_control"] = accuracy_table(
        r7["records"], "R7 budget control: baseline handed 25% and 50% more "
                       "depth than the reference")

    out = RESULTS / "R6_summary_stats.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
