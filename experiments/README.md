# Experiment ladder

Cheap to expensive. Each rung has its own brief, its own `scripts/`, its own
`results/` (JSON, one file per run) and a `REPORT.md` written at the end.

| # | Brief | Cost | Needs an LLM |
|---|---|---|---|
| E1 | [gamma sweep on TrapGrid](E1_trapgrid_sweep/BRIEF.md) | free | no |
| E2 | [Fractal-of-Thought with Phi](E2_fot_phi/BRIEF.md) | ~$25 | yes |
| E3 | [Phi for a coding agent](E3_coding_agent/BRIEF.md) | free to build | at run time |
| E4 | [competence with no goal](E4_no_goal/BRIEF.md) | free | no |
| E5 | [ablations](E5_ablation/BRIEF.md) | free | no |

Rule for every rung: the decision rule is written into the brief **before** the
run. A result that was not pre-registered is a hypothesis for the next rung, not
a finding in this one.

Shared constraint from `docs/SPEC.md` section 5: an environment with no
irreversibility and no finite budget cannot test this layer. Check that the cone
actually narrows before spending anything.
