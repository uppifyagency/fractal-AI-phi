#!/usr/bin/env python3
"""phi_repo: the Phi layer for a repository.

The reference `fractal-coding-loop` plugin's survival term is `R_alive`, a
binary multiplicative factor that is 0 when the walker's branch does not
compile. Binary and instantaneous: it notices death, it does not anticipate it.
A walker that just deleted the failing test has `R_alive = 1` and a perfect
`R_tests`, and is already dead with respect to the task.

This module supplies the anticipating factor. Following docs/SPEC.md section 2,

    Phi(x) = p_surv(x) * exp(H({distinct viable states reachable from x}))

specialised to a repository as three multiplicative components, none of which
needs a model call:

    phi_repo(x) = phi_viable_actions_repo(x)/K      # 1. option count, depth 1
                * phi_slack_multi(budgets)          # 2. budget slack
                * phi_irreversibility(ops)          # 3. cone contractions

Multiplicative, not additive (paper section 2.2.2): under an additive
composition a large enough goal reward buys a state with no futures, which is
exactly the shortcut this layer exists to refuse.

Component 1 is `fmcphi.phi.phi_viable_actions` with the environment being the
repo. Component 2 is `fmcphi.phi.phi_slack` generalised to several budgets.
Component 3 is new here and is the part that is genuinely repo-specific: a
static classifier over the proposed tool call and the diff it would produce.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from fmcphi.phi import phi_cone, phi_viable_actions

# ---------------------------------------------------------------------------
# Component 3: static irreversibility classifier
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IrreversibleOp:
    name: str
    weight: float
    evidence: str

    def as_dict(self):
        return {"name": self.name, "weight": self.weight, "evidence": self.evidence}


# (regex, name, base weight, amplified_if_published)
#
# Weights are ordinal, not calibrated: 1.0 is "the option is gone for good",
# 0.5 is "recoverable but only by someone who kept a copy". The ordering is the
# claim; the exact numbers are a knob, and REPORT.md reports the sensitivity.
_PATTERNS: Tuple[Tuple[str, str, float, bool], ...] = (
    (r"\bgit\s+push\b[^\n]*--force-with-lease", "force-push-with-lease", 0.4, True),
    # --force must not also fire on --force-with-lease, or the safer operation
    # would score worse than the reckless one.
    (r"\bgit\s+push\b[^\n]*(--force(?!-with-lease)\b|(?<!-)-f\b)", "force-push", 0.9, True),
    (r"\bgit\s+push\b[^\n]*(--delete\b|\s:\w)", "delete-remote-branch", 1.0, False),
    (r"\bgit\s+commit\b[^\n]*--amend", "amend", 0.4, True),
    (r"\bgit\s+rebase\b", "rebase", 0.4, True),
    (r"\bgit\s+reset\b[^\n]*--hard", "reset-hard", 0.7, True),
    (r"\bgit\s+clean\b[^\n]*-[a-z]*f", "clean-force", 0.8, False),
    (r"\bgit\s+branch\b[^\n]*\s-D\b", "delete-branch", 0.6, True),
    (r"\bgit\s+(filter-branch|filter-repo)\b", "history-rewrite", 1.0, True),
    (r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r)\b", "rm-rf", 1.0, False),
    (r"(?<!\.)\brm\s+(?!-)", "rm", 0.5, False),
    (r"\b(DROP\s+TABLE|TRUNCATE\s+TABLE|DROP\s+DATABASE)\b", "destructive-sql", 1.0, False),
    (r"\bALTER\s+TABLE\b[^\n]*\bDROP\s+COLUMN\b", "drop-column", 1.0, False),
    (r"\b(alembic|django-admin|manage\.py)\b[^\n]*\bmigrate\b", "db-migration", 0.5, False),
    (r"\balembic\s+upgrade\b", "db-migration", 0.5, False),
    (r"\bnpm\s+publish\b|\btwine\s+upload\b|\bcargo\s+publish\b", "publish-package", 1.0, False),
    (r"\bgh\s+pr\s+merge\b", "merge-pr", 0.7, False),
    (r"\bterraform\s+apply\b", "terraform-apply", 0.8, False),
    (r"\baws\s+s3\s+rm\b|\bgsutil\s+rm\b", "cloud-object-delete", 1.0, False),
    (r"\b(sendmail|mailx|mail\s+-s)\b", "send-email", 1.0, False),
    (r"\bslack\b[^\n]*\bchat\.postMessage\b", "send-message", 1.0, False),
)

_COMPILED = tuple((re.compile(p, re.IGNORECASE), n, w, amp) for p, n, w, amp in _PATTERNS)

# Diff-side detectors: a deletion is a cone contraction even when the command
# that produced it looks innocent.
_DIFF_DELETED_FILE = re.compile(r"^deleted file mode ", re.MULTILINE)
_DIFF_FILE_HEADER = re.compile(r"^\+\+\+ b/(\S+)", re.MULTILINE)
_DIFF_MINUS_HEADER = re.compile(r"^--- a/(\S+)", re.MULTILINE)
_TEST_PATH = re.compile(r"(^|/)(tests?|spec)/|(^|/)test_[^/]*\.py$|_test\.(py|go|js|ts)$")


def classify_irreversibility(
    command: str = "",
    diff: str = "",
    published: bool = False,
) -> List[IrreversibleOp]:
    """Detect option-destroying operations from a tool call and its diff.

    Static, deterministic, no model call. `published` is a real git fact (is
    this branch on a remote), and it amplifies the rewriting operations: an
    amend on a local branch costs nothing, an amend on a pushed branch forces
    every other clone to recover by hand.

    Returns the list of detected operations; an empty list means "reversible as
    far as this detector can tell", which is a statement about the detector,
    not about the world. See REPORT.md, failure mode F3.
    """
    found: List[IrreversibleOp] = []
    seen = set()
    for rx, name, weight, amplified in _COMPILED:
        m = rx.search(command)
        if not m:
            continue
        if name in seen:
            continue
        seen.add(name)
        w = min(1.0, weight * 2.0) if (amplified and published) else weight
        found.append(IrreversibleOp(name, w, m.group(0).strip()))

    if diff:
        n_deleted = len(_DIFF_DELETED_FILE.findall(diff))
        if n_deleted:
            removed = [p for p in _DIFF_MINUS_HEADER.findall(diff)
                       if p not in set(_DIFF_FILE_HEADER.findall(diff))]
            test_deletions = [p for p in removed if _TEST_PATH.search(p)]
            if test_deletions:
                found.append(IrreversibleOp(
                    "delete-test-file", 0.7,
                    ",".join(sorted(test_deletions)[:3])))
            other = n_deleted - len(test_deletions)
            if other > 0:
                found.append(IrreversibleOp("delete-file", 0.3 * other,
                                            f"{other} file(s)"))
    return found


def phi_irreversibility(ops: Iterable[IrreversibleOp], lam: float = 0.7) -> float:
    """Component 3: exp(-lam * total option-destruction weight), in (0, 1].

    Exponential and not linear so the factor is multiplicative in the weights:
    two independent half-destructions compose like one whole one. It never
    reaches 0, because the hard kill belongs to `viable`/`kill_dead`, not here
    (docs/SPEC.md section 3 keeps "dead" and "merely cornered" distinct).
    """
    total = sum(op.weight for op in ops)
    return float(np.exp(-lam * total))


# ---------------------------------------------------------------------------
# Component 2: several budgets
# ---------------------------------------------------------------------------

def phi_slack_multi(
    budgets: Dict[str, Tuple[float, float]],
    mode: str = "min",
) -> float:
    """Generalise `fmcphi.phi.phi_slack` to context, tokens, wall clock, money.

    `budgets` maps a name to (remaining, maximum). Returns a value in [0, 1].

    mode="min"  : the binding constraint. An agent with 90% of its money and 2%
                  of its context window has 2% of its freedom.
    mode="prod" : independent budgets, each one a separate way to die.

    "min" is the default because in practice the budgets are strongly coupled
    (spending tokens spends money and wall clock), and "prod" then triple-counts
    the same exhaustion. REPORT.md reports both.
    """
    if not budgets:
        return 1.0
    vals = []
    for name, (remaining, maximum) in budgets.items():
        if maximum <= 0:
            vals.append(1.0)
        else:
            vals.append(max(0.0, min(1.0, float(remaining) / float(maximum))))
    if mode == "prod":
        out = 1.0
        for v in vals:
            out *= v
        return float(out)
    if mode == "min":
        return float(min(vals))
    raise ValueError(f"unknown mode {mode!r}")


# ---------------------------------------------------------------------------
# Component 1 + composite
# ---------------------------------------------------------------------------

def phi_viable_actions_repo(env, state) -> float:
    """Component 1, normalised to [0, 1].

    Straight `fmcphi.phi.phi_viable_actions` divided by K, so the three
    components live on a common scale and the product stays in [0, 1].
    """
    k = len(list(env.actions()))
    if k == 0:
        return 0.0
    return phi_viable_actions(env, state) / float(k)


def phi_repo(
    env,
    state,
    budgets: Optional[Dict[str, Tuple[float, float]]] = None,
    ops: Sequence[IrreversibleOp] = (),
    lam: float = 0.7,
    slack_mode: str = "min",
) -> float:
    """The depth-1 instrument: cheap, deterministic, no simulation, no model.

    Cost is K environment steps (component 1) plus a regex pass (component 3).
    Use `phi_repo_deep` when the budget allows a real fan-out.
    """
    return (
        phi_viable_actions_repo(env, state)
        * phi_slack_multi(budgets or {}, mode=slack_mode)
        * phi_irreversibility(ops, lam=lam)
    )


def phi_repo_deep(
    env,
    state,
    rng: np.random.Generator,
    m: int = 4,
    h: int = 3,
    budgets: Optional[Dict[str, Tuple[float, float]]] = None,
    ops: Sequence[IrreversibleOp] = (),
    lam: float = 0.7,
    slack_mode: str = "min",
    weight_by_survival: bool = True,
) -> float:
    """Component 1 replaced by the real causal cone of docs/SPEC.md section 2.

    Cost is m*h simulated tool calls. Normalised by m so the result stays in
    [0, 1] and is comparable with `phi_repo`.
    """
    cone = phi_cone(env, state, rng, m=m, h=h,
                    weight_by_survival=weight_by_survival) / float(m)
    return (
        cone
        * phi_slack_multi(budgets or {}, mode=slack_mode)
        * phi_irreversibility(ops, lam=lam)
    )


# ---------------------------------------------------------------------------
# Real git facts
# ---------------------------------------------------------------------------

class GitProbe:
    """Read-only probe over a real repository. Never writes, never rewrites."""

    def __init__(self, repo: Path):
        self.repo = Path(repo)

    def _git(self, *args) -> str:
        p = subprocess.run(["git", *args], cwd=str(self.repo),
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           text=True, check=False)
        return p.stdout

    def is_published(self, rev: str) -> bool:
        """True iff `rev` is contained in some remote-tracking branch.

        This is the fact that turns amend/rebase/force-push from free into
        irreversible, and it costs one git call.
        """
        out = self._git("branch", "-r", "--contains", rev).strip()
        return bool(out)

    def diff(self, rev: str) -> str:
        return self._git("show", "--format=", "--patch", rev)

    def changed_files(self, rev: str) -> List[str]:
        out = self._git("show", "--format=", "--name-only", rev).strip()
        return [line for line in out.splitlines() if line]


def phi_repo_from_git(
    probe: GitProbe,
    rev: str,
    pending_command: str = "",
    budgets: Optional[Dict[str, Tuple[float, float]]] = None,
    n_viable_actions: Optional[float] = None,
    k_actions: int = 6,
    lam: float = 0.7,
    slack_mode: str = "min",
) -> dict:
    """Evaluate phi_repo against a real SHA in a real repository.

    `n_viable_actions` is component 1; it has to come from outside because
    deciding whether a strategy "still applies without breaking the build"
    requires running the build. In E3 it is read from the measured node table
    that `build_fixture.py` produced; in a live loop it would come from the
    plugin's own per-walker compile check.
    """
    published = probe.is_published(rev)
    ops = classify_irreversibility(pending_command, probe.diff(rev), published)
    c1 = (n_viable_actions / float(k_actions)) if n_viable_actions is not None else 1.0
    c2 = phi_slack_multi(budgets or {}, mode=slack_mode)
    c3 = phi_irreversibility(ops, lam=lam)
    return {
        "rev": rev,
        "published": published,
        "ops": [o.as_dict() for o in ops],
        "c1_viable_actions": c1,
        "c2_slack": c2,
        "c3_irreversibility": c3,
        "phi_repo": c1 * c2 * c3,
    }


if __name__ == "__main__":
    demo = [
        "git push origin main",
        "git push --force origin main",
        "git push --force-with-lease origin main",
        "git commit --amend --no-edit",
        "rm -rf build/",
        "pytest -q",
        "psql -c 'ALTER TABLE users DROP COLUMN email'",
        "npm publish",
    ]
    for cmd in demo:
        for pub in (False, True):
            ops = classify_irreversibility(cmd, published=pub)
            print(f"{cmd:50s} published={int(pub)} "
                  f"phi3={phi_irreversibility(ops):.3f} "
                  f"{[o.name for o in ops]}")
