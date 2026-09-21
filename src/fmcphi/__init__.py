"""fmcphi -- the Phi layer for Fractal Monte Carlo.

Adds a causal-cone freedom factor to the FMC virtual reward:

    VR_i = Rhat_i^alpha * Dhat_i^beta * Phihat_i^gamma

fmcphi.core is a verbatim vendor of the fmc-core reference implementation
(MIT, uppifyagency/fractal-AI-FMC). Everything else is new.
"""

from fmcphi.core import (
    clone_step,
    decide,
    effective_branching_factor,
    effective_sample_size,
    relativize,
    virtual_reward,
)
from fmcphi.phi import (
    kill_dead,
    phi_composite,
    phi_cone,
    phi_slack,
    phi_viable_actions,
    virtual_reward_phi,
)
from fmcphi.planner import EpisodeResult, PlanResult, plan, run_episode

__version__ = "0.1.0"
__all__ = [
    "relativize", "virtual_reward", "clone_step", "decide",
    "effective_sample_size", "effective_branching_factor",
    "phi_cone", "phi_viable_actions", "phi_slack", "phi_composite",
    "virtual_reward_phi", "kill_dead",
    "plan", "run_episode", "PlanResult", "EpisodeResult",
]
