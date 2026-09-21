"""Environment protocol for fmcphi.

Extends the fmc-core protocol with the two predicates the Phi layer needs.

A user environment must implement:

  - actions()         -> Iterable[Hashable]    : the discrete action set A
  - clone_state(s)    -> S                     : deep copy / serialization of state
  - step(s, a)        -> S                     : apply action, return new state
  - observe(s)        -> np.ndarray            : observation used for distance
  - reward(s)         -> float                 : raw reward (pre-relativize)
  - sample_action(s, rng) -> a                 : scanning policy (default: uniform)

and, new in fmcphi:

  - viable(s)         -> bool                  : False iff s is an absorbing /
                                                 dead state from which no goal
                                                 is reachable any more
  - key(s)            -> Hashable              : identity of a state for the
                                                 purpose of counting distinct
                                                 reachable futures

`viable` and `key` are what make the causal cone measurable. An environment
where `viable` is always True has a cone that never narrows, and the Phi term
is constant by construction: see docs/SPEC.md section 5 (null-test trap).
"""

from __future__ import annotations

from typing import Hashable, Iterable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Environment(Protocol):
    def actions(self) -> Iterable[Hashable]:
        ...

    def clone_state(self, state):
        ...

    def step(self, state, action):
        ...

    def observe(self, state) -> np.ndarray:
        ...

    def reward(self, state) -> float:
        ...

    def sample_action(self, state, rng: np.random.Generator):
        ...

    def viable(self, state) -> bool:
        ...

    def key(self, state) -> Hashable:
        ...
