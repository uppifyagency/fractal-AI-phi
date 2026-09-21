"""TrapGrid: a 2D cliff-walk with absorbing traps and a finite fuel budget.

This is the environment for E1, E4 and E5. Its only job is to make the causal
cone *actually narrow*, which is the precondition the Phi layer needs in order
to be measurable at all (docs/SPEC.md section 5).

Layout (default 12 x 4, '.' free, 'X' trap, 'S' start, 'G' goal):

    y=3  . . . . . . . . . . . .
    y=2  . . . . . . . . . . . .
    y=1  . . . . . . . . . . . .
    y=0  S X X X X X X X X X X G

The short route (straight along y=0) is 11 steps and passes through the trap
row. The safe route (up, across, down) is 13 steps. With a fuel budget between
the two, neither pure goal-seeking nor pure caution wins by itself, which is
the tension the alpha/gamma ratio is supposed to arbitrate.

Both routes are reachable from the start, and the reward is a smooth function of
distance to the goal everywhere, so the swarm always has a gradient to follow:
the flat-negative failure mode documented in the reference repo's THEORY.md
section 4.3.1 does not apply here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Tuple

import numpy as np


# Action labels are integer indices so that the vendored fmcphi.core.decide
# (which casts labels to int) works unchanged. DELTAS holds the geometry.
ACTIONS: Tuple[int, ...] = (0, 1, 2, 3, 4)
# up, down, left, right, stay
DELTAS: Tuple[Tuple[int, int], ...] = ((0, 1), (0, -1), (-1, 0), (1, 0), (0, 0))


@dataclass
class State:
    x: int
    y: int
    fuel: float


class TrapGrid:
    """Cliff-walk with fuel. Deterministic transitions, walls clamp."""

    def __init__(
        self,
        width: int = 12,
        height: int = 4,
        fuel: float = 16.0,
        trap_row: int = 0,
        goal: Tuple[int, int] | None = None,
        start: Tuple[int, int] = (0, 0),
    ):
        self.width = width
        self.height = height
        self.fuel0 = fuel
        self.trap_row = trap_row
        self.start = start
        self.goal = goal if goal is not None else (width - 1, trap_row)

    # -- construction --------------------------------------------------------

    def reset(self) -> State:
        return State(x=self.start[0], y=self.start[1], fuel=self.fuel0)

    def is_trap(self, x: int, y: int) -> bool:
        if y != self.trap_row:
            return False
        return 0 < x < self.width - 1

    def is_goal(self, state: State) -> bool:
        return (state.x, state.y) == self.goal

    # -- Environment protocol ------------------------------------------------

    def actions(self):
        return ACTIONS

    def clone_state(self, state: State) -> State:
        return State(x=state.x, y=state.y, fuel=state.fuel)

    def step(self, state: State, action: int) -> State:
        dx, dy = DELTAS[action]
        x = max(0, min(self.width - 1, state.x + dx))
        y = max(0, min(self.height - 1, state.y + dy))
        moved = (x, y) != (state.x, state.y)
        return State(x=x, y=y, fuel=state.fuel - (1.0 if moved else 0.5))

    def observe(self, state: State) -> np.ndarray:
        return np.array([state.x, state.y], dtype=np.float64)

    def reward(self, state: State) -> float:
        """Negative Manhattan distance to the goal. Smooth, never flat."""
        return -float(abs(state.x - self.goal[0]) + abs(state.y - self.goal[1]))

    def sample_action(self, state: State, rng: np.random.Generator) -> int:
        return int(rng.integers(0, len(ACTIONS)))

    def death_cause(self, state: State) -> str | None:
        """Why this state is dead, or None if it is alive.

        Trap death and fuel exhaustion are reported separately because the Phi
        layer is expected to trade one for the other: keeping options open costs
        steps, and steps cost fuel.
        """
        if self.is_trap(state.x, state.y):
            return "trap"
        if state.fuel <= 0.0:
            return "fuel"
        return None

    def viable(self, state: State) -> bool:
        if state.fuel <= 0.0:
            return False
        return not self.is_trap(state.x, state.y)

    def key(self, state: State) -> Hashable:
        return (state.x, state.y)
