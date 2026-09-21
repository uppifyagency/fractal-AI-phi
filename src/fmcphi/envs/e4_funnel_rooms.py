"""FunnelRooms: the second E4 map. Narrowing chambers, scattered pits, fuel.

Owned by experiments/E4_no_goal. Deliberately *not* registered in any shared
__init__: import it directly with

    from fmcphi.envs.e4_funnel_rooms import FunnelRooms

Why a second map
----------------
`TrapGrid` is a cliff-walk: one lethal row, one safe corridor, goal in the same
corner as the lethal row. Any E4 result on it alone is exposed to the objection
that the geometry itself encodes the goal direction. `FunnelRooms` changes every
one of those properties while keeping the two that docs/SPEC.md section 5 says
are non-negotiable, irreversibility and a finite budget:

  * free space *narrows monotonically* toward the goal (38 -> 10 -> 3 -> 1
    cells), so the goal sits at the bottom of a causal-cone funnel instead of
    beside a cliff;
  * hazards are scattered pits, not a contiguous row;
  * walls are impassable rather than clamping, so a wall is a lost option and
    not a free "stay";
  * the goal is a dead end: from the goal cell exactly one neighbour is free.

The prediction that motivates the map: a gamma-driven agent with alpha = 0
should *refuse* to enter the funnel, because Phi falls at every doorway. If it
enters anyway, the geometry is leaking reward information and E4-P3 is dead.

Layout (15 x 7, y increasing upward, '.' free, '#' wall, 'X' pit,
'S' start, 'G' goal)::

    y=6  . . . . . . # # # # # # # # #
    y=5  . . X . . . # # # # # # # # #
    y=4  . . . X . . # . . X . # # # #
    y=3  S . . . . . . . . . . . . . G
    y=2  . . . X . . # . . X . # # # #
    y=1  . . X . . . # # # # # # # # #
    y=0  . . . . . . # # # # # # # # #

Room A is x in [0, 5], all y: 42 cells minus 4 pits = 38 free, genuinely open.
Doorway at (6, 3). Room B is x in [7, 10], y in [2, 4]: 12 cells minus 2 pits
= 10 free. Doorway at (11, 3). Corridor C is (12, 3), (13, 3), (14, 3) with the
goal at (14, 3).

The straight route along y = 3 is 14 steps and is pit-free, but it threads
between the pit pairs at x = 3 and x = 9, so a random walker dies there with
probability 2/5 per step. The default fuel of 20 makes the route feasible and
wandering fatal: short lethal-adjacent route, long safe loitering, same tension
as TrapGrid but arranged differently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Tuple

import numpy as np

# Same integer action labels as TrapGrid, so fmcphi.core.decide works unchanged.
ACTIONS: Tuple[int, ...] = (0, 1, 2, 3, 4)
# up, down, left, right, stay
DELTAS: Tuple[Tuple[int, int], ...] = ((0, 1), (0, -1), (-1, 0), (1, 0), (0, 0))

# Rows are written top (y = height-1) first, as they appear in the docstring.
MAP_ROWS: Tuple[str, ...] = (
    "......#########",
    "..X...#########",
    "...X..#..X.####",
    "S.............G",
    "...X..#..X.####",
    "..X...#########",
    "......#########",
)


@dataclass
class State:
    x: int
    y: int
    fuel: float


class FunnelRooms:
    """Narrowing chambers with absorbing pits and a finite fuel budget.

    Transitions are deterministic. A move into a wall or off the board is
    refused: the agent stays put and pays the idle fuel cost, exactly as if it
    had chosen "stay". A move onto a pit succeeds and is absorbing: `step` on a
    non-viable state returns that state unchanged, so death is a fixed point of
    the dynamics and not merely a flag read by `viable`.
    """

    def __init__(
        self,
        fuel: float = 20.0,
        rows: Tuple[str, ...] = MAP_ROWS,
        move_cost: float = 1.0,
        idle_cost: float = 0.5,
    ):
        self.rows = tuple(rows)
        self.height = len(self.rows)
        self.width = len(self.rows[0])
        if any(len(r) != self.width for r in self.rows):
            raise ValueError("FunnelRooms map rows must all have the same width")
        self.fuel0 = float(fuel)
        self.move_cost = float(move_cost)
        self.idle_cost = float(idle_cost)

        self.walls = set()
        self.pits = set()
        start = goal = None
        for j, row in enumerate(self.rows):
            y = self.height - 1 - j
            for x, ch in enumerate(row):
                if ch == "#":
                    self.walls.add((x, y))
                elif ch == "X":
                    self.pits.add((x, y))
                elif ch == "S":
                    start = (x, y)
                elif ch == "G":
                    goal = (x, y)
        if start is None or goal is None:
            raise ValueError("FunnelRooms map needs exactly one 'S' and one 'G'")
        self.start = start
        self.goal = goal

    # -- construction --------------------------------------------------------

    def reset(self) -> State:
        return State(x=self.start[0], y=self.start[1], fuel=self.fuel0)

    def is_wall(self, x: int, y: int) -> bool:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return True
        return (x, y) in self.walls

    def is_pit(self, x: int, y: int) -> bool:
        return (x, y) in self.pits

    def is_goal(self, state: State) -> bool:
        return (state.x, state.y) == self.goal

    # -- Environment protocol ------------------------------------------------

    def actions(self):
        return ACTIONS

    def clone_state(self, state: State) -> State:
        return State(x=state.x, y=state.y, fuel=state.fuel)

    def step(self, state: State, action: int) -> State:
        # Death is ABSORBING. Without this guard a non-viable state (pit or
        # fuel-exhausted) transitions normally, so a swarm rolled out inside
        # plan() simulates straight through its own death and the arms that
        # never test viability (gamma = 0) are scored on futures that do not
        # exist. docs/SPEC.md section 5 names irreversibility as a
        # precondition for this layer to be measurable at all; the guard is
        # what makes the claim in this class's docstring true.
        if not self.viable(state):
            return self.clone_state(state)
        dx, dy = DELTAS[action]
        nx, ny = state.x + dx, state.y + dy
        if self.is_wall(nx, ny) or (dx == 0 and dy == 0):
            return State(x=state.x, y=state.y, fuel=state.fuel - self.idle_cost)
        return State(x=nx, y=ny, fuel=state.fuel - self.move_cost)

    def observe(self, state: State) -> np.ndarray:
        return np.array([state.x, state.y], dtype=np.float64)

    def reward(self, state: State) -> float:
        """Negative Manhattan distance to the goal. Smooth, never flat."""
        return -float(abs(state.x - self.goal[0]) + abs(state.y - self.goal[1]))

    def sample_action(self, state: State, rng: np.random.Generator) -> int:
        return int(rng.integers(0, len(ACTIONS)))

    def death_cause(self, state: State) -> str | None:
        if self.is_pit(state.x, state.y):
            return "pit"
        if state.fuel <= 0.0:
            return "fuel"
        return None

    def viable(self, state: State) -> bool:
        if state.fuel <= 0.0:
            return False
        return not self.is_pit(state.x, state.y)

    def key(self, state: State) -> Hashable:
        return (state.x, state.y)

    # -- map diagnostics (not part of the protocol) --------------------------

    def region(self, x: int, y: int) -> str:
        """Which chamber a cell belongs to: A (open), B, C (corridor), goal."""
        if (x, y) == self.goal:
            return "goal"
        if x <= 5:
            return "A"
        if x == 6:
            return "door_AB"
        if x <= 10:
            return "B"
        if x == 11:
            return "door_BC"
        return "C"

    def free_cells(self):
        return [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if not self.is_wall(x, y) and not self.is_pit(x, y)
        ]
