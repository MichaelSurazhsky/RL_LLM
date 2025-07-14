from __future__ import annotations
import random
from typing import Tuple, List, Dict

from config.config_helper import get_flat_config

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class GridWorld:
    """Simple stochastic grid with random start, goal and trap cells."""

    def __init__(self, size: int, n_traps: int) -> None:
        self.rows, self.cols = size, size
        self.n_traps = n_traps
        self.goal: Tuple[int, int] | None = None
        self.traps: List[Tuple[int, int]] = []
        self.state: Tuple[int, int] | None = None

        # Initialize actions - always include sense action
        self.ACTIONS = {
            0: (-1,  0),  # up
            1: ( 1,  0),  # down
            2: ( 0, -1),  # left
            3: ( 0,  1),  # right
            4: "sense"    # always available
        }

    # ---- Public API --------------------------------------------------------

    def reset(self) -> Tuple[int, int]:
        """Sample a fresh layout and return the start state."""
        self._sample_layout()
        self.state = self._random_unoccupied_cell()
        return self.state

    def step(self, action: int):
        """Apply one action; return (next_state, reward, done, info)."""
        config = get_flat_config()

        if action == 4:  # sense action
            # Reveal adjacent trap information
            adjacent_traps = self._get_adjacent_traps(self.state)
            reward = config.get("sense_penalty", -0.1)
            return self.state, reward, False, {"adjacent_traps": adjacent_traps}

        dr, dc = self.ACTIONS[action]
        r, c = self.state
        nr = max(0, min(self.rows - 1, r + dr))
        nc = max(0, min(self.cols - 1, c + dc))
        self.state = (nr, nc)

        reward = config["move_penalty"]
        done = False
        if self.state == self.goal:
            reward += config["goal_reward"]
            done = True
        elif self.state in self.traps:
            reward += config["trap_penalty"]
            # episode continues after falling in a trap

        return self.state, reward, done, {}

    # ---- Utility -----------------------------------------------------------

    def state_shape(self) -> Tuple[int, int]:
        return self.rows, self.cols

    def n_actions(self) -> int:
        return len(self.ACTIONS)
    
    # Add to env.py after the existing private helper methods
    def _get_adjacent_traps(self, state: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get list of adjacent cells that contain traps."""
        if state is None:
            return []
    
        r, c = state
        adjacent_traps = []
    
        # Check all 4 adjacent directions
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            # Check if adjacent cell is within bounds and contains a trap
            if (0 <= nr < self.rows and 0 <= nc < self.cols and 
                (nr, nc) in self.traps):
                adjacent_traps.append((nr, nc))
    
        return adjacent_traps

    # ---- Private helpers ---------------------------------------------------

    def _sample_layout(self) -> None:
        """Pick new goal + trap cells (all distinct)."""
        occupied = set()
        self.goal = self._random_cell_excluding(occupied)
        occupied.add(self.goal)

        self.traps = []
        for _ in range(self.n_traps):
            cell = self._random_cell_excluding(occupied)
            self.traps.append(cell)
            occupied.add(cell)

    def _random_unoccupied_cell(self) -> Tuple[int, int]:
        occupied = {self.goal, *self.traps}
        return self._random_cell_excluding(occupied)

    def _random_cell_excluding(self, excluded: set) -> Tuple[int, int]:
        while True:
            cell = (random.randrange(self.rows), random.randrange(self.cols))
            if cell not in excluded:
                return cell
