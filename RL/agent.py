from __future__ import annotations
import numpy as np
import random
from typing import Tuple, List, Dict

from config.config_helper import get_flat_config
from env import GridWorld

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """Tabular Q-learning agent with ε-greedy exploration."""

    def __init__(self, env: GridWorld) -> None:
        config = get_flat_config()  # Get flat config once
        
        # Cache config values as instance attributes
        self.alpha = config["learning_rate"]  # or config["alpha"] if you prefer
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon_start"]
        self.epsilon_min = config["epsilon_min"]
        self.epsilon_decay = config["epsilon_decay"]
        
        # Initialize Q-table
        r, c = env.state_shape()
        self.Q = np.zeros((r, c, env.n_actions()), dtype=np.float32)

    def select_action(self, state: Tuple[int, int]) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.Q.shape[-1])
        r, c = state
        return int(np.argmax(self.Q[r, c]))

    def learn(
        self,
        s: Tuple[int, int],
        a: int,
        r: float,
        s_next: Tuple[int, int],
        done: bool,
    ) -> None:
        r1, c1 = s
        r2, c2 = s_next
        best_next = np.max(self.Q[r2, c2]) if not done else 0.0
        td_target = r + self.gamma * best_next
        td_error = td_target - self.Q[r1, c1, a]
        self.Q[r1, c1, a] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)