from __future__ import annotations
import numpy as np
import random
from typing import Tuple, List, Dict

from config.config_helper import get_flat_config
from env import GridWorld
from RL.agent import Agent


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(env: GridWorld, agent: Agent, episodes: int) -> List[float]:
    """Run many episodes and return per-episode returns."""
    config = get_flat_config()

    returns: List[float] = []
    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_return = 0.0
        steps = 0

        while not done and steps < config["max_steps_per_episode"]:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)

            state = next_state
            ep_return += reward
            steps += 1

        agent.decay_epsilon()
        returns.append(ep_return)

        if (ep + 1) % 50 == 0:
            avg_ret = np.mean(returns[-50:])
            print(
                f"Episode {ep + 1:4d} | "
                f"Avg return (last 50): {avg_ret:+.3f} | "
                f"ε = {agent.epsilon:.2f}"
            )

    return returns

# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    config = get_flat_config()
    random.seed(config["seed"])
    np.random.seed(config["seed"])

    env = GridWorld(
        config["grid_size"],
        config["n_traps"],
    )
    agent = Agent(env)
    train(env, agent, config["episodes"])

if __name__ == "__main__":
    main()