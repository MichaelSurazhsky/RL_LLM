"""
param_descriptions.py
~~~~~~~~~~~~~~~~~~~~~
Parameter descriptions for the LLM tuning prompt.
"""

from typing import Dict

def get_param_descriptions(config: Dict) -> Dict[str, str]:
    """Get parameter descriptions with dynamic bounds based on current config."""
    
    grid_size = config["environment"]["grid_size"]
    max_traps = grid_size
    
    return {
        "grid_rows": "Grid height (1-20) - larger = harder",
        "grid_cols": "Grid width (1-20) - larger = harder", 
        "n_traps": f"Number of trap cells (0 to {max_traps})",
        "move_penalty": "Penalty per step (-1.0 to 0.0) - more negative = faster episodes",
        "trap_penalty": "Penalty for hitting trap (-10.0 to 0.0) - helps avoid traps",
        "goal_reward": "Reward for reaching goal (0.1 to 10.0) - motivates success",
        "learning_rate": "Q-learning rate (0.01 to 1.0) - higher = faster learning but less stable",
        "gamma": "Discount factor (0.1 to 0.99) - higher = more long-term planning",
        "epsilon_start": "Initial exploration rate (0.1 to 1.0) - start exploring",
        "epsilon_min": "Minimum exploration rate (0.01 to 0.5) - maintain some exploration", 
        "epsilon_decay": "Exploration decay rate (0.9 to 0.999) - how fast to reduce exploration",
        "episodes": "Episodes per cycle (50 to 1000) - more = better learning but slower",
        "max_steps_per_episode": "Max steps per episode (50 to 500) - prevents infinite loops",
        "seed": "Random seed (any integer) - change for different layouts"
    }