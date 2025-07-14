"""
Parameter validation rules for patches.
"""

from typing import Dict, Tuple, Optional, Any
from .test_runner import run_performance_test

# Parameter bounds: (min, max) where None means no bound
PARAMETER_BOUNDS = {
    "grid_size": (1, 10),
    "n_traps": (0, None),  # Max calculated dynamically
    "move_penalty": (-1.0, 0.0),
    "trap_penalty": (-10.0, 0.0),
    "goal_reward": (0.1, 10.0),
    "learning_rate": (0.01, 1.0),
    "gamma": (0.1, 0.99),
    "epsilon_start": (0.1, 1.0),
    "epsilon_min": (0.01, 0.5),
    "epsilon_decay": (0.9, 0.999),
    "episodes": (50, 1000),
    "max_steps_per_episode": (50, 500),
    "seed": (None, None)  # No bounds for seed
}

# Parameter to config section mapping
PARAMETER_CATEGORIES = {
    "grid_size": "environment",
    "n_traps": "environment",
    "move_penalty": "rewards",
    "trap_penalty": "rewards",
    "goal_reward": "rewards",
    "learning_rate": "agent",
    "gamma": "agent",
    "epsilon_start": "agent",
    "epsilon_min": "agent",
    "epsilon_decay": "agent",
    "episodes": "training",
    "max_steps_per_episode": "training",
    "seed": "system"
}

def validate_parameter(key: str, value: Any, config: Dict) -> Tuple[bool, str]:
    """Validate a single parameter value."""
    
    # Check if parameter exists
    if key not in PARAMETER_BOUNDS:
        return False, f"Unknown parameter '{key}'"
    
    # Get bounds
    min_val, max_val = PARAMETER_BOUNDS[key]
    
    # Check bounds
    if min_val is not None and value < min_val:
        return False, f"{key}={value} outside bounds [{min_val}, {max_val}]"
    
    if max_val is not None and value > max_val:
        return False, f"{key}={value} outside bounds [{min_val}, {max_val}]"
    
    # Special validation for n_traps
    if key == "n_traps":
        size = config["environment"]["grid_size"]
        max_traps = size
        if value > max_traps:
            return False, f"n_traps={value} too high for {size}x{size} grid (max {max_traps})"
    
    return True, ""

def get_parameter_category(key: str) -> Optional[str]:
    """Get the config section for a parameter."""
    return PARAMETER_CATEGORIES.get(key)
    
def test_config_performance(config: Dict, num_episodes: int, config_name: str) -> Dict[str, float]:
    """Test a config's performance by running training episodes."""
    return run_performance_test(
        test_config=config,
        agent_file=None,  # Use default agent
        num_episodes=num_episodes,
        test_name=config_name
    )