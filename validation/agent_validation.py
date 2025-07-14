"""
Agent validation for rewriting.
"""

from typing import Dict, Tuple, Optional, Any
import importlib
from pathlib import Path
from config.config_helper import get_flat_config
from .test_runner import run_performance_test

def should_force_rewrite(metrics: Dict[str, float]) -> bool:
    """Check if metrics are so bad that we should force an agent rewrite."""
    config = get_flat_config()
    window_size = config.get("metrics_window_size", 50)
    
    avg_key = f"avg_last_{window_size}"
    avg_return = metrics.get(avg_key, 0)
    success_rate = metrics.get("success_rate", 0)
    trap_hit_rate = metrics.get("trap_hit_rate", 0)
    
    # Force rewrite if:
    # 1. Very poor average return (much worse than random)
    # 2. Almost no success episodes
    # 3. Hitting traps constantly
    return (
        avg_return < -5.0 or                    # Very poor performance
        (success_rate < 0.05 and avg_return < -2.0) or  # Almost no success + poor performance
        trap_hit_rate > 0.8                     # Hitting traps 80%+ of the time
    )

def validate_agent_code(agentManager) -> bool:
    """Validate that the new agent code compiles and has required class."""
    try:
        spec = importlib.util.spec_from_file_location("agent", agentManager.agent_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Check if Agent class exists and has required methods
        if not hasattr(module, 'Agent'):
            print("❌ No Agent class found")
            return False
            
        agent_class = getattr(module, 'Agent')
        required_methods = ['select_action', 'learn', 'decay_epsilon']
        
        for method in required_methods:
            if not hasattr(agent_class, method):
                print(f"❌ Missing required method: {method}")
                return False
        
        # Test that demo_train runs successfully with new agent
        return test_demo_run()
        
    except Exception as e:
        print(f"❌ Code validation failed: {e}")
        return False
        
def test_demo_run() -> bool:
    """Test that the new agent works with a single demo run."""
    try:
        print("🧪 Testing new agent with single demo run...")
        
        # Use the unified test runner for a quick validation
        metrics = run_performance_test(
            test_config=None,  # Use current config
            agent_file=None,   # Use current agent
            num_episodes=1,    # Just one episode for validation
            test_name="Demo validation"
        )
        
        # Check if the test completed successfully (not error state)
        if metrics['avg_return'] > -999:
            print("✅ Single demo run test passed")
            return True
        else:
            print("❌ Single demo run test failed")
            return False
            
    except Exception as e:
        print(f"❌ Error running single demo run test: {e}")
        return False
        
def test_agent_performance(agent_file: Path, num_episodes: int, agent_name: str) -> Dict[str, float]:
    """Test an agent's performance over multiple episodes."""
    return run_performance_test(
        test_config=None,    # Use current config
        agent_file=agent_file,
        num_episodes=num_episodes,
        test_name=agent_name
    )