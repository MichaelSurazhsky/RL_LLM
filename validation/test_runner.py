
"""
test_runner.py
~~~~~~~~~~~~~~
Unified testing infrastructure for configs and agents.
"""

import sys
import json
import tempfile
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any

def run_performance_test(
    test_config: Optional[Dict] = None,
    agent_file: Optional[Path] = None,
    num_episodes: int = 100,
    test_name: str = "Test"
) -> Dict[str, float]:
    """
    Universal performance test runner for configs and agents.
    
    Args:
        test_config: Config dict to test (if None, uses current config)
        agent_file: Path to agent file (if None, uses default RL/agent.py)
        num_episodes: Number of episodes to run
        test_name: Name for logging
        
    Returns:
        Dict with metrics: avg_return, success_rate, test_name
    """
    
    # Generate the test script
    test_script = _generate_test_script(test_config, agent_file, num_episodes)
    
    # Execute the test
    return _execute_test_script(test_script, test_name)

def _generate_test_script(
    test_config: Optional[Dict],
    agent_file: Optional[Path], 
    num_episodes: int
) -> str:
    """Generate the test script content."""
    
    # Determine agent import strategy
    if agent_file:
        agent_import = f'''
# Import specific agent file
import importlib.util
spec = importlib.util.spec_from_file_location("test_agent", "{agent_file}")
test_agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_agent_module)
Agent = test_agent_module.Agent
'''
    else:
        agent_import = '''
# Import default agent
from RL.agent import Agent
'''
    
    # Determine config strategy
    if test_config:
        config_setup = f'''
# Use provided test config
test_config = {json.dumps(test_config)}
flat_config = {{}}
flat_config.update(test_config["environment"])
flat_config.update(test_config["rewards"])
flat_config.update(test_config["agent"])
flat_config.update(test_config["training"])
flat_config.update(test_config["system"])
'''
    else:
        config_setup = '''
# Use current config
from config.config_helper import get_flat_config
flat_config = get_flat_config()
'''
    
    return f'''
import sys
import os
import numpy as np
import random
from pathlib import Path

# Add the project directory to sys.path
project_dir = "{Path.cwd()}"
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

{config_setup}

{agent_import}

from env import GridWorld

try:
    # Set seeds for reproducible test
    random.seed(42)
    np.random.seed(42)
    
    # Create environment and agent
    env = GridWorld(flat_config["grid_size"], flat_config["n_traps"])
    agent = Agent(env)
    
    returns = []
    
    # Run test episodes
    for ep in range({num_episodes}):
        state = env.reset()
        done = False
        ep_return = 0.0
        steps = 0
        
        while not done and steps < flat_config["max_steps_per_episode"]:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            ep_return += reward
            steps += 1
        
        agent.decay_epsilon()
        returns.append(ep_return)
    
    # Calculate metrics
    avg_return = float(np.mean(returns))
    success_rate = float(sum(1 for r in returns if r > 0) / len(returns))
    
    print(f"RESULTS: avg_return={{avg_return:.3f}}, success_rate={{success_rate:.3f}}")
    
except Exception as e:
    print(f"ERROR: {{e}}")
    import traceback
    traceback.print_exc()
    print("RESULTS: avg_return=-999.000, success_rate=0.000")
    sys.exit(1)
'''

def _execute_test_script(test_script: str, test_name: str) -> Dict[str, float]:
    """Execute the test script and parse results."""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        test_file = f.name
    
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=Path.cwd()
        )
        
        # Parse results
        avg_return, success_rate = _parse_test_output(result.stdout)
        
        return {
            'avg_return': avg_return,
            'success_rate': success_rate,
            'test_name': test_name
        }
        
    except subprocess.TimeoutExpired:
        print(f"❌ {test_name} test timed out")
        return {'avg_return': -999.0, 'success_rate': 0.0, 'test_name': test_name}
    except Exception as e:
        print(f"❌ Error running {test_name} test: {e}")
        return {'avg_return': -999.0, 'success_rate': 0.0, 'test_name': test_name}
    finally:
        Path(test_file).unlink(missing_ok=True)

def _parse_test_output(stdout: str) -> tuple[float, float]:
    """Parse avg_return and success_rate from test output."""
    avg_return = -999.0
    success_rate = 0.0
    
    for line in stdout.split('\n'):
        if line.startswith('RESULTS:'):
            try:
                parts = line.split('RESULTS: ')[1].split(', ')
                avg_return = float(parts[0].split('=')[1])
                success_rate = float(parts[1].split('=')[1])
                break
            except (IndexError, ValueError):
                pass
    
    return avg_return, success_rate