"""
tuning_prompt.py
~~~~~~~~~~~~~~~~
Main prompt template for LLM-guided parameter tuning.
"""

from typing import Dict
import json

# ---------- Main Prompt Building Functions ----------

def build_tuning_prompt(metrics: Dict[str, float], flat_config: Dict, param_descriptions: Dict[str, str]) -> str:
    """Build the main tuning prompt for the LLM."""
    
    return f"""You are an RL expert tuning a grid-world Q-learning agent.

{_format_metrics_section(metrics)}

CURRENT CONFIG:
```json
{_format_json(flat_config)}
```

TUNABLE PARAMETERS:
{_format_parameters(param_descriptions)}

{_get_analysis_guidelines("tuning")}

Choose your response:
1. To test rewrite agent configs, reply: {{"optimize_agent_configs": true, "reason": "your reason here"}}
2. To test rewrite training configs, reply: {{"optimize_training_configs": true, "reason": "your reason here"}}
3. To rewrite the agent, reply: {{"rewrite_agent": true, "reason": "your reason here"}}
5. If performance is satisfactory, reply: {{"stop": true}}

Use parameter names as shown in CURRENT CONFIG. Reply with JSON only, no explanations."""

def build_agent_rewrite_prompt(config, metrics: Dict[str, float], current_agent_code: str) -> str:
    """Build prompt for agent code rewriting."""
    
    # Get specific improvement suggestions based on metrics
    improvement_suggestions = get_agent_improvement_suggestions(metrics)
    
    return f"""You are an RL expert. Rewrite the Agent class to improve performance.

{_format_metrics_section(metrics)}

CURRENT AGENT CODE:
```python
{current_agent_code}
```

CURRENT CONFIG:
```json
{_format_json(config)}
```

SPECIFIC IMPROVEMENT SUGGESTIONS:
{improvement_suggestions}

{_get_analysis_guidelines("agent")}

REQUIREMENTS:
1. Keep the same class name "Agent" and constructor signature
2. Keep methods: select_action(state), learn(s, a, r, s_next, done), decay_epsilon()
3. Use imports: numpy as np, random, typing, from config.config_helper import get_flat_config, from env import GridWorld
4. Return complete, runnable Python code with proper imports

Respond with the complete rewritten agent.py file content:"""

def build_multi_agent_prompt(config, metrics: Dict[str, float], current_agent_code: str, num_agents: int) -> str:
    """Build prompt for generating multiple agent candidates."""
    
    # Get the base single-agent prompt
    base_prompt = build_agent_rewrite_prompt(config, metrics, current_agent_code)
    
    # Replace the final instruction with multi-agent instruction
    multi_agent_instruction = f"""

{_get_strategy_descriptions("agent", num_agents)}

ADDITIONAL REQUIREMENTS FOR MULTI-AGENT:
• Make each agent use a DIFFERENT approach to improvement
• Each agent should implement a distinct strategy from the suggestions above

Format your response as:

# AGENT 1
```python
[complete agent code]
```

# AGENT 2  
```python
[complete agent code]
```

# AGENT 3
```python
[complete agent code]
```"""
    
    # Replace the last line of the base prompt
    lines = base_prompt.split('\n')
    lines[-1] = multi_agent_instruction
    
    return '\n'.join(lines)

def build_multi_config_prompt(config, metrics: Dict[str, float], focus_area: str, num_variants: int = 3) -> str:
    """Build prompt for generating multiple config variants."""
    
    if focus_area == "agent":
        focus_description = "agent learning parameters (learning_rate, gamma, epsilon_start, epsilon_min, epsilon_decay)"
        strategy_description = """
• Variant 1: Conservative changes (small learning rate adjustments, moderate exploration)
• Variant 2: Aggressive learning (higher learning rate, faster epsilon decay)
• Variant 3: Exploration-focused (longer exploration phase, different epsilon strategy)"""
        
    elif focus_area == "training":
        focus_description = "training parameters (episodes, max_steps_per_episode)"
        strategy_description = """
• Variant 1: More episodes for better convergence
• Variant 2: Longer episodes for complex exploration
• Variant 3: Balanced approach with optimal episode length"""
        
    else:
        focus_description = "general parameters"
        strategy_description = _get_strategy_descriptions("config", num_variants)
    
    current_params = _get_relevant_params(config, focus_area)
    
    return f"""You are an RL expert. Generate {num_variants} different parameter configurations to improve agent performance.

{_format_metrics_section(metrics)}

CURRENT RELEVANT PARAMETERS:
```json
{_format_json(current_params)}
```

FOCUS AREA: {focus_description}

GENERATE {num_variants} DIFFERENT CONFIGURATIONS with these strategies:
{strategy_description}

{_get_analysis_guidelines("config")}

REQUIREMENTS:
1. Only modify parameters in the focus area: {focus_area}
2. Keep parameter values within reasonable bounds
3. Make each variant use a DIFFERENT strategy
4. Provide reasoning for each change

Format your response as:

# VARIANT 1
```json
{{"learning_rate": 0.3, "gamma": 0.95}}
```
Reasoning: [brief explanation]

# VARIANT 2
```json
{{"learning_rate": 0.8, "epsilon_decay": 0.98}}
```
Reasoning: [brief explanation]

# VARIANT 3
```json
{{"epsilon_start": 1.0, "epsilon_min": 0.01}}
```
Reasoning: [brief explanation]"""

# ---------- Helper Functions (moved to top) ----------

def _format_json(data: Dict) -> str:
    """Format dictionary as JSON with proper indentation."""
    return json.dumps(data, indent=2)

def _format_metrics_section(metrics: Dict[str, float]) -> str:
    """Format metrics section consistently."""
    return f"""CURRENT PERFORMANCE METRICS:
```json
{_format_json(metrics)}
```"""

def _format_parameters(param_descriptions: Dict[str, str]) -> str:
    """Format parameter descriptions as bullet points."""
    return "\n".join(f"• {k}: {v}" for k, v in param_descriptions.items())

def _get_analysis_guidelines(context: str = "general") -> str:
    """Get analysis guidelines based on context."""
    if context == "config":
        return """ANALYSIS GUIDELINES:
• If avg_last_50 < -2.0: Try higher learning rates, longer exploration
• If success_rate < 0.1: Increase exploration time, reduce epsilon decay
• If learning is unstable: Reduce learning rate, increase episodes
• If convergence is slow: Increase learning rate, optimize epsilon schedule"""
    
    elif context == "agent":
        return """GENERAL ANALYSIS GUIDELINES:
• If avg_last_50 < 0: Try different exploration strategies, learning methods
• If learning is unstable: Add experience replay, target networks, or better exploration
• If convergence is slow: Improve Q-learning updates, add prioritized experience replay
• If getting stuck in local optima: Add curiosity-driven exploration, count-based bonuses"""
    
    else:  # general/tuning
        return """ANALYSIS GUIDELINES:
• avg_last_50 > 0.5: Good performance, small tweaks or stop
• avg_last_50 around 0: Mediocre, try learning rate or exploration changes
• avg_last_50 < -0.5: Poor performance, consider different params
• avg_last_50 < -5.0: Very Poor performance, consider rewriting agent
• Flat learning curve: Increase learning_rate or change epsilon decay
• Unstable learning: Decrease learning_rate or increase episodes"""

def _get_strategy_descriptions(strategy_type: str, num_variants: int) -> str:
    """Get strategy descriptions for different contexts."""
    if strategy_type == "agent":
        return f"""GENERATE {num_variants} DIFFERENT AGENTS with these strategies:
• Agent 1: Conservative improvements (better exploration, learning rate adjustments)
• Agent 2: Moderate changes (experience replay, different exploration strategies) 
• Agent 3: Advanced techniques (curiosity-driven exploration, count-based bonuses, UCB)"""
    
    elif strategy_type == "config":
        return f"""GENERATE {num_variants} DIFFERENT CONFIGURATIONS with these strategies:
• Variant 1: Conservative improvements
• Variant 2: Moderate changes
• Variant 3: Aggressive optimization"""
    
    return ""

def _get_relevant_params(config: Dict, focus_area: str) -> Dict:
    """Extract relevant parameters based on focus area."""
    if focus_area == "agent":
        return config.get("agent", {})
    elif focus_area == "training":
        return config.get("training", {})
    elif focus_area == "rewards":
        return config.get("rewards", {})
    else:
        # Return flattened view
        result = {}
        for section in ["agent", "training", "rewards"]:
            result.update(config.get(section, {}))
        return result

def get_agent_improvement_suggestions(metrics: Dict[str, float]) -> str:
    """Get specific suggestions for agent improvements based on metrics."""
    
    suggestions = []
    
    avg_return = metrics.get("avg_last_50", 0)
    success_rate = metrics.get("success_rate", 0)
    exploration_eff = metrics.get("exploration_efficiency", 0)
    
    if avg_return < -1.0:
        suggestions.append("Consider Double Q-Learning to reduce overestimation bias")
        suggestions.append("Add experience replay buffer for better sample efficiency")
        
    if success_rate < 0.1:
        suggestions.append("Implement curiosity-driven exploration (count-based bonuses)")
        suggestions.append("Try Upper Confidence Bound (UCB) exploration instead of ε-greedy")
        suggestions.append("Add 'Sense' action to gather environment information before moving")
        
    if exploration_eff < 0.01:
        suggestions.append("Add intrinsic motivation or exploration bonuses")
        suggestions.append("Consider Boltzmann exploration for better action selection")
        suggestions.append("Implement 'Sense' action to improve state awareness and exploration")
        
    return "\n".join(f"• {s}" for s in suggestions)
