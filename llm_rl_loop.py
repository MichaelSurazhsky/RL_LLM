from __future__ import annotations
from statistics import mean
from typing import Dict, List
import numpy as np

from env import GridWorld                
from RL.agent import Agent       

from config.config_helper import get_config_singleton, get_flat_config

from llm_modifier import create_openai_client, build_prompt, query_llm, apply_multi_agent_rewrite, apply_multi_config_optimization
from validation.agent_validation import should_force_rewrite

# ---------- training & metrics ---------------------------------------------
def run_training(episodes: int) -> List[float]:
    """Run training and return episode returns."""
    
    config = get_flat_config()  # Get config when needed
    
    env = GridWorld(
        config["grid_size"],
        config["n_traps"],
    )
    agent = Agent(env)
    returns = []

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

    return returns

def summarise_returns(rets: List[float]) -> Dict[str, float]:
    """Calculate training metrics from episode returns."""
    # Use constant window size instead of config
    window_size = 50
    tail = rets[-window_size:] if len(rets) >= window_size else rets
    
    config = get_flat_config()  # Only get config for other values
    trap_episodes = sum(1 for r in tail if r < config.get("move_penalty", -0.05))
    
    return {
        f"avg_last_{window_size}": mean(tail),
        "avg_all": mean(rets),
        "min": min(rets),
        "max": max(rets),
        f"std_last_{window_size}": np.std(tail),
        "success_rate": sum(1 for r in tail if r > 0) / len(tail),
        "trap_hit_rate": trap_episodes / len(tail),
        "convergence_trend": mean(tail[-10:]) - mean(tail[-20:-10]) if len(tail) >= 20 else 0,
        "exploration_efficiency": mean(tail) / len(rets) if rets else 0,
        "episode_length_trend": "increasing" if len(tail) > 10 and mean([len(str(r)) for r in tail[-5:]]) > mean([len(str(r)) for r in tail[-10:-5]]) else "stable"
    }
# ---------- main loop -------------------------------------------------------
def main() -> None:
    """Main training loop with LLM guidance."""
    client = create_openai_client()
    config = get_config_singleton()  # Use nested config for accessing sections
    
    max_cycles = config["system"]["max_cycles"]
    episodes_per_cycle = config["training"]["episodes_per_cycle"]

    performance_history = []

    for cycle in range(1, max_cycles + 1):
        print(f"\n=== Cycle {cycle}/{max_cycles} — {episodes_per_cycle} episodes ===")
        
        returns = run_training(episodes_per_cycle)
        metrics = summarise_returns(returns)
        performance_history.append(metrics)
        print("Metrics:", metrics)

        decision = query_llm(client, build_prompt(metrics))

        if decision.get("stop") is True:
            print("🏆  LLM signalled convergence — stopping.")
            break

        elif decision.get("rewrite_agent") is True or should_force_rewrite(metrics):
            reason = decision.get("reason", "LLM decided rewrite needed")
            print(f"🤖 LLM wants to rewrite agent because: {reason}")
            agent_rewritten = apply_multi_agent_rewrite(metrics, num_agents=3)
            if agent_rewritten:
                print("🔄 Agent was rewritten - performance may change significantly")
                performance_history = performance_history[-2:]

        elif decision.get("optimize_agent_configs") is True:
            reason = decision.get("reason", "LLM decided to optimize agent configs")
            print(f"🔧 LLM wants to optimize agent configs because: {reason}")
            config_optimized = apply_multi_config_optimization(metrics, focus_area="agent", num_variants=3)
            if config_optimized:
                print("🔄 Agent config was optimized - performance may improve")

        elif decision.get("optimize_training_configs") is True:
            reason = decision.get("reason", "LLM decided to optimize training configs")
            print(f"🔧 LLM wants to optimize training configs because: {reason}")
            config_optimized = apply_multi_config_optimization(metrics, focus_area="training", num_variants=3)
            if config_optimized:
                print("🔄 Training config was optimized - performance may improve")

    print("\n🏁 Finished.")

if __name__ == "__main__":
    main()