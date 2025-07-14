# Text-Based Automated RL-LLM Loop Overview

This project implements an automated pipeline where an LLM observes the performance of an RL agent in a grid-world environment and makes intelligent modifications to improve learning. The system can tune hyperparameters, modify reward structures, and even rewrite the agent's learning algorithm entirely.

## Core Components

### 🌍 Grid World Environment (`env.py`)

Configurable grid with Random start position, goal state, multiple trap states
- 4 movement actions (up, down, left, right) + sense action
- **Rewards**: Goal reached, Trap hit (negative), Movement (negative), Sensing (negative)

### 🤖 RL Agent (`RL/agent.py`)
Start with a very basic Tabular Q-learning with ε-greedy exploration

### 🧠 LLM Integration (`llm_modifier.py`)
- **Capabilities**:
  - Analyzes training metrics and suggests improvements
  - Generates multiple config/agent variants for testing
  - Rewrites agent code with advanced RL techniques
  - Makes data-driven decisions based on performance

### ⚙️ Configuration System (`config/`)
- JSON-based Structured configuration with validation of changes
- **Categories**: Environment, rewards, agent, training, system parameters (including the llm model)  
- **Version Control**: Automatic backup and rollback of configurations
- **Multi-variant Testing**: Compare multiple parameter sets simultaneously

## Quick Start

### 1. Setup
```bash
# Install dependencies
pip install openai numpy

# Add your OpenAI API key
echo "your-api-key-here" > openai_key.txt
```

### 2. Basic Training
```bash
# Run single training session for training demo
python demo_train.py

# Run LLM-guided optimization loop
python llm_rl_loop.py
```

### 3. Configuration
Edit `config/config.json` to modify manually:
- Grid size and trap count
- Reward values and penalties  
- Learning parameters (learning rate, exploration)
- Training episodes and optimization cycles
- LLM model

## How It Works

### The Optimization Loop

1. **Train Agent**: Run RL training for specified episodes
2. **Collect Metrics**: Analyze performance (success rate, average return, convergence)
3. **LLM Analysis**: Send metrics to LLM for intelligent analysis
4. **Make Changes**: LLM decides to:
   - Tune hyperparameters (learning rate, exploration, rewards)
   - Test multiple parameter variants
   - Rewrite the agent with advanced techniques
   - Stop if performance is satisfactory

### LLM Decision Making

The LLM receives detailed performance metrics and can choose from several optimization strategies:

- **Parameter Tuning**: Adjust learning rates, exploration schedules, reward values
- **Multi-Config Testing**: Generate and test 3 different parameter sets, keep the best
- **Agent Rewriting**: Implement advanced RL techniques like:
  - Double Q-Learning
  - Experience replay
  - Curiosity-driven exploration
  - Count-based bonuses
  - Upper Confidence Bound (UCB) exploration

### Advanced Features

- **Automatic Validation**: All changes are tested before being applied
- **Rollback System**: Failed modifications automatically revert to working versions
- **Performance Tracking**: Detailed metrics including success rates, trap avoidance, convergence trends
- **Sense Action**: Optional sensing capability that reveals nearby traps at a small cost

## Example Workflow

```
Cycle 1: Train for 200 episodes → Poor performance (-1.5 avg return)
↓
LLM Analysis: "Learning rate too low, exploration insufficient"
↓
Action: Test 3 parameter variants with higher learning rates
↓
Result: Best variant improves to -0.8 avg return
↓
Cycle 2: Continue training → Moderate improvement
↓
LLM Analysis: "Need better exploration strategy"
↓
Action: Test 3 agents with curiosity-driven exploration
↓
Result: New agent achieves +0.5 avg return
```

## File Structure

```
├── env.py                 # Grid world environment
├── demo_train.py          # Basic training script
├── llm_rl_loop.py        # Main LLM optimization loop
├── llm_modifier.py       # LLM interaction and decision logic
├── config/
│   ├── config.json       # Main configuration file
│   ├── config_helper.py  # Config loading utilities
│   └── config_manager.py # Multi-config testing
├── RL/
│   ├── agent.py          # Q-learning agent
│   └── agent_manager.py  # Agent versioning and testing
├── prompts/
│   ├── prompt_builder.py # LLM prompt templates
│   └── param_descriptions.py # Parameter documentation
└── validation/
    ├── agent_validation.py # Agent code validation
    ├── param_validation.py # Parameter bounds checking
    └── test_runner.py      # Unified testing infrastructure
```

## Research Goals

This project demonstrates:
1. **Automated RL Optimization**: LLMs can intelligently tune RL systems
2. **Dynamic Algorithm Evolution**: Agents can be rewritten during training
3. **Multi-Objective Optimization**: Balance exploration, learning speed, and performance
4. **Robust Validation**: All modifications are tested before deployment

The goal is not necessarily to achieve perfect performance, but to showcase a working pipeline where LLMs can make informed decisions about RL system improvements based on quantitative feedback.

## Future Extensions

- **Environment Evolution**: LLM modifies grid layouts and reward structures
- **Advanced Metrics**: Include sample efficiency and robustness measures
- **Neural Network Agents**: Extend beyond tabular methods to deep RL

---

*This project implements the concept of meta-learning where an AI system learns how to improve another AI system's learning process.*
