from __future__ import annotations
import json
import re
from typing import Dict, List
import os
from pathlib import Path

from openai import OpenAI                 # new ≥1.0.0 interface

# Import prompt modules
from prompts.prompt_builder import build_tuning_prompt, build_multi_agent_prompt
from prompts.param_descriptions import get_param_descriptions

from config.config_helper import get_config_singleton, get_flat_config

from RL.agent_manager import AgentManager
from config.config_manager import ConfigManager
from prompts.prompt_builder import build_multi_config_prompt

KEY_FILE = Path(__file__).with_name("openai_key.txt")

# ---------- OpenAI client ---------------------------------------------------
def create_openai_client() -> OpenAI:
    """Load the API key and return an OpenAI client instance."""
    if KEY_FILE.exists():
        api_key = KEY_FILE.read_text().strip()
        if not api_key:
            raise RuntimeError(f"{KEY_FILE} is empty.")
    else:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "No OpenAI key found. Put it in 'openai_key.txt' or set OPENAI_API_KEY."
        )

    return OpenAI(api_key=api_key)

# ---------- prompt & chat completion ---------------------------------------
_JSON_RE = re.compile(r"{.*}", re.DOTALL)

def build_prompt(metrics: Dict[str, float]) -> str:
    """Build the tuning prompt using the prompts module."""
    
    # Use nested config for param_descriptions (which needs nested structure)
    config = get_config_singleton()
    param_descriptions = get_param_descriptions(config)
    
    # Use flat config for simple parameter display
    flat_config = get_flat_config()
    
    return build_tuning_prompt(metrics, flat_config, param_descriptions)

def query_llm(client: OpenAI, prompt: str) -> Dict[str, object]:
    config = get_config_singleton()

    resp = client.chat.completions.create(
        model=config["system"]["model_name"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=config["system"]["max_tokens"],
        temperature=config["system"]["temperature"],
    )
    raw = resp.choices[0].message.content
    match = _JSON_RE.search(raw or "")
    if not match:
        print("⚠️  LLM response lacked JSON — skipping patch.")
        return {"patch": {}}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        print("⚠️  Failed to parse JSON — skipping patch.")
        return {"patch": {}}

# ---------- CONFIG patching -------------------------------------------------
def query_llm_for_multiple_configs(client: OpenAI, metrics: Dict[str, float], focus_area: str, num_variants: int = 3) -> List[Dict]:
    """Query LLM to generate multiple config variants."""
    config = get_config_singleton()
    
    prompt = build_multi_config_prompt(config, metrics, focus_area, num_variants)
    
    resp = client.chat.completions.create(
        model=config["system"]["model_name"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.5,  # Moderate temperature for variety
    )
    
    response_text = resp.choices[0].message.content or ""
    return extract_multiple_configs(response_text, num_variants)

def extract_multiple_configs(response_text: str, expected_count: int) -> List[Dict]:
    """Extract multiple config variants from LLM response."""
    variants = []
    
    # Split by variant separators
    sections = response_text.split("# VARIANT")
    
    for i, section in enumerate(sections[1:], 1):  # Skip first empty section
        # Extract JSON from this section
        json_lines = []
        in_json_block = False
        
        for line in section.split('\n'):
            if line.strip().startswith('```json') or line.strip().startswith('```'):
                in_json_block = not in_json_block
                continue
            if in_json_block:
                json_lines.append(line)
        
        json_text = '\n'.join(json_lines).strip()
        if json_text:
            try:
                variant = json.loads(json_text)
                variants.append(variant)
                print(f"📝 Extracted Variant {i}: {variant}")
            except json.JSONDecodeError as e:
                print(f"⚠️  Failed to parse Variant {i} JSON: {e}")
    
    print(f"🔍 Extracted {len(variants)} config variants from LLM response")
    return variants

def apply_multi_config_optimization(metrics: Dict[str, float], focus_area: str = "agent", num_variants: int = 3) -> bool:
    """Generate and test multiple configs, keeping the best one."""
    try:
        client = create_openai_client()
        config_manager = ConfigManager()
        
        # Get multiple config variants from LLM
        print(f"🤖 Querying LLM for {num_variants} {focus_area} config variants...")
        param_changes_list = query_llm_for_multiple_configs(client, metrics, focus_area, num_variants)
        
        if len(param_changes_list) < num_variants:
            print(f"⚠️  Only got {len(param_changes_list)} variants instead of {num_variants}")
        
        if not param_changes_list:
            print("⚠️  No valid config variants extracted")
            return False
        
        # Generate full config variants
        base_config = get_config_singleton()
        config_variants = config_manager.generate_config_variants(base_config, param_changes_list, focus_area)
        
        if not config_variants:
            print("⚠️  No valid config variants generated")
            return False
        
        # Test all variants and find the best one
        best_index, best_metrics = config_manager.test_multiple_configs(config_variants, num_test_episodes=100)
        
        if best_index == -1:
            print("🏆 Original config outperformed all variants")
            return False
        
        # Save the best config
        success = config_manager.save_best_config(config_variants, best_index)
        if success:
            print(f"🔄 Config replaced with Variant {best_index + 1}")
            print(f"📈 Performance improved to {best_metrics['avg_return']:.3f}")
        
        return success
        
    except Exception as e:
        print(f"❌ Error in multi-config optimization: {e}")
        return False

# ---------- Agent rewriting -------------------------------------------------
def query_llm_for_multiple_agents(client: OpenAI, metrics: Dict[str, float], current_agent_code: str, num_agents: int = 3) -> List[str]:
    """Query LLM to generate multiple agent candidates."""
    config = get_config_singleton()
    
    prompt = build_multi_agent_prompt(config, metrics, current_agent_code, num_agents)
    
    resp = client.chat.completions.create(
        model=config["system"]["model_name"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=6000,  # More tokens for multiple agents
        temperature=0.7,  # Higher temperature for diversity
    )
    
    response_text = resp.choices[0].message.content or ""
    return extract_multiple_agents(response_text, num_agents)

def extract_multiple_agents(response_text: str, expected_count: int) -> List[str]:
    """Extract multiple agent code blocks from LLM response."""
    agents = []
    
    # Split by agent separators
    sections = response_text.split("# AGENT")
    
    for i, section in enumerate(sections[1:], 1):  # Skip first empty section
        # Extract code from this section
        code_lines = []
        in_code_block = False
        
        for line in section.split('\n'):
            if line.strip().startswith('```python') or line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                code_lines.append(line)
        
        code = '\n'.join(code_lines).strip()
        if code and 'class Agent:' in code:
            agents.append(code)
            print(f"📝 Extracted Agent {i} ({len(code)} chars)")
    
    print(f"🔍 Extracted {len(agents)} agents from LLM response")
    return agents

def apply_multi_agent_rewrite(metrics: Dict[str, float], num_agents: int = 3) -> bool:
    """Generate and test multiple agents, keeping the best one."""
    try:
        client = create_openai_client()
        agent_manager = AgentManager()
        
        # Read current agent code
        with open("RL/agent.py", 'r') as f:
            current_code = f.read()
        
        # Get multiple agent candidates from LLM
        print(f"🤖 Querying LLM for {num_agents} agent candidates...")
        agent_codes = query_llm_for_multiple_agents(client, metrics, current_code, num_agents)
        
        if len(agent_codes) < num_agents:
            print(f"⚠️  Only got {len(agent_codes)} agents instead of {num_agents}")
        
        if not agent_codes:
            print("⚠️  No valid agent code extracted")
            return False
        
        # Test all candidates and find the best one
        best_index, best_metrics = agent_manager.test_multiple_agents(agent_codes, num_test_episodes=100)
        
        if best_index == -1:
            print("🏆 Original agent outperformed all candidates")
            return False
        
        # Save the best agent
        success = agent_manager.save_best_agent(agent_codes, best_index)
        if success:
            print(f"🔄 Agent replaced with Candidate {best_index + 1}")
            print(f"📈 Performance improved to {best_metrics['avg_return']:.3f}")
            agent_manager.reload_agent_module()
        
        return success
        
    except Exception as e:
        print(f"❌ Error in multi-agent rewrite: {e}")
        return False
    
