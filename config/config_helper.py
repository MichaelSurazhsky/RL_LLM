import json
from pathlib import Path
from typing import Dict, List

# ─────────── CONFIG LOADING ──────────────────────────────────────────────
def load_config() -> Dict:
    """Load configuration from config.json file."""
    config_path = Path(__file__).parent / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)
    
def get_config_singleton():
    """Get a singleton config instance to avoid repeated loading."""
    if not hasattr(get_config_singleton, '_config'):
        get_config_singleton._config = load_config()
    return get_config_singleton._config

def get_flat_config():
    """Get flattened config with caching."""
    if not hasattr(get_flat_config, '_flat_config'):
        config_data = get_config_singleton()
        flat_config = {}
        flat_config.update(config_data["environment"])
        flat_config.update(config_data["rewards"])
        flat_config.update(config_data["agent"])
        flat_config.update(config_data["training"])
        flat_config.update(config_data["system"])
        
        # Add alias for backward compatibility
        if "learning_rate" in flat_config:
            flat_config["alpha"] = flat_config["learning_rate"]
        
        get_flat_config._flat_config = flat_config
    return get_flat_config._flat_config

def save_config(config: Dict) -> None:
    """Save configuration and clear cache."""
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Clear cached configs to force reload
    if hasattr(get_config_singleton, '_config'):
        delattr(get_config_singleton, '_config')
    if hasattr(get_flat_config, '_flat_config'):
        delattr(get_flat_config, '_flat_config')
