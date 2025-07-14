"""
config_manager.py
~~~~~~~~~~~~~~~~
Manages configuration versions and allows LLM to test multiple hyperparameter sets.
"""

import sys
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

from config.config_helper import get_config_singleton, save_config
from validation.param_validation import validate_parameter, get_parameter_category, test_config_performance

class ConfigManager:
    """Manages config versions and multi-config testing."""
    
    def __init__(self):
        self.config_file = Path("config/config.json")
        self.history_dir = Path("config/config_history")
        self.temp_configs_dir = Path("config/temp_configs")
        self.history_dir.mkdir(exist_ok=True)
        self.temp_configs_dir.mkdir(exist_ok=True)
        
        # Define which parameters are for environment vs agent optimization
        self.environment_params = {
            "grid_size", "n_traps", "move_penalty", "trap_penalty", "goal_reward"
        }
        self.agent_params = {
            "learning_rate", "gamma", "epsilon_start", "epsilon_min", "epsilon_decay"
        }
        self.training_params = {
            "episodes", "max_steps_per_episode"
        }
        
    def backup_current_config(self) -> str:
        """Backup current config.json with timestamp."""
        if not self.config_file.exists():
            return ""
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"config_{timestamp}.json"
        backup_path = self.history_dir / backup_name
        
        shutil.copy2(self.config_file, backup_path)
        print(f"📝 Backed up config to {backup_path}")
        return str(backup_path)
    
    def generate_config_variants(self, base_config: Dict, param_changes_list: List[Dict], focus_area: str) -> List[Dict]:
        """Generate config variants based on parameter changes."""
        variants = []
        
        for i, param_changes in enumerate(param_changes_list):
            variant = deepcopy(base_config)
            
            # Apply parameter changes to the appropriate sections
            for param, value in param_changes.items():
                if focus_area == "agent" and param not in self.agent_params:
                    print(f"⚠️  Skipping {param} - not an agent parameter")
                    continue
                elif focus_area == "training" and param not in self.training_params:
                    print(f"⚠️  Skipping {param} - not a training parameter")
                    continue
                    
                # Validate parameter
                is_valid, error_msg = validate_parameter(param, value, base_config)
                if not is_valid:
                    print(f"⚠️  {error_msg} - skipping variant {i+1}")
                    continue
                
                # Apply to correct section
                category = get_parameter_category(param)
                if category and category in variant:
                    variant[category][param] = value
                else:
                    print(f"⚠️  Unknown category for {param}")
            
            variants.append(variant)
        
        return variants
    
    def test_multiple_configs(self, config_variants: List[Dict], num_test_episodes: int = 100) -> Tuple[int, Dict[str, float]]:
        """
        Test multiple config variants and return the index of the best one.
        Returns (best_index, metrics_dict) where best_index is -1 if original is best.
        """
        print(f"🧪 Testing {len(config_variants)} config variants...")
        
        # First, test the original config
        original_metrics = test_config_performance(get_config_singleton(), num_test_episodes, "Original")
        print(f"📊 Original config performance: {original_metrics['avg_return']:.3f}")
        
        # If original failed, skip testing variants
        if original_metrics['avg_return'] <= -999:
            print("❌ Original config test failed - skipping variant testing")
            return -1, original_metrics
        
        best_metrics = original_metrics
        best_index = -1  # -1 means original is best
        
        # Test each variant
        for i, config_variant in enumerate(config_variants):
            try:
                # Save variant to temp file
                variant_file = self.temp_configs_dir / f"variant_{i}.json"
                with open(variant_file, 'w') as f:
                    json.dump(config_variant, f, indent=4)
                
                # Test this variant
                metrics = test_config_performance(config_variant, num_test_episodes, f"Variant {i+1}")
                print(f"📊 Variant {i+1} performance: {metrics['avg_return']:.3f}")
                
                # Skip failed tests
                if metrics['avg_return'] <= -999:
                    print(f"❌ Variant {i+1} test failed - skipping")
                    continue
                
                # Check if this variant is better by a meaningful margin
                improvement_threshold = 0.05  # 5% improvement required
                if metrics['avg_return'] > best_metrics['avg_return'] + improvement_threshold:
                    print(f"🎯 Variant {i+1} is new best! ({metrics['avg_return']:.3f} vs {best_metrics['avg_return']:.3f})")
                    best_metrics = metrics
                    best_index = i
                    
            except Exception as e:
                print(f"❌ Variant {i+1} failed testing: {e}")
                continue
        
        # Clean up temp files
        self._cleanup_temp_configs()
        
        return best_index, best_metrics
    
    def save_best_config(self, config_variants: List[Dict], best_index: int) -> bool:
        """Save the best config after testing."""
        if best_index == -1:
            print("🏆 Original config is still the best - no changes made")
            return False
        
        try:
            # Backup current version
            self.backup_current_config()
            
            # Save the best variant
            best_config = config_variants[best_index]
            save_config(best_config)
            
            print(f"✅ Best config (Variant {best_index + 1}) saved successfully")
            return True
                
        except Exception as e:
            print(f"❌ Error saving best config: {e}")
            self._restore_latest_backup()
            return False
    
    def _restore_latest_backup(self):
        """Restore the most recent backup."""
        backups = list(self.history_dir.glob("config_*.json"))
        if backups:
            latest = max(backups, key=lambda p: p.stat().st_ctime)
            shutil.copy2(latest, self.config_file)
            print(f"🔄 Restored from backup: {latest}")
    
    def _cleanup_temp_configs(self):
        """Clean up temporary config files."""
        for temp_file in self.temp_configs_dir.glob("variant_*.json"):
            temp_file.unlink(missing_ok=True)