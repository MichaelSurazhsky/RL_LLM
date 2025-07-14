"""
agent_manager.py
~~~~~~~~~~~~~~~~
Manages agent code versions and allows LLM to rewrite the agent class.
"""

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import importlib.util

from validation.agent_validation import validate_agent_code, test_agent_performance

class AgentManager:
    """Manages agent code versions and dynamic reloading."""
    
    def __init__(self):
        self.agent_file = Path("RL/agent.py")
        self.history_dir = Path("RL/agent_history")
        self.temp_agents_dir = Path("RL/temp_agents")
        self.history_dir.mkdir(exist_ok=True)
        self.temp_agents_dir.mkdir(exist_ok=True)
        
    def backup_current_agent(self) -> str:
        """Backup current agent.py with timestamp."""
        if not self.agent_file.exists():
            return ""
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"agent_{timestamp}.py"
        backup_path = self.history_dir / backup_name
        
        shutil.copy2(self.agent_file, backup_path)
        print(f"📝 Backed up agent to {backup_path}")
        return str(backup_path)
    
    def _restore_latest_backup(self):
        """Restore the most recent backup."""
        backups = list(self.history_dir.glob("agent_*.py"))
        if backups:
            latest = max(backups, key=os.path.getctime)
            shutil.copy2(latest, self.agent_file)
            print(f"🔄 Restored from backup: {latest}")
    
    def reload_agent_module(self):
        """Reload the agent module to pick up changes."""
        if 'agent' in sys.modules:
            importlib.reload(sys.modules['agent'])
        else:
            import RL.agent as agent
    
    def list_agent_history(self) -> list:
        """List all agent backup files."""
        backups = list(self.history_dir.glob("agent_*.py"))
        return sorted(backups, key=os.path.getctime, reverse=True)
    
    def test_multiple_agents(self, agent_codes: List[str], num_test_episodes: int = 100) -> Tuple[int, Dict[str, float]]:
        """
        Test multiple agent candidates and return the index of the best one.
        Returns (best_index, metrics_dict) where best_index is -1 if original is best.
        """
        print(f"🧪 Testing {len(agent_codes)} agent candidates...")
        
        # First, test the original agent
        original_metrics = test_agent_performance(self.agent_file, num_test_episodes, "Original")
        print(f"📊 Original agent performance: {original_metrics['avg_return']:.3f}")
        
        best_metrics = original_metrics
        best_index = -1  # -1 means original is best
        
        # Test each candidate agent
        for i, agent_code in enumerate(agent_codes):
            candidate_file = self.temp_agents_dir / f"candidate_{i}.py"
            
            try:
                # Write candidate code to temp file
                with open(candidate_file, 'w') as f:
                    f.write(agent_code)
                
                # Test this candidate
                metrics = test_agent_performance(candidate_file, num_test_episodes, f"Candidate {i+1}")
                print(f"📊 Candidate {i+1} performance: {metrics['avg_return']:.3f}")
                
                # Check if this candidate is better by a meaningful margin
                improvement_threshold = 0.1  # 10% improvement required
                if metrics['avg_return'] > best_metrics['avg_return'] * (1 + improvement_threshold):
                    print(f"🎯 Candidate {i+1} is new best! ({metrics['avg_return']:.3f} vs {best_metrics['avg_return']:.3f})")
                    best_metrics = metrics
                    best_index = i
                    
            except Exception as e:
                print(f"❌ Candidate {i+1} failed testing: {e}")
                continue
        
        # Clean up temp files
        self._cleanup_temp_agents()
        
        return best_index, best_metrics
    
    def save_best_agent(self, agent_codes: List[str], best_index: int) -> bool:
        """Save the best agent after testing."""
        if best_index == -1:
            print("🏆 Original agent is still the best - no changes made")
            return False
        
        try:
            # Backup current version
            self.backup_current_agent()
            
            # Save the best candidate
            best_code = agent_codes[best_index]
            with open(self.agent_file, 'w') as f:
                f.write(best_code)
            
            # Validate the saved code
            if validate_agent_code(self):
                print(f"✅ Best agent (Candidate {best_index + 1}) saved successfully")
                return True
            else:
                print("❌ Best agent failed final validation")
                self._restore_latest_backup()
                return False
                
        except Exception as e:
            print(f"❌ Error saving best agent: {e}")
            self._restore_latest_backup()
            return False
    
    def _cleanup_temp_agents(self):
        """Clean up temporary agent files."""
        for temp_file in self.temp_agents_dir.glob("candidate_*.py"):
            temp_file.unlink(missing_ok=True)
        
        # Clean up __pycache__ folder in temp_agents
        pycache_dir = self.temp_agents_dir / "__pycache__"
        if pycache_dir.exists():
            shutil.rmtree(pycache_dir)