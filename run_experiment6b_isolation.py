#!/usr/bin/env python3
"""
Experiment 6B: Isolation Mode Comparison
Compare different isolation architectures and quantify trust boundary violations
"""

import os
import sys
import json
import sqlite3
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from unified_pipeline import UnifiedDefensePipeline
from src.models.execution_trace import ExecutionTrace
from src.database import DatabaseManager
from src.config import Config

def run_isolation_experiment():
    """Run experiment comparing different isolation modes."""
    
    print("="*80)
    print("EXPERIMENT 6B: Isolation Mode Comparison")
    print("="*80)
    
    # Define configurations for different isolation modes
    configurations = {
        "pod1_bad": {
            "name": "Bad Isolation (Concatenation)",
            "layer3_isolation_mode": "bad",
            "coordination_enabled": False,
            "description": "Simple concatenation - no isolation"
        },
        "pod2_good": {
            "name": "Good Isolation (Role Separation)",
            "layer3_isolation_mode": "good", 
            "coordination_enabled": False,
            "description": "Basic role separation"
        },
        "pod3_metadata": {
            "name": "Metadata Isolation (XML Tags)",
            "layer3_isolation_mode": "metadata",
            "coordination_enabled": False,
            "description": "Explicit boundaries with metadata"
        },
        "pod4_strict": {
            "name": "Strict Isolation (Hard Boundary)",
            "layer3_isolation_mode": "strict",
            "coordination_enabled": False,
            "description": "Hard isolation with validation"
        }
    }
    
    # Load attack prompts
    from data.attack_prompts import ATTACK_PROMPTS
    attack_prompts = ATTACK_PROMPTS[:21]  # Use subset for this experiment
    
    print(f"Loaded {len(attack_prompts)} attack prompts")
    print(f"Configurations: {list(configurations.keys())}")
    
    # Initialize database
    db_path = Path("results") / "exp6b_isolation.db"
    db_path.parent.mkdir(exist_ok=True)
    if db_path.exists():
        db_path.unlink()  # Remove old results
    
    db = DatabaseManager(str(db_path))
    db.init_schema()
    
    # Run experiment for each configuration
    results = {}
    
    for config_name, config_params in configurations.items():
        print(f"\n--- Running {config_params['name']} ---")
        
        # Create pipeline with specific configuration
        pipeline = UnifiedDefensePipeline(
            isolation_mode=config_params["layer3_isolation_mode"],
            coordination_enabled=config_params["coordination_enabled"]
        )
        
        # Run all attacks for this configuration
        config_results = []
        successful_attacks = 0
        total_attacks = len(attack_prompts)
        
        for i, attack_prompt in enumerate(attack_prompts):
            try:
                # Execute the attack
                trace = pipeline.execute_with_trace(attack_prompt)
                
                # Record result
                config_results.append({
                    "config_name": config_name,
                    "prompt_id": i,
                    "attack_prompt": attack_prompt,
                    "attack_successful": trace.attack_successful,
                    "propagation_path": trace.propagation_path,
                    "trust_boundary_violations": trace.trust_boundary_violations,
                    "critical_failure_analysis": trace.critical_failure_analysis,
                    "timestamp": datetime.now().isoformat()
                })
                
                if trace.attack_successful:
                    successful_attacks += 1
                
                # Progress indicator
                if (i + 1) % 5 == 0:
                    print(f"  Processed {i + 1}/{total_attacks} attacks")
                    
            except Exception as e:
                print(f"  Error processing attack {i}: {e}")
                continue
        
        # Calculate ASR for this configuration
        asr = successful_attacks / total_attacks if total_attacks > 0 else 0
        print(f"  ASR: {asr:.1%} ({successful_attacks}/{total_attacks})")
        
        # Count trust boundary violations
        total_violations = sum(
            len(result["trust_boundary_violations"]) 
            for result in config_results
        )
        avg_violations = total_violations / total_attacks if total_attacks > 0 else 0
        print(f"  Avg. Trust Boundary Violations: {avg_violations:.2f}")
        
        # Store results
        results[config_name] = {
            "asr": asr,
            "successful_attacks": successful_attacks,
            "total_attacks": total_attacks,
            "avg_trust_violations": avg_violations,
            "detailed_results": config_results
        }
        
        # Save to database
        for result in config_results:
            db.insert_execution_trace(result)
    
    print("\n" + "="*80)
    print("EXPERIMENT 6B RESULTS SUMMARY")
    print("="*80)
    
    for config_name, result in results.items():
        config_info = configurations[config_name]
        print(f"\n{config_info['name']}:")
        print(f"  ASR: {result['asr']:.1%}")
        print(f"  Avg. Trust Violations: {result['avg_trust_violations']:.2f}")
        print(f"  Description: {config_info['description']}")
    
    # Save detailed results as JSON
    json_path = Path("results") / "exp6b_isolation_summary.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to:")
    print(f"  Database: {db_path}")
    print(f"  JSON: {json_path}")
    
    # Perform statistical analysis
    perform_statistical_analysis(results)
    
    return results

def perform_statistical_analysis(results):
    """Perform basic statistical analysis on the results."""
    print("\n" + "-"*50)
    print("STATISTICAL ANALYSIS")
    print("-"*50)
    
    # Compare configurations
    configs = list(results.keys())
    for i, config1 in enumerate(configs):
        for j, config2 in enumerate(configs[i+1:], i+1):
            asr1 = results[config1]["asr"]
            asr2 = results[config2]["asr"]
            
            diff = asr1 - asr2
            print(f"{config1} vs {config2}:")
            print(f"  ASR Difference: {diff:.1%}")
            if diff > 0:
                print(f"  {config1} has {diff:.1%} higher ASR (worse)")
            else:
                print(f"  {config2} has {abs(diff):.1%} higher ASR (worse)")

if __name__ == "__main__":
    run_isolation_experiment()