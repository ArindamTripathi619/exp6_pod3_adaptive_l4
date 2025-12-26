#!/usr/bin/env python3
"""
Experiment 6C: Fixed vs Adaptive Comparison
Compare all-layers-fixed vs all-layers-adaptive to prove coordination value
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

def run_fixed_vs_adaptive_experiment():
    """Run experiment comparing fixed vs adaptive configurations."""
    
    print("="*80)
    print("EXPERIMENT 6C: Fixed vs Adaptive Comparison")
    print("="*80)
    
    # Define configurations for fixed vs adaptive
    configurations = {
        "pod1_fixed": {
            "name": "All Layers Fixed (No Coordination)",
            "coordination_enabled": False,
            "layer3_isolation_mode": "good",  # FIXED
            "layer4_enhanced_monitoring": True,  # FIXED ON for fair comparison
            "layer5_strict_mode": True,  # FIXED ON for fair comparison
            "adaptive_layer3": False,
            "adaptive_layer4": False,
            "adaptive_layer5": False,
            "description": "All layers enabled but no adaptive behavior"
        },
        "pod2_adaptive": {
            "name": "All Layers Adaptive (Full Coordination)",
            "coordination_enabled": True,
            "layer3_isolation_mode": "good",  # Can adapt based on risk
            "layer4_enhanced_monitoring": False,  # Will adapt based on risk
            "layer5_strict_mode": False,  # Will adapt based on risk
            "adaptive_layer3": True,
            "adaptive_layer4": True,
            "adaptive_layer5": True,
            "description": "All layers with adaptive coordination"
        }
    }
    
    # Load attack prompts
    from data.attack_prompts import ATTACK_PROMPTS
    attack_prompts = ATTACK_PROMPTS[:21]  # Use subset for this experiment
    
    print(f"Loaded {len(attack_prompts)} attack prompts")
    print(f"Configurations: {list(configurations.keys())}")
    
    # Initialize database
    db_path = Path("results") / "exp6c_fixed_vs_adaptive.db"
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
            coordination_enabled=config_params["coordination_enabled"],
            layer4_enhanced_monitoring=config_params.get("layer4_enhanced_monitoring", False),
            layer5_strict_mode=config_params.get("layer5_strict_mode", False),
            adaptive_layer3=config_params.get("adaptive_layer3", False),
            adaptive_layer4=config_params.get("adaptive_layer4", False),
            adaptive_layer5=config_params.get("adaptive_layer5", False)
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
    print("EXPERIMENT 6C RESULTS SUMMARY")
    print("="*80)
    
    for config_name, result in results.items():
        config_info = configurations[config_name]
        print(f"\n{config_info['name']}:")
        print(f"  ASR: {result['asr']:.1%}")
        print(f"  Avg. Trust Violations: {result['avg_trust_violations']:.2f}")
        print(f"  Description: {config_info['description']}")
    
    # Calculate coordination benefit
    if "pod1_fixed" in results and "pod2_adaptive" in results:
        fixed_asr = results["pod1_fixed"]["asr"]
        adaptive_asr = results["pod2_adaptive"]["asr"]
        benefit = fixed_asr - adaptive_asr
        
        print(f"\nCOORDINATION BENEFIT:")
        print(f"  Fixed ASR: {fixed_asr:.1%}")
        print(f"  Adaptive ASR: {adaptive_asr:.1%}")
        print(f"  Coordination Benefit: {benefit:.1%} reduction")
        print(f"  Improvement: {((fixed_asr - adaptive_asr) / fixed_asr * 100):.1f}% relative improvement")
    
    # Save detailed results as JSON
    json_path = Path("results") / "exp6c_fixed_vs_adaptive_summary.json"
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
    
    if "pod1_fixed" in results and "pod2_adaptive" in results:
        fixed_asr = results["pod1_fixed"]["asr"]
        adaptive_asr = results["pod2_adaptive"]["asr"]
        
        print(f"Fixed Configuration ASR: {fixed_asr:.1%}")
        print(f"Adaptive Configuration ASR: {adaptive_asr:.1%}")
        
        benefit = fixed_asr - adaptive_asr
        print(f"Absolute Benefit: {benefit:.1%}")
        
        if fixed_asr > 0:
            relative_improvement = (benefit / fixed_asr) * 100
            print(f"Relative Improvement: {relative_improvement:.1f}%")
        
        if benefit > 0:
            print(f"✓ Coordination provides {benefit:.1%} ASR reduction")
        else:
            print(f"⚠ Coordination does not provide benefit (negative {abs(benefit):.1%})")

if __name__ == "__main__":
    run_fixed_vs_adaptive_experiment()