"""
Unified Pipeline Interface for All Experiments

This module provides a unified interface that can handle:
- Basic pipeline (DefensePipeline)
- Adaptive pipeline (AdaptiveDefensePipeline) 
- Fixed vs Adaptive comparisons
- Isolation mode comparisons
"""

import sys
from pathlib import Path
from typing import Optional, Literal, Dict, Any
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline import DefensePipeline as OriginalDefensePipeline
from adaptive_pipeline import AdaptiveDefensePipeline
from src.models.execution_trace import ExecutionTrace
from src.models.request import RequestEnvelope as Request
from src.config import Config


class UnifiedDefensePipeline:
    """
    Unified interface for all pipeline configurations.
    
    This class provides a single interface that can handle:
    - Basic pipeline functionality
    - Adaptive coordination
    - Fixed vs adaptive comparisons
    - Isolation mode comparisons
    """
    
    def __init__(
        self,
        isolation_mode: Literal["bad", "good", "metadata", "strict"] = "good",
        coordination_enabled: bool = False,
        layer4_enhanced_monitoring: bool = False,
        layer5_strict_mode: bool = False,
        adaptive_layer3: bool = False,
        adaptive_layer4: bool = False,
        adaptive_layer5: bool = False,
        config: Optional[Config] = None
    ):
        """
        Initialize unified pipeline with configuration options.
        
        Args:
            isolation_mode: Context isolation mode
            coordination_enabled: Enable inter-layer coordination
            layer4_enhanced_monitoring: Fixed enhanced monitoring for Layer 4
            layer5_strict_mode: Fixed strict mode for Layer 5
            adaptive_layer3: Enable adaptive Layer 3
            adaptive_layer4: Enable adaptive Layer 4
            adaptive_layer5: Enable adaptive Layer 5
            config: Configuration object
        """
        self.isolation_mode = isolation_mode
        self.coordination_enabled = coordination_enabled
        self.layer4_enhanced_monitoring = layer4_enhanced_monitoring
        self.layer5_strict_mode = layer5_strict_mode
        
        # Determine which pipeline to use
        if coordination_enabled or adaptive_layer3 or adaptive_layer4 or adaptive_layer5:
            # Use adaptive pipeline for true coordination
            coord_config = {
                "enabled": coordination_enabled,
                "adaptive_layer3": adaptive_layer3,
                "adaptive_layer4": adaptive_layer4,
                "adaptive_layer5": adaptive_layer5,
                "layer3_escalation_thresholds": {
                    "strict": 0.6,
                    "metadata": 0.4
                },
                "layer4_monitoring_threshold": 0.5,
                "layer5_threshold_adjustment": 0.2
            }
            
            self.pipeline_type = "adaptive"
            self.pipeline = AdaptiveDefensePipeline(
                system_prompt=getattr(config, 'system_prompt', '').content if config else "You are a helpful assistant.",
                isolation_mode=isolation_mode,
                coordination_config=coord_config
            )
        else:
            # Use basic pipeline with enhanced parameters
            self.pipeline_type = "basic"
            self.pipeline = OriginalDefensePipeline(
                config=config,
                layer4_enhanced_monitoring=layer4_enhanced_monitoring,
                layer5_strict_mode=layer5_strict_mode
            )
    
    def execute_with_trace(self, user_input: str, experiment_id: Optional[str] = None) -> ExecutionTrace:
        """
        Execute a request and return detailed execution trace.
        
        Args:
            user_input: The user input/prompt to process
            experiment_id: Optional experiment identifier
            
        Returns:
            ExecutionTrace with complete execution history
        """
        # Create request object
        request = Request(
            user_input=user_input,
            session_id="unified_pipeline",
            metadata={
                "experiment_id": experiment_id,
                "pipeline_type": self.pipeline_type,
                "isolation_mode": self.isolation_mode,
                "coordination_enabled": self.coordination_enabled,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        if self.pipeline_type == "adaptive":
            # Use adaptive pipeline
            trace = self.pipeline.process_request(request)
            
            # Ensure the trace has all required fields for Experiment 6B/6C
            # Convert to standard ExecutionTrace if needed
            if not hasattr(trace, 'trust_boundary_violations'):
                trace.trust_boundary_violations = []
            if not hasattr(trace, 'bypass_mechanisms'):
                trace.bypass_mechanisms = []
            if not hasattr(trace, 'critical_failure_analysis'):
                trace.critical_failure_analysis = {}
                
        else:
            # Use basic pipeline with process method
            trace = self.pipeline.process(
                request,
                isolation_mode=self.isolation_mode,
                experiment_id=experiment_id,
                coordination_enabled=self.coordination_enabled,
                layer3_isolation_mode=self.isolation_mode  # Use the specified isolation mode
            )
        
        # Ensure critical_failure_analysis exists for compatibility with 6B/6C
        if not hasattr(trace, 'critical_failure_analysis') or trace.critical_failure_analysis is None:
            # Create a basic critical failure analysis
            trace.critical_failure_analysis = {
                "critical_failure_point": getattr(trace, 'blocked_at_layer', None),
                "reason": "First layer that failed to block attack" if getattr(trace, 'attack_successful', False) else "No critical failure - attack blocked",
                "risk_score": max([step.get('risk_score', 0) for step in getattr(trace, 'propagation_path', [])], default=0)
            }
        
        # Ensure critical_failure_point exists for database compatibility
        if not hasattr(trace, 'critical_failure_point'):
            trace.critical_failure_point = trace.critical_failure_analysis
        
        return trace
    
    def __repr__(self) -> str:
        """String representation."""
        return f"UnifiedDefensePipeline(type={self.pipeline_type}, isolation={self.isolation_mode}, coordination={self.coordination_enabled})"