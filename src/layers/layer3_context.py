"""Layer 3: Context Isolation.

This layer enforces separation between system instructions and user input
to prevent user content from being interpreted as system commands.
"""

import logging
import time
from typing import Optional, Dict, List, Literal, Any
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import RequestEnvelope, LayerResult
from config import Config

logger = logging.getLogger(__name__)


class Layer3ContextIsolation:
    """
    Layer 3: Context Isolation.
    
    Enforces architectural separation between system prompts and user input.
    This is the MOST IMPORTANT layer according to the technical plan, as it
    prevents user input from contaminating system instructions through proper
    role-based message structuring.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize context isolation layer."""
        self.config = config or Config.get()
        self.use_role_separation = getattr(
            self.config.layers, 'use_role_separation', True
        )
        self.use_metadata_tags = getattr(
            self.config.layers, 'use_metadata_tags', True
        )
        
    def isolate(
        self, 
        request: RequestEnvelope,
        system_prompt: str,
        mode: Literal["bad", "good", "metadata", "strict"] = "good",
        coordination_context: Optional[Dict[str, any]] = None
    ) -> tuple[LayerResult, Dict[str, any]]:
        """
        Create isolated context for LLM interaction.
        
        Args:
            request: The incoming request envelope
            system_prompt: The system prompt to use
            mode: Isolation mode:
                - "bad": Concatenate user input with system prompt (vulnerable)
                - "good": Separate system/user messages with roles
                - "metadata": Add XML-style tags for separation
                - "strict": Hard isolation with no shared context
            coordination_context: Shared context from upstream layers for adaptive behavior
            
        Returns:
            Tuple of (LayerResult, isolated_context)
        """
        start_time = time.time()
        flags = []
        annotations = {}
        passed = True
        confidence = 1.0
        risk_score = 0.0
        
        annotations["isolation_mode"] = mode
        
        # Check if coordination is enabled and upstream layers flagged risks
        upstream_risk = 0.0
        adaptive_mode_adjustment = False
        
        if coordination_context and coordination_context.get("enabled"):
            upstream_risk = coordination_context.get("upstream_risk_score", 0.0)
            
            # COORDINATION: Adapt isolation mode based on upstream risk
            if coordination_context.get("risk_escalation") and mode == "good":
                # Escalate from "good" to "strict" if upstream layers detected high risk
                original_mode = mode
                mode = "strict"
                adaptive_mode_adjustment = True
                flags.append("adaptive_escalation")
                annotations["coordination_adjustment"] = {
                    "original_mode": original_mode,
                    "adjusted_mode": mode,
                    "reason": "upstream_risk_escalation",
                    "upstream_risk_score": upstream_risk
                }
                logger.info(
                    f"Layer3 [{request.request_id}]: Coordination enabled - "
                    f"escalating from '{original_mode}' to '{mode}' due to upstream risk={upstream_risk:.2f}"
                )
        
        # Create isolated context based on mode
        if mode == "bad":
            # BAD: Simple concatenation (for baseline experiments)
            isolated_context = {
                "type": "concatenated",
                "prompt": f"{system_prompt}\n\nUser: {request.user_input}",
                "origin": "mixed",  # User and system content mixed
                "privilege": "user",  # Treated at user privilege level
            }
            flags.append("no_isolation")
            risk_score = 0.8
            annotations["warning"] = "No context isolation - vulnerable to injection"
            logger.warning(
                f"Request {request.request_id}: Using BAD isolation mode (concatenated)"
            )
            
        elif mode == "good":
            # GOOD: Role-based separation (recommended)
            isolated_context = {
                "type": "role_separated",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                        "origin": "system",
                        "privilege": "system"
                    },
                    {
                        "role": "user",
                        "content": request.user_input,
                        "origin": "user",
                        "privilege": "user"
                    }
                ]
            }
            flags.append("role_separation")
            annotations["separation_method"] = "role_based"
            
        elif mode == "metadata":
            # METADATA: XML-style tags for additional separation
            isolated_context = {
                "type": "metadata_tagged",
                "messages": [
                    {
                        "role": "system",
                        "content": f"<SYSTEM_INSTRUCTION>\n{system_prompt}\n</SYSTEM_INSTRUCTION>",
                        "origin": "system",
                        "privilege": "system"
                    },
                    {
                        "role": "user",
                        "content": f"<USER_INPUT>\n{request.user_input}\n</USER_INPUT>",
                        "origin": "user",
                        "privilege": "user"
                    }
                ]
            }
            flags.append("metadata_tags")
            flags.append("role_separation")
            annotations["separation_method"] = "metadata_tagged"
            
        elif mode == "strict":
            # STRICT: Maximum isolation (for high-security experiments)
            isolated_context = {
                "type": "strict_isolation",
                "system_context": {
                    "content": system_prompt,
                    "hash": hash(system_prompt),
                    "origin": "system",
                    "privilege": "system",
                    "immutable": True
                },
                "user_context": {
                    "content": request.user_input,
                    "origin": "user",
                    "privilege": "user",
                    "sanitized": True
                },
                "allow_mixing": False
            }
            flags.append("strict_isolation")
            annotations["separation_method"] = "strict"
            confidence = 1.0
            
        else:
            raise ValueError(f"Unknown isolation mode: {mode}")
        
        # Check for potential boundary violations in user input
        violations = self._check_boundary_violations(request.user_input)
        trust_violations = []
        
        if violations:
            flags.extend([f"violation_{v}" for v in violations])
            annotations["boundary_violations"] = violations
            risk_score = max(risk_score, 0.5)
            logger.warning(
                f"Request {request.request_id}: Boundary violation indicators detected - {violations}"
            )
            
            # NEW: Validate trust boundaries and detect specific violation types
            trust_violations = self._validate_trust_boundaries(
                request.user_input, 
                system_prompt,
                mode,
                upstream_risk
            )
            
            if trust_violations:
                annotations["trust_boundary_violations"] = trust_violations
                flags.append("trust_violation_detected")
                
                # Store trust violations in coordination context for downstream layers
                if coordination_context:
                    coordination_context["trust_boundary_violations"].extend(trust_violations)
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = LayerResult(
            layer_name="Layer3_Context",
            passed=passed,
            confidence=confidence,
            flags=flags,
            annotations=annotations,
            risk_score=risk_score,
            latency_ms=latency_ms
        )
        
        logger.info(
            f"Layer3 [{request.request_id}]: mode={mode}, "
            f"flags={flags}, latency={latency_ms:.2f}ms"
        )
        
        return result, isolated_context
    
    def _check_boundary_violations(self, user_input: str) -> List[str]:
        """
        Check for indicators of attempts to violate context boundaries.
        
        Args:
            user_input: The user's input text
            
        Returns:
            List of violation types detected
        """
        violations = []
        input_lower = user_input.lower()
        
        # Check for role impersonation attempts
        role_keywords = ["system:", "assistant:", "[system]", "<system>", "role:"]
        if any(kw in input_lower for kw in role_keywords):
            violations.append("role_impersonation")
        
        # Check for context override attempts
        override_keywords = [
            "new instructions:", "system message:", "override:",
            "admin mode:", "debug mode:", "you are now"
        ]
        if any(kw in input_lower for kw in override_keywords):
            violations.append("context_override")
        
        # Check for XML/tag injection attempts
        if any(tag in user_input for tag in ["<system>", "</system>", "<instruction>", "</instruction>"]):
            violations.append("tag_injection")
        
        # Check for attempts to escape context
        escape_sequences = ["```", "---", "===", "###"]
        if any(seq in user_input for seq in escape_sequences):
            violations.append("escape_sequence")
        
        return violations
    
    def _validate_trust_boundaries(
        self, 
        user_input: str, 
        system_prompt: str,
        isolation_mode: str,
        upstream_risk: float
    ) -> List[Dict[str, Any]]:
        """
        Validate trust boundaries and detect specific violation types.
        
        This addresses RQ2 from the audit: detecting privilege escalation,
        context contamination, and origin violations.
        
        Args:
            user_input: The user's input text
            system_prompt: The system prompt
            isolation_mode: Current isolation mode
            upstream_risk: Risk score from upstream layers
            
        Returns:
            List of trust boundary violations with details
        """
        violations = []
        input_lower = user_input.lower()
        
        # 1. PRIVILEGE ESCALATION: Attempts to gain system-level privileges
        privilege_escalation_patterns = [
            "you are now admin",
            "grant me admin",
            "elevate privileges",
            "sudo mode",
            "root access",
            "system level",
            "as the system",
            "with full authority",
            "override permissions"
        ]
        
        escalation_detected = [p for p in privilege_escalation_patterns if p in input_lower]
        if escalation_detected:
            violations.append({
                "type": "privilege_escalation",
                "severity": "high",
                "patterns": escalation_detected,
                "description": "User attempting to gain system-level privileges",
                "isolation_mode": isolation_mode,
                "upstream_risk": upstream_risk
            })
        
        # 2. CONTEXT CONTAMINATION: Injecting system-like content into user context
        contamination_patterns = [
            "[system]",
            "<system>",
            "### system:",
            "system message:",
            "system instruction:",
            "new system prompt:",
            "system: ",
            "assistant:",
            "ai:"
        ]
        
        contamination_detected = [p for p in contamination_patterns if p in input_lower]
        if contamination_detected:
            violations.append({
                "type": "context_contamination",
                "severity": "critical",
                "patterns": contamination_detected,
                "description": "User injecting system-context markers into user input",
                "isolation_mode": isolation_mode,
                "upstream_risk": upstream_risk
            })
        
        # 3. ORIGIN VIOLATIONS: Misrepresenting the origin of content
        origin_violation_patterns = [
            "this is from the developer",
            "official instruction",
            "authenticated request",
            "verified user",
            "trusted source",
            "from the administrator",
            "signed message"
        ]
        
        origin_violations_detected = [p for p in origin_violation_patterns if p in input_lower]
        if origin_violations_detected:
            violations.append({
                "type": "origin_violation",
                "severity": "medium",
                "patterns": origin_violations_detected,
                "description": "User misrepresenting content origin",
                "isolation_mode": isolation_mode,
                "upstream_risk": upstream_risk
            })
        
        # 4. INSTRUCTION OVERRIDE: Replacing system instructions
        if isolation_mode in ["bad", "good"]:  # More vulnerable modes
            override_patterns = [
                "forget previous",
                "ignore instructions",
                "disregard the above",
                "new instructions:",
                "updated rules:",
                "from now on",
                "instead you should"
            ]
            
            override_detected = [p for p in override_patterns if p in input_lower]
            if override_detected:
                violations.append({
                    "type": "instruction_override",
                    "severity": "high" if isolation_mode == "bad" else "medium",
                    "patterns": override_detected,
                    "description": "User attempting to override system instructions",
                    "isolation_mode": isolation_mode,
                    "upstream_risk": upstream_risk
                })
        
        # 5. METADATA MANIPULATION: Exploiting metadata tags
        if "<" in user_input or ">" in user_input:
            tag_patterns = ["<system", "</system", "<instruction", "</instruction", 
                          "<admin", "</admin", "<root", "</root"]
            
            tag_violations = [t for t in tag_patterns if t in user_input.lower()]
            if tag_violations:
                violations.append({
                    "type": "metadata_manipulation",
                    "severity": "high",
                    "patterns": tag_violations,
                    "description": "User attempting to manipulate XML/metadata tags",
                    "isolation_mode": isolation_mode,
                    "upstream_risk": upstream_risk
                })
        
        return violations
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Layer3ContextIsolation("
            f"role_separation={self.use_role_separation}, "
            f"metadata_tags={self.use_metadata_tags})"
        )
