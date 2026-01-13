"""
controller.py - Recursive Reasoning Controller
"""
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, List, Optional

from state import ThoughtState
from prompt import build_prompt
from parser import parse_model_output, ParseResult
from logger import ReasoningLogger, StepLog


@dataclass
class ControllerConfig:
    """Configuration for the reasoning controller."""
    max_steps: int = 10
    confidence_threshold: float = 0.9
    min_change_threshold: float = 0.01
    max_parse_retries: int = 2


class Controller:
    """
    Controls the recursive reasoning loop.
    
    Responsibilities:
    - Call model and parse outputs
    - Update ThoughtState
    - Enforce stopping conditions
    - Log all steps
    """
    
    def __init__(
        self,
        model,
        config: ControllerConfig,
        logger: ReasoningLogger
    ):
        """
        Initialize the controller.
        
        Args:
            model: ModelWrapper instance
            config: Controller configuration
            logger: ReasoningLogger instance
        """
        self.model = model
        self.config = config
        self.logger = logger
        self.state_history: List[ThoughtState] = []
    
    def run(self, problem: str) -> ThoughtState:
        """
        Run the recursive reasoning loop on a problem.
        
        Args:
            problem: The problem to solve
            
        Returns:
            Final ThoughtState after reasoning completes
        """
        start_time = time.time()
        
        # Initialize state
        state = ThoughtState(problem=problem)
        self.state_history = [state.copy()]
        
        stop_reason = None
        
        while True:
            # Build prompt from current state
            prompt = build_prompt(state)
            
            # Generate model response (with retries)
            parse_result = None
            for retry in range(self.config.max_parse_retries + 1):
                raw_output = self.model.generate(prompt)
                parse_result = parse_model_output(raw_output)
                
                if parse_result.success:
                    break
                    
                if retry < self.config.max_parse_retries:
                    print(f"Parse failed (retry {retry + 1}): {parse_result.error}")
            
            # Determine decision and update state
            if parse_result.success:
                decision = parse_result.decision
                new_state = state.update_from_model_output(parse_result.updated_state)
            else:
                # Parse failed - treat as CONTINUE with no state change
                decision = "CONTINUE"
                new_state = state.copy()
                new_state.step += 1
            
            # Check stopping conditions
            should_stop, stop_reason = self._should_stop(
                new_state, state, decision, parse_result
            )
            
            # Log this step
            step_log = StepLog(
                timestamp=datetime.now().isoformat(),
                step=state.step,
                state_before=state.to_dict(),
                model_output=parse_result.raw_output if parse_result else "",
                parse_success=parse_result.success if parse_result else False,
                parse_error=parse_result.error if parse_result else "No output",
                decision=decision,
                stop_reason=stop_reason if should_stop else None,
                state_after=new_state.to_dict()
            )
            self.logger.log_step(step_log)
            
            # Print progress
            print(f"Step {new_state.step}: decision={decision}, "
                  f"confidence={new_state.confidence:.2f}")
            
            # Update state
            state = new_state
            self.state_history.append(state.copy())
            
            if should_stop:
                break
        
        # Log summary
        elapsed = time.time() - start_time
        self.logger.log_summary(
            final_state=state.to_dict(),
            total_steps=state.step,
            stop_reason=stop_reason,
            elapsed_time=elapsed
        )
        
        return state
    
    def _should_stop(
        self,
        new_state: ThoughtState,
        prev_state: ThoughtState,
        decision: str,
        parse_result: Optional[ParseResult]
    ) -> Tuple[bool, str]:
        """
        Determine if reasoning should stop.
        
        Returns:
            Tuple of (should_stop, reason)
        """
        # Check max steps
        if new_state.step >= self.config.max_steps:
            return True, "max_steps_reached"
        
        # Check model decision
        if decision == "STOP":
            return True, "model_decided_stop"
        
        # Check confidence threshold
        if new_state.confidence >= self.config.confidence_threshold:
            return True, "confidence_threshold_reached"
        
        # Check for stagnation
        if self._detect_stagnation(new_state, prev_state):
            return True, "state_stagnation"
        
        # Check for loops
        if self._detect_loop():
            return True, "loop_detected"
        
        return False, ""
    
    def _detect_stagnation(
        self,
        new_state: ThoughtState,
        prev_state: ThoughtState
    ) -> bool:
        """Detect if state has stopped changing meaningfully."""
        similarity = new_state.similarity_to(prev_state)
        return similarity > (1.0 - self.config.min_change_threshold)
    
    def _detect_loop(self) -> bool:
        """Detect if we are in a reasoning loop."""
        if len(self.state_history) < 3:
            return False
        
        # Check if recent states are too similar
        recent = self.state_history[-3:]
        for i, s1 in enumerate(recent):
            for s2 in recent[i+1:]:
                if s1.similarity_to(s2) > 0.95:
                    return True
        
        return False
