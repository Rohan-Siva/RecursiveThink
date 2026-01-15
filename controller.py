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
    max_steps: int = 10
    confidence_threshold: float = 0.9
    min_change_threshold: float = 0.01
    max_parse_retries: int = 2


class Controller:
    
    def __init__(
        self,
        model,
        config: ControllerConfig,
        logger: ReasoningLogger
    ):
        self.model = model
        self.config = config
        self.logger = logger
        self.state_history: List[ThoughtState] = []
    
    def run(self, problem: str) -> ThoughtState:
        start_time = time.time()
        
        state = ThoughtState(problem=problem)
        self.state_history = [state.copy()]
        
        stop_reason = None
        
        while True:
            system_prompt, user_prompt = build_prompt(state)
            
            parse_result = None
            true_confidence = None
            for retry in range(self.config.max_parse_retries + 1):
                model_response = self.model.generate(system_prompt, user_prompt)
                raw_output = model_response.text
                true_confidence = model_response.true_confidence
                parse_result = parse_model_output(raw_output)
                
                if parse_result.success:
                    break
                    
                if retry < self.config.max_parse_retries:
                    print(f"Parse failed (retry {retry + 1}): {parse_result.error}")
            
            if parse_result.success:
                decision = parse_result.decision
                new_state = state.update_from_model_output(parse_result.updated_state)
            else:
                decision = "CONTINUE"
                new_state = state.copy()
                new_state.step += 1
            
            should_stop, stop_reason = self._should_stop(
                new_state, state, decision, parse_result
            )
            
            step_log = StepLog(
                timestamp=datetime.now().isoformat(),
                step=state.step,
                state_before=state.to_dict(),
                model_output=parse_result.raw_output if parse_result else "",
                parse_success=parse_result.success if parse_result else False,
                parse_error=parse_result.error if parse_result else "No output",
                decision=decision,
                stop_reason=stop_reason if should_stop else None,
                state_after=new_state.to_dict(),
                true_confidence=true_confidence
            )
            self.logger.log_step(step_log)
            
            print(f"Step {new_state.step}: decision={decision}, "
                  f"confidence={new_state.confidence:.2f}")
            
            state = new_state
            self.state_history.append(state.copy())
            
            if should_stop:
                break
        
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
        if new_state.step >= self.config.max_steps:
            return True, "max_steps_reached"
        
        if decision == "STOP":
            return True, "model_decided_stop"
        
        if new_state.confidence >= self.config.confidence_threshold:
            return True, "confidence_threshold_reached"
        
        if self._detect_stagnation(new_state, prev_state):
            return True, "state_stagnation"
        
        if self._detect_loop():
            return True, "loop_detected"
        
        return False, ""
    
    def _detect_stagnation(
        self,
        new_state: ThoughtState,
        prev_state: ThoughtState
    ) -> bool:
        similarity = new_state.similarity_to(prev_state)
        return similarity > (1.0 - self.config.min_change_threshold)
    
    def _detect_loop(self) -> bool:
        if len(self.state_history) < 3:
            return False
        
        recent = self.state_history[-3:]
        for i, s1 in enumerate(recent):
            for s2 in recent[i+1:]:
                if s1.similarity_to(s2) > 0.95:
                    return True
        
        return False
