import time
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, List, Optional
import json

from state import ThoughtState
from prompt import build_prompt, build_critic_prompt
from parser import parse_model_output, ParseResult
from logger import ReasoningLogger, StepLog


@dataclass
class ControllerConfig:
    max_steps: int = 10
    confidence_threshold: float = 0.9
    min_change_threshold: float = 0.01
    max_parse_retries: int = 2
    use_critic: bool = False


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
                model_response = self.model.generate(system_prompt, user_prompt, json_mode=True)
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
            
            if self.config.use_critic:
                should_stop, stop_reason, critic_verdict = self._critic_evaluate(new_state)
                if critic_verdict:
                    decision = critic_verdict
            else:
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
            
            critic_indicator = " [critic]" if self.config.use_critic else ""
            print(f"Step {new_state.step}: decision={decision}{critic_indicator}, "
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
    
    def _critic_evaluate(self, state: ThoughtState) -> Tuple[bool, str, Optional[str]]:
        if state.step >= self.config.max_steps:
            return True, "max_steps_reached", None
        
        system_prompt, user_prompt = build_critic_prompt(state)
        
        try:
            model_response = self.model.generate(system_prompt, user_prompt, json_mode=True)
            raw_output = model_response.text
            
            raw_output = raw_output.replace("```json", "").replace("```", "")
            start = raw_output.find("{")
            end = raw_output.rfind("}") + 1
            if start != -1 and end > start:
                json_str = raw_output[start:end]
                critic_result = json.loads(json_str)
                
                verdict = critic_result.get("verdict", "CONTINUE").upper()
                
                if verdict == "STOP":
                    return True, "critic_approved_stop", "STOP"
                else:
                    return False, "", "CONTINUE"
            else:
                return False, "", None
                
        except Exception:
            return False, "", None
    
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
