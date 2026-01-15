import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class StepLog:
    timestamp: str
    step: int
    state_before: Dict[str, Any]
    model_output: str
    parse_success: bool
    parse_error: Optional[str]
    decision: Optional[str]
    stop_reason: Optional[str]
    state_after: Dict[str, Any]


class ReasoningLogger:
    
    def __init__(self, log_path: str = "reasoning.jsonl"):
        self.log_path = Path(log_path)
        self.step_logs: list = []
        self.log_path.write_text("")
    
    def log_step(self, step_log: StepLog) -> None:
        self.step_logs.append(step_log)
        
        with open(self.log_path, "a") as f:
            log_dict = asdict(step_log)
            f.write(json.dumps(log_dict) + "\n")
    
    def log_summary(
        self,
        final_state: Dict[str, Any],
        total_steps: int,
        stop_reason: str,
        elapsed_time: float
    ) -> None:
        summary = {
            "type": "summary",
            "timestamp": datetime.now().isoformat(),
            "total_steps": total_steps,
            "stop_reason": stop_reason,
            "elapsed_time_seconds": round(elapsed_time, 2),
            "final_state": final_state
        }
        
        with open(self.log_path, "a") as f:
            f.write(json.dumps(summary) + "\n")
    
    def get_logs(self) -> list:
        return self.step_logs
