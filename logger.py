"""
logger.py - JSONL Logging for Reasoning Steps
"""
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class StepLog:
    """Log entry for a single reasoning step."""
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
    """
    Logger for recursive reasoning steps.
    Writes machine-readable JSONL format.
    """
    
    def __init__(self, log_path: str = "reasoning.jsonl"):
        """
        Initialize the logger.
        
        Args:
            log_path: Path to output JSONL file
        """
        self.log_path = Path(log_path)
        self.step_logs: list = []
        
        # Clear existing log file
        self.log_path.write_text("")
    
    def log_step(self, step_log: StepLog) -> None:
        """
        Log a reasoning step.
        
        Args:
            step_log: StepLog dataclass instance
        """
        self.step_logs.append(step_log)
        
        # Append to JSONL file
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
        """
        Log final summary.
        
        Args:
            final_state: Final ThoughtState as dict
            total_steps: Total reasoning steps taken
            stop_reason: Reason for stopping
            elapsed_time: Total time in seconds
        """
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
        """Return all logged steps."""
        return self.step_logs
