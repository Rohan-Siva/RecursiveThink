from dataclasses import dataclass, field, asdict
from typing import Optional
import copy


@dataclass
class ThoughtState:
    problem: str
    current_solution: str = ""
    open_questions: str = ""
    confidence: float = 0.0
    step: int = 0
    
    def __post_init__(self):
        if not isinstance(self.problem, str):
            raise TypeError("problem must be a string")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.step < 0:
            raise ValueError("step must be non-negative")
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "ThoughtState":
        return cls(
            problem=data.get("problem", ""),
            current_solution=data.get("current_solution", ""),
            open_questions=data.get("open_questions", ""),
            confidence=float(data.get("confidence", 0.0)),
            step=int(data.get("step", 0))
        )
    
    def copy(self) -> "ThoughtState":
        return copy.deepcopy(self)
    
    def update_from_model_output(self, updated_state: dict) -> "ThoughtState":
        new_state = self.copy()
        new_state.step = self.step + 1
        
        if "current_solution" in updated_state:
            new_state.current_solution = str(updated_state["current_solution"])
        if "open_questions" in updated_state:
            new_state.open_questions = str(updated_state["open_questions"])
        if "confidence" in updated_state:
            conf = float(updated_state["confidence"])
            new_state.confidence = max(0.0, min(1.0, conf))
            
        return new_state
    
    def similarity_to(self, other: "ThoughtState") -> float:
        if not isinstance(other, ThoughtState):
            return 0.0
            
        solution_match = 1.0 if self.current_solution == other.current_solution else 0.0
        questions_match = 1.0 if self.open_questions == other.open_questions else 0.0
        confidence_diff = abs(self.confidence - other.confidence)
        confidence_match = 1.0 - confidence_diff
        
        return 0.5 * solution_match + 0.3 * questions_match + 0.2 * confidence_match
    
    def __repr__(self) -> str:
        return (
            f"ThoughtState(step={self.step}, confidence={self.confidence:.2f}, "
            f"solution_len={len(self.current_solution)}, "
            f"questions_len={len(self.open_questions)})"
        )
