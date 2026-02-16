from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BranchResult:
    branch_id: int
    answer: str
    sample_count: int
    verification_confidence: float = 0.0
    is_winner: bool = False
    failure_reason: Optional[str] = None
    solution_text: Optional[str] = None


@dataclass
class SubgoalState:
    subgoal_id: int
    description: str
    dependencies: List[int] = field(default_factory=list)
    result: Optional[str] = None
    confidence: float = 0.0
    branches: List[BranchResult] = field(default_factory=list)
    negative_knowledge: List[str] = field(default_factory=list)
    raw_samples: Optional[list] = None

    @property
    def is_resolved(self) -> bool:
        return self.result is not None and self.confidence > 0

    def to_dict(self) -> dict:
        return {
            "subgoal_id": self.subgoal_id,
            "description": self.description,
            "dependencies": self.dependencies,
            "result": self.result,
            "confidence": self.confidence,
            "negative_knowledge": self.negative_knowledge,
            "branches": [
                {
                    "branch_id": b.branch_id,
                    "answer": b.answer,
                    "sample_count": b.sample_count,
                    "verification_confidence": b.verification_confidence,
                    "is_winner": b.is_winner,
                    "failure_reason": b.failure_reason,
                }
                for b in self.branches
            ],
        }
