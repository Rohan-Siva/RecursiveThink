"""
Sampling-based uncertainty estimation for recursive reasoning.

Instead of trusting the model's self-reported confidence, we measure it
empirically by generating multiple samples and checking consistency.
Inspired by LaMSeI (Pang et al., TMLR 2025) and self-consistency (Wang et al., 2022).

Core idea: If the model gives the same answer across N independent samples,
it's confident. If answers scatter, it's uncertain.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from parser import parse_model_output


def extract_boxed_answer(solution: str) -> Optional[str]:
    match = re.search(r"\\+boxed\{([^}]+)\}", solution)
    if match:
        return match.group(1).strip()
    nums = re.findall(r"\d[\d,]*\.?\d*", solution)
    if nums:
        return nums[-1].strip().rstrip(".,")
    return solution.strip()


def normalize_answer(answer: str) -> str:
    answer = answer.strip().lower()
    answer = answer.replace(",", "").replace(" ", "")
    try:
        return str(float(answer))
    except ValueError:
        return answer


def normalized_comparator(a: str, b: str) -> bool:
    return normalize_answer(a) == normalize_answer(b)


@dataclass
class UncertaintyResult:
    measured_confidence: float
    num_samples: int
    answers: List[Optional[str]]
    answer_counts: dict
    majority_answer: Optional[str]
    majority_solution: Optional[str]
    agreement_ratio: float
    self_reported_confidences: List[float]


class UncertaintyEstimator:
    """Estimates confidence by sampling multiple responses and measuring agreement."""

    def __init__(
        self,
        model,
        num_samples: int = 5,
        answer_extractor=None,
        answer_comparator=None,
    ):
        """
        Args:
            model: The LLM wrapper (MistralWrapper or GeminiWrapper).
            num_samples: Number of independent samples to generate per step.
            answer_extractor: Function(solution_text) -> extracted_answer.
                              If None, uses raw current_solution field.
            answer_comparator: Function(answer_a, answer_b) -> bool.
                               If None, uses exact string match.
        """
        self.model = model
        self.num_samples = num_samples
        self.answer_extractor = answer_extractor
        self.answer_comparator = answer_comparator or (lambda a, b: a == b)

    def estimate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Tuple[UncertaintyResult, list]:
        """
        Generate num_samples responses, parse them, and measure agreement.

        Returns:
            (UncertaintyResult, list of ParseResults) - the uncertainty measurement
            and all parsed samples for downstream use.
        """
        samples = []
        answers = []
        solutions = []
        self_reported_confidences = []

        for _ in range(self.num_samples):
            response = self.model.generate(system_prompt, user_prompt, json_mode=True)
            parse_result = parse_model_output(response.text)
            samples.append(parse_result)

            if parse_result.success:
                solution = parse_result.updated_state.get("current_solution", "")
                self_conf = float(parse_result.updated_state.get("confidence", 0.0))
                self_reported_confidences.append(self_conf)

                # Extract answer from solution
                if self.answer_extractor:
                    answer = self.answer_extractor(solution)
                else:
                    answer = solution
                answers.append(answer)
                solutions.append(solution)
            else:
                answers.append(None)
                solutions.append(None)
                self_reported_confidences.append(0.0)

        # Compute agreement
        result = self._compute_agreement(answers, solutions, self_reported_confidences)
        return result, samples

    def _compute_agreement(
        self,
        answers: List[Optional[str]],
        solutions: List[Optional[str]],
        self_reported_confidences: List[float],
    ) -> UncertaintyResult:
        """Compute confidence from answer agreement across samples."""
        # Filter out None (parse failures)
        valid_answers = [a for a in answers if a is not None]

        if not valid_answers:
            return UncertaintyResult(
                measured_confidence=0.0,
                num_samples=self.num_samples,
                answers=answers,
                answer_counts={},
                majority_answer=None,
                majority_solution=None,
                agreement_ratio=0.0,
                self_reported_confidences=self_reported_confidences,
            )

        # Group answers into equivalence classes using the comparator
        clusters = []  # list of (canonical_answer, [indices])
        for i, ans in enumerate(valid_answers):
            matched = False
            for cluster in clusters:
                if self.answer_comparator(ans, cluster[0]):
                    cluster[1].append(i)
                    matched = True
                    break
            if not matched:
                clusters.append((ans, [i]))

        # Find the largest cluster
        clusters.sort(key=lambda c: len(c[1]), reverse=True)
        largest_cluster = clusters[0]
        majority_answer = largest_cluster[0]
        majority_count = len(largest_cluster[1])

        # Agreement ratio: fraction of valid samples that agree with majority
        agreement_ratio = majority_count / len(valid_answers)

        # Measured confidence: agreement among all samples (including failures)
        measured_confidence = majority_count / self.num_samples

        # Find the solution corresponding to the majority answer
        # Pick the first valid solution from the majority cluster
        majority_idx = largest_cluster[1][0]
        valid_solutions = [s for s in solutions if s is not None]
        majority_solution = valid_solutions[majority_idx] if majority_idx < len(valid_solutions) else None

        # Build answer counts
        answer_counts = {}
        for canonical, indices in clusters:
            label = str(canonical) if canonical is not None else "None"
            answer_counts[label] = len(indices)
        failed = sum(1 for a in answers if a is None)
        if failed > 0:
            answer_counts["_parse_failed"] = failed

        return UncertaintyResult(
            measured_confidence=round(measured_confidence, 4),
            num_samples=self.num_samples,
            answers=answers,
            answer_counts=answer_counts,
            majority_answer=majority_answer,
            majority_solution=majority_solution,
            agreement_ratio=round(agreement_ratio, 4),
            self_reported_confidences=self_reported_confidences,
        )
