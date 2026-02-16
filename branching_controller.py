import time
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple, Dict
from pathlib import Path

from state import ThoughtState
from subgoal import SubgoalState, BranchResult
from decompose import decompose_problem, topological_order
from branching_prompts import (
    build_subgoal_prompt,
    build_verify_prompt,
    build_assembly_prompt,
)
from uncertainty import UncertaintyEstimator, extract_boxed_answer, normalized_comparator
from parser import parse_model_output
from logger import ReasoningLogger


@dataclass
class BranchingConfig:
    branch_threshold: float = 0.8
    num_samples: int = 5
    max_branches: int = 3
    branch_verification_steps: int = 2
    final_check_samples: int = 5


class BranchingController:

    def __init__(self, model, config: BranchingConfig, logger: ReasoningLogger):
        self.model = model
        self.config = config
        self.logger = logger
        self.estimator = UncertaintyEstimator(
            model=model,
            num_samples=config.num_samples,
            answer_extractor=extract_boxed_answer,
            answer_comparator=normalized_comparator,
        )
        self.api_calls = 0

    def run(self, problem: str) -> Tuple[ThoughtState, List[SubgoalState]]:
        start_time = time.time()

        # Phase 1: Decompose
        print("Phase 1: Decomposing problem...")
        self.api_calls += 1
        subgoals = decompose_problem(self.model, problem)
        ordered = topological_order(subgoals)
        by_id = {sg.subgoal_id: sg for sg in ordered}

        print(f"  Decomposed into {len(ordered)} subgoals:")
        for sg in ordered:
            deps = f" (deps: {sg.dependencies})" if sg.dependencies else ""
            print(f"    {sg.subgoal_id}. {sg.description}{deps}")

        self._log_event("decomposition", {
            "subgoals": [sg.to_dict() for sg in ordered],
        })

        # Phase 2 + 3: Solve each subgoal with uncertainty, branch if needed
        resolved_results: Dict[int, str] = {}

        for sg in ordered:
            print(f"\nPhase 2: Solving subgoal {sg.subgoal_id}: {sg.description[:80]}...")

            dep_results = {d: resolved_results.get(d, "") for d in sg.dependencies}
            neg_knowledge = []
            for d in sg.dependencies:
                dep_sg = by_id.get(d)
                if dep_sg and dep_sg.negative_knowledge:
                    neg_knowledge.extend(dep_sg.negative_knowledge)

            system_prompt, user_prompt = build_subgoal_prompt(
                problem, sg, dep_results, neg_knowledge if neg_knowledge else None,
            )

            uncertainty_result, samples = self.estimator.estimate(system_prompt, user_prompt)
            self.api_calls += self.config.num_samples

            sg.confidence = uncertainty_result.measured_confidence
            sg.result = uncertainty_result.majority_answer

            print(f"  Confidence: {sg.confidence:.2f}, "
                  f"agreement: {uncertainty_result.agreement_ratio:.0%}, "
                  f"clusters: {uncertainty_result.answer_counts}")

            self._log_event("subgoal_solved", {
                "subgoal_id": sg.subgoal_id,
                "confidence": sg.confidence,
                "agreement_ratio": uncertainty_result.agreement_ratio,
                "answer_counts": uncertainty_result.answer_counts,
                "majority_answer": sg.result,
                "self_reported": uncertainty_result.self_reported_confidences,
            })

            # Phase 3: Branch if uncertain
            if sg.confidence < self.config.branch_threshold:
                print(f"  ⚡ Low confidence ({sg.confidence:.2f}) — branching...")
                self._branch_and_verify(problem, sg, dep_results, uncertainty_result)
                print(f"  ✓ After branching: confidence={sg.confidence:.2f}, answer={sg.result}")

            if sg.result:
                # Use the full solution text from the majority sample for downstream context
                majority_solution = uncertainty_result.majority_solution or sg.result
                resolved_results[sg.subgoal_id] = majority_solution
            else:
                resolved_results[sg.subgoal_id] = "(unsolved)"

        # Phase 4: Assemble final solution
        print("\nPhase 4: Assembling final solution...")
        system_prompt, user_prompt = build_assembly_prompt(problem, resolved_results)
        assembly_response = self.model.generate(system_prompt, user_prompt, json_mode=True)
        self.api_calls += 1
        assembly_parse = parse_model_output(assembly_response.text)

        if assembly_parse.success:
            assembled_solution = assembly_parse.updated_state.get("current_solution", "")
        else:
            assembled_solution = " | ".join(
                f"Subgoal {k}: {v}" for k, v in sorted(resolved_results.items())
            )

        # Phase 5: Final consistency check
        print("\nPhase 5: Final consistency check...")
        from prompt import build_prompt
        final_state = ThoughtState(problem=problem, current_solution=assembled_solution)
        fsys, fuser = build_prompt(final_state)
        final_result, _ = self.estimator.estimate(fsys, fuser)
        self.api_calls += self.config.final_check_samples

        final_confidence = final_result.measured_confidence
        final_answer = final_result.majority_answer
        if final_result.majority_solution:
            assembled_solution = final_result.majority_solution

        print(f"  Final confidence: {final_confidence:.2f}, "
              f"agreement: {final_result.agreement_ratio:.0%}, "
              f"clusters: {final_result.answer_counts}")

        self._log_event("final_check", {
            "confidence": final_confidence,
            "agreement_ratio": final_result.agreement_ratio,
            "answer_counts": final_result.answer_counts,
            "majority_answer": final_answer,
        })

        elapsed = time.time() - start_time

        final_state = ThoughtState(
            problem=problem,
            current_solution=assembled_solution,
            confidence=min(1.0, max(0.0, final_confidence)),
            step=len(ordered),
        )

        self._log_event("summary", {
            "total_subgoals": len(ordered),
            "total_api_calls": self.api_calls,
            "elapsed_time": round(elapsed, 2),
            "final_confidence": final_confidence,
            "final_answer": final_answer,
            "subgoals": [sg.to_dict() for sg in ordered],
        })

        return final_state, ordered

    def _branch_and_verify(
        self,
        problem: str,
        sg: SubgoalState,
        dep_results: Dict[int, str],
        uncertainty_result,
    ):
        clusters = uncertainty_result.answer_counts.copy()
        clusters.pop("_parse_failed", None)

        sorted_clusters = sorted(clusters.items(), key=lambda x: x[1], reverse=True)
        branch_candidates = sorted_clusters[:self.config.max_branches]

        branches: List[BranchResult] = []

        for idx, (answer, count) in enumerate(branch_candidates):
            branch = BranchResult(
                branch_id=idx,
                answer=answer,
                sample_count=count,
            )

            print(f"    Branch {idx}: answer='{answer}' ({count} samples) — verifying...")

            sys_prompt, usr_prompt = build_verify_prompt(
                problem, sg, answer, dep_results,
            )

            verify_confidences = []
            for step in range(self.config.branch_verification_steps):
                response = self.model.generate(sys_prompt, usr_prompt, json_mode=True)
                self.api_calls += 1
                parse_result = parse_model_output(response.text)

                if parse_result.success:
                    conf = float(parse_result.updated_state.get("confidence", 0.0))
                    verify_confidences.append(conf)
                    solution = parse_result.updated_state.get("current_solution", "")
                    branch.solution_text = solution

                    if conf < 0.3:
                        oq = parse_result.updated_state.get("open_questions", "")
                        branch.failure_reason = oq or f"Verification found issues with answer '{answer}'"
                        break
                else:
                    verify_confidences.append(0.0)

            branch.verification_confidence = (
                sum(verify_confidences) / len(verify_confidences)
                if verify_confidences else 0.0
            )

            print(f"      Verification confidence: {branch.verification_confidence:.2f}")
            branches.append(branch)

        if not branches:
            return

        branches.sort(key=lambda b: (b.verification_confidence, b.sample_count), reverse=True)
        winner = branches[0]
        winner.is_winner = True

        sg.result = winner.answer
        sg.confidence = winner.verification_confidence
        sg.branches = branches

        for b in branches:
            if not b.is_winner and b.failure_reason:
                sg.negative_knowledge.append(b.failure_reason)

        self._log_event("branching_result", {
            "subgoal_id": sg.subgoal_id,
            "branches": [
                {
                    "branch_id": b.branch_id,
                    "answer": b.answer,
                    "sample_count": b.sample_count,
                    "verification_confidence": b.verification_confidence,
                    "is_winner": b.is_winner,
                    "failure_reason": b.failure_reason,
                }
                for b in branches
            ],
            "winner_answer": winner.answer,
            "negative_knowledge": sg.negative_knowledge,
        })

    def _log_event(self, event_type: str, data: dict):
        entry = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            **data,
        }
        with open(self.logger.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
