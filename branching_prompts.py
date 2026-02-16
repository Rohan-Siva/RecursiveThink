from typing import Tuple

from state import ThoughtState
from subgoal import SubgoalState


DECOMPOSE_SYSTEM = """You are an expert problem decomposer. Given a math problem, break it into small, ordered subgoals.

## Output Format (CRITICAL)
Respond with ONLY valid JSON:
{
  "subgoals": [
    {"id": 1, "description": "First step description", "dependencies": []},
    {"id": 2, "description": "Second step description", "dependencies": [1]},
    {"id": 3, "description": "Third step description", "dependencies": [2]}
  ]
}

## Rules
- Each subgoal should be a single, well-defined mathematical step
- Use 3-6 subgoals for most problems (fewer for simple, more for complex)
- Dependencies list which subgoals must be completed first
- The final subgoal should compute the answer
- Output ONLY valid JSON (begin with { end with })
"""


def build_decompose_prompt(problem: str) -> Tuple[str, str]:
    user = f"""Break this problem into ordered subgoals:

PROBLEM: {problem}

Provide your decomposition as JSON."""
    return DECOMPOSE_SYSTEM, user


SUBGOAL_SOLVE_SYSTEM = """You are a world-class expert problem solver.

## Your Task
Solve ONE specific subgoal of a larger problem. You are given the original problem, the subgoal to solve, and results from previous subgoals.

## Output Format (CRITICAL)
Respond with ONLY valid JSON:
{{
  "analysis": "Your step-by-step reasoning for this subgoal",
  "decision": "CONTINUE or STOP",
  "updated_state": {{
    "current_solution": "Your solution for THIS subgoal. For math: include \\\\boxed{{answer}}",
    "open_questions": "Any remaining uncertainties about this subgoal",
    "confidence": 0.0 to 1.0
  }}
}}

## Rules
- Focus ONLY on the current subgoal
- Use results from previous subgoals as given facts
- Output ONLY valid JSON (begin with {{ end with }})
- For math: ALWAYS include \\\\boxed{{answer}} with the subgoal's result
"""


def build_subgoal_prompt(
    problem: str,
    subgoal: SubgoalState,
    dependency_results: dict,
    negative_knowledge: list = None,
) -> Tuple[str, str]:
    dep_text = ""
    if dependency_results:
        dep_lines = []
        for sg_id, result in sorted(dependency_results.items()):
            dep_lines.append(f"  Subgoal {sg_id}: {result}")
        dep_text = "\nPREVIOUS RESULTS (use as given facts):\n" + "\n".join(dep_lines)

    neg_text = ""
    if negative_knowledge:
        neg_lines = "\n".join(f"  - {nk}" for nk in negative_knowledge)
        neg_text = f"\nKNOWN DEAD ENDS (do not repeat these approaches):\n{neg_lines}"

    user = f"""ORIGINAL PROBLEM: {problem}

CURRENT SUBGOAL (#{subgoal.subgoal_id}): {subgoal.description}
{dep_text}{neg_text}

Solve this subgoal and provide your JSON response."""
    return SUBGOAL_SOLVE_SYSTEM, user


VERIFY_SYSTEM = """You are a critical verifier. Check whether a proposed solution to a subgoal is correct.

## Output Format (CRITICAL)
Respond with ONLY valid JSON:
{{
  "analysis": "Your verification reasoning",
  "decision": "STOP",
  "updated_state": {{
    "current_solution": "The verified (or corrected) solution. Include \\\\boxed{{answer}}",
    "open_questions": "Any issues found",
    "confidence": 0.0 to 1.0
  }}
}}

## Rules
- If the approach has a flaw, explain what went wrong and set confidence low
- If the approach is valid, confirm it and set confidence high
- Output ONLY valid JSON
"""


def build_verify_prompt(
    problem: str,
    subgoal: SubgoalState,
    proposed_answer: str,
    dependency_results: dict,
) -> Tuple[str, str]:
    dep_text = ""
    if dependency_results:
        dep_lines = []
        for sg_id, result in sorted(dependency_results.items()):
            dep_lines.append(f"  Subgoal {sg_id}: {result}")
        dep_text = "\nPREVIOUS RESULTS:\n" + "\n".join(dep_lines)

    user = f"""ORIGINAL PROBLEM: {problem}

CURRENT SUBGOAL (#{subgoal.subgoal_id}): {subgoal.description}
{dep_text}

PROPOSED ANSWER: {proposed_answer}

Verify this answer. Is it correct? Provide your JSON response."""
    return VERIFY_SYSTEM, user


ASSEMBLE_SYSTEM = """You are an expert problem solver assembling a final answer from solved subgoals.

## Output Format (CRITICAL)
Respond with ONLY valid JSON:
{{
  "analysis": "Brief summary of how the subgoal results combine",
  "decision": "STOP",
  "updated_state": {{
    "current_solution": "The complete final answer. Include \\\\boxed{{final_answer}}",
    "open_questions": "None",
    "confidence": 0.0 to 1.0
  }}
}}

## Rules
- Combine all subgoal results into one coherent final answer
- The final numerical answer MUST be in \\\\boxed{{}}
- Output ONLY valid JSON
"""


def build_assembly_prompt(
    problem: str,
    subgoal_results: dict,
) -> Tuple[str, str]:
    result_lines = []
    for sg_id, result in sorted(subgoal_results.items()):
        result_lines.append(f"  Subgoal {sg_id}: {result}")
    results_text = "\n".join(result_lines)

    user = f"""ORIGINAL PROBLEM: {problem}

SOLVED SUBGOALS:
{results_text}

Assemble these into a final answer. Provide your JSON response."""
    return ASSEMBLE_SYSTEM, user
