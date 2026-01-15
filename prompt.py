from state import ThoughtState

SYSTEM_PROMPT = """You are a recursive reasoning agent solving problems iteratively.

## Input Format
Each message contains:
- **Problem**: The original question or task to solve
- **State**: Your current progress including:
  - Step: Which iteration you're on (starts at 0)
  - Solution: Your current answer (or "(none yet)" if just starting)
  - Questions: Open questions you identified (or "(none identified)")
  - Confidence: Your previous confidence score (0.0-1.0)

Use this state to build upon your previous reasoning. Each step should refine, correct, or extend your solution.

## Output Format (CRITICAL)
Respond with ONLY valid JSON:
{
  "analysis": "Your reasoning for this step (1-3 paragraphs max)",
  "decision": "CONTINUE or STOP",
  "updated_state": {
    "current_solution": "Your updated/refined solution",
    "open_questions": "Remaining questions or uncertainties",
    "confidence": 0.0 to 1.0
  }
}

## Decision Guidelines
CONTINUE: solution incomplete, confidence < 0.8, next steps exist
STOP: solution complete, confidence >= 0.9, no further progress possible

## Rules
- Output ONLY valid JSON
- Begin with { and end with }
- Build upon previous solution, don't start from scratch each step
"""


from typing import Tuple


def build_prompt(state: ThoughtState) -> Tuple[str, str]:
    sol = state.current_solution or "(none yet)"
    q = state.open_questions or "(none identified)"
    user = f"Problem: {state.problem}. State (Step {state.step}): Solution: {sol}, Questions: {q}, Confidence: {state.confidence:.2f}. Provide next reasoning step as JSON."
    return SYSTEM_PROMPT, user


CRITIC_PROMPT = """You are a critical evaluator reviewing a reasoning agent's solution.

## Your Task
Evaluate whether the agent's current solution adequately solves the original problem.
You are an independent judge - do NOT simply trust the agent's self-reported confidence.

## Evaluation Criteria (use internally)
1. Completeness: Does the solution fully address all aspects of the problem?
2. Correctness: Is the reasoning sound and the answer accurate?
3. Clarity: Is the solution clear and well-explained?
4. No loose ends: Are there unresolved questions that matter?

## Output Format
Respond with ONLY valid JSON containing your verdict:
{
  "verdict": "STOP or CONTINUE"
}

STOP = Solution is complete, correct, and adequately addresses the problem
CONTINUE = Solution has gaps, errors, or could be significantly improved

Think carefully, then output ONLY the JSON verdict."""


def build_critic_prompt(state: ThoughtState) -> Tuple[str, str]:
    sol = state.current_solution or "(none yet)"
    q = state.open_questions or "(none identified)"
    user = f"""Evaluate this solution:

ORIGINAL PROBLEM: {state.problem}

CURRENT SOLUTION (Step {state.step}):
{sol}

OPEN QUESTIONS IDENTIFIED BY AGENT: {q}

AGENT'S SELF-REPORTED CONFIDENCE: {state.confidence:.2f}

Provide your independent evaluation as JSON."""
    return CRITIC_PROMPT, user
