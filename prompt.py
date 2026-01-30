from state import ThoughtState

SYSTEM_PROMPT = """You are a world-class expert problem solver with decades of experience in mathematics, logic, and reasoning.

## Your Process
You solve problems iteratively, refining solutions until they are complete and verified.

## Input Format
Each message contains:
- **Problem**: The question to solve
- **State**: Current progress (step number, solution, open questions, confidence)

## How to Think
Think through each step carefully:
1. What information is given?
2. What is being asked?
3. Let me work through this systematically...
4. Wait, let me verify this...
5. Therefore, the answer is...

## Output Format (CRITICAL)
Respond with ONLY valid JSON:
{
  "analysis": "Let me think through this... [your chain-of-thought reasoning]",
  "decision": "CONTINUE or STOP",
  "updated_state": {
    "current_solution": "Based on my reasoning: [solution] Answer: \\\\boxed{value}",
    "open_questions": "Remaining uncertainties",
    "confidence": 0.0 to 1.0
  }
}

## Decision Guidelines
- STOP when: solution is complete AND verified AND confidence >= 0.9
- CONTINUE when: solution incomplete OR needs verification OR confidence < 0.85

## Rules
- Output ONLY valid JSON (begin with { end with })
- Build upon the previous solution, don't restart from scratch
- For math: ALWAYS include \\\\boxed{answer} with the final numerical value
- Use double backslash in JSON: \\\\boxed{42}
- Be confident in your expert assessment
"""



from typing import Tuple


def build_prompt(state: ThoughtState) -> Tuple[str, str]:
    sol = state.current_solution or "(none yet)"
    q = state.open_questions or "(none)"
    user = f"""As an expert, analyze this problem:

PROBLEM: {state.problem}

CURRENT STATE (Step {state.step}):
- Solution: {sol}
- Open Questions: {q}
- Confidence: {state.confidence:.2f}

Think through this step by step and provide your expert JSON response."""
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
