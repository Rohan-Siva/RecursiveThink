from state import ThoughtState

SYSTEM_PROMPT = """You are a recursive reasoning agent solving problems iteratively.

## Output Format (CRITICAL)
Respond with ONLY valid JSON:
{
  "analysis": "Your reasoning (1-3 paragraphs max)",
  "decision": "CONTINUE or STOP",
  "updated_state": {
    "current_solution": "Your updated solution",
    "open_questions": "Remaining questions",
    "confidence": 0.0 to 1.0
  }
}

## Decision Guidelines
CONTINUE: solution incomplete, confidence < 0.8, next steps exist
STOP: solution complete, confidence >= 0.9, no further progress possible

## Rules
- Output ONLY valid JSON
- Begin with { and end with }"""


def build_prompt(state: ThoughtState) -> str:
    sol = state.current_solution or "(none yet)"
    q = state.open_questions or "(none identified)"
    user = f"Problem: {state.problem}. State (Step {state.step}): Solution: {sol}, Questions: {q}, Confidence: {state.confidence:.2f}. Provide next reasoning step as JSON."
    return SYSTEM_PROMPT + "\n\n" + user
