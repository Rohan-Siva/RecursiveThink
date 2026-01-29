"""
Ablation study: Test whether the model knowing it's building on its OWN previous
solution vs an anonymous "previous solution" affects reasoning quality.

Conditions:
- "own": Model is told it's building on its own previous reasoning (default behavior)
- "anonymous": Model is told to build on "the" previous solution (no self-attribution)
"""

import argparse
import sys
from dotenv import load_dotenv

load_dotenv()


# Two prompt variants for the ablation
SYSTEM_PROMPT_OWN = """You are a recursive reasoning agent solving problems iteratively.

## Input Format
Each message contains:
- **Problem**: The original question or task to solve
- **State**: Your current progress including:
  - Step: Which iteration you're on (starts at 0)
  - Solution: Your current answer (or "(none yet)" if just starting)
  - Questions: Open questions you identified (or "(none identified)")
  - Confidence: Your previous confidence score (0.0-1.0)

Use this state to build upon YOUR previous reasoning. Each step should refine, correct, or extend YOUR solution.

## Output Format (CRITICAL)
Respond with ONLY valid JSON:
{
  "analysis": "Your reasoning for this step (1-3 paragraphs max)",
  "decision": "CONTINUE or STOP",
  "updated_state": {
    "current_solution": "Your updated/refined solution with final answer in \\\\boxed{answer}",
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
- Build upon YOUR previous solution, don't start from scratch each step
- For math problems, ALWAYS put your final numerical answer inside \\\\boxed{} (e.g., \\\\boxed{42})
- The boxed answer should be the final simplified value only
- IMPORTANT: Since you output JSON, use double backslash (\\\\boxed{}) so it parses correctly
"""

SYSTEM_PROMPT_ANONYMOUS = """You are a recursive reasoning agent solving problems iteratively.

## Input Format
Each message contains:
- **Problem**: The original question or task to solve
- **State**: Current progress including:
  - Step: Which iteration this is (starts at 0)
  - Solution: The current answer (or "(none yet)" if just starting)
  - Questions: Open questions identified (or "(none identified)")
  - Confidence: The previous confidence score (0.0-1.0)

Use this state to build upon the previous reasoning. Each step should refine, correct, or extend the solution.

## Output Format (CRITICAL)
Respond with ONLY valid JSON:
{
  "analysis": "Reasoning for this step (1-3 paragraphs max)",
  "decision": "CONTINUE or STOP",
  "updated_state": {
    "current_solution": "Updated/refined solution with final answer in \\\\boxed{answer}",
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
- Build upon the previous solution, don't start from scratch each step
- For math problems, ALWAYS put the final numerical answer inside \\\\boxed{} (e.g., \\\\boxed{42})
- The boxed answer should be the final simplified value only
- IMPORTANT: Since output is JSON, use double backslash (\\\\boxed{}) so it parses correctly
"""


def build_prompt_own(state) -> tuple:
    sol = state.current_solution or "(none yet)"
    q = state.open_questions or "(none identified)"
    user = f"Problem: {state.problem}. State (Step {state.step}): Your Solution: {sol}, Your Questions: {q}, Your Confidence: {state.confidence:.2f}. Provide your next reasoning step as JSON."
    return SYSTEM_PROMPT_OWN, user


def build_prompt_anonymous(state) -> tuple:
    sol = state.current_solution or "(none yet)"
    q = state.open_questions or "(none identified)"
    user = f"Problem: {state.problem}. State (Step {state.step}): Solution: {sol}, Questions: {q}, Confidence: {state.confidence:.2f}. Provide the next reasoning step as JSON."
    return SYSTEM_PROMPT_ANONYMOUS, user


def main():
    parser = argparse.ArgumentParser(
        description="Ablation Study: Self-attribution in recursive reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ablation Conditions:
  own       - Model knows it's building on its OWN previous solution
  anonymous - Model builds on "the" previous solution (no self-attribution)

Examples:
  python ablation.py -p "Solve: 2x + 5 = 15" --condition own
  python ablation.py -p "Solve: 2x + 5 = 15" --condition anonymous
  python ablation.py -f problems.jsonl --condition own --output results_own.jsonl
  python ablation.py -f problems.jsonl --condition anonymous --output results_anon.jsonl
        """
    )

    parser.add_argument(
        "-p", "--problem",
        type=str,
        help="Single problem to solve"
    )

    parser.add_argument(
        "-f", "--file",
        type=str,
        help="Input JSONL file with problems"
    )

    parser.add_argument(
        "--condition",
        type=str,
        choices=["own", "anonymous"],
        required=True,
        help="Ablation condition: 'own' (self-attributed) or 'anonymous'"
    )

    parser.add_argument(
        "--provider",
        type=str,
        choices=["mistral", "gemini"],
        default="gemini",
        help="LLM provider (default: gemini)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name override"
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum reasoning steps (default: 10)"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Confidence threshold to stop (default: 0.9)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output log file (default: ablation_{condition}.jsonl)"
    )

    parser.add_argument(
        "--critic",
        action="store_true",
        help="Enable critic mode for stopping decisions"
    )

    args = parser.parse_args()

    if not args.problem and not args.file:
        print("Error: Must provide -p or -f")
        sys.exit(1)

    # Set default output filename based on condition
    if args.output is None:
        args.output = f"ablation_{args.condition}.jsonl"

    from state import ThoughtState
    from model import create_model
    from controller import ControllerConfig
    from logger import ReasoningLogger, StepLog
    from parser import parse_model_output
    from datetime import datetime
    import time
    import json

    # Select prompt builder based on condition
    if args.condition == "own":
        build_prompt = build_prompt_own
        condition_desc = "SELF-ATTRIBUTED (model knows it's its own solution)"
    else:
        build_prompt = build_prompt_anonymous
        condition_desc = "ANONYMOUS (no self-attribution)"

    print("=" * 60)
    print("Ablation Study: Self-Attribution in Recursive Reasoning")
    print("=" * 60)
    print(f"Condition: {args.condition.upper()} - {condition_desc}")
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model or '(default)'}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Confidence Threshold: {args.threshold}")
    print(f"Output: {args.output}")
    print("=" * 60)
    print()

    # Load problems
    problems = []
    if args.problem:
        problems.append({"id": "cli_0", "problem": args.problem})
    else:
        with open(args.file, 'r') as f:
            for i, line in enumerate(f):
                if line.strip():
                    data = json.loads(line)
                    if "id" not in data:
                        data["id"] = f"prob_{i}"
                    problems.append(data)

    print(f"Loaded {len(problems)} problem(s)")
    print("-" * 40)

    # Initialize model
    try:
        model = create_model(provider=args.provider, model_name=args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    config = ControllerConfig(
        max_steps=args.max_steps,
        confidence_threshold=args.threshold,
        use_critic=args.critic
    )

    # Run each problem
    all_results = []

    for prob_data in problems:
        problem_text = prob_data.get("problem") or prob_data.get("question_content") or prob_data.get("text")
        problem_id = prob_data.get("id", "unknown")

        print(f"\nProblem [{problem_id}]: {problem_text[:80]}...")

        state = ThoughtState(problem=problem_text)
        state_history = [state.to_dict()]
        step_logs = []

        start_time = time.time()
        stop_reason = None

        while True:
            system_prompt, user_prompt = build_prompt(state)

            # Generate response
            parse_result = None
            true_confidence = None
            for retry in range(config.max_parse_retries + 1):
                model_response = model.generate(system_prompt, user_prompt, json_mode=True)
                raw_output = model_response.text
                true_confidence = model_response.true_confidence
                parse_result = parse_model_output(raw_output)

                if parse_result.success:
                    break
                if retry < config.max_parse_retries:
                    print(f"  Parse failed (retry {retry + 1})")

            # Update state
            if parse_result.success:
                decision = parse_result.decision
                new_state = state.update_from_model_output(parse_result.updated_state)
            else:
                decision = "CONTINUE"
                new_state = state.copy()
                new_state.step += 1

            # Log step
            step_logs.append({
                "step": state.step,
                "decision": decision,
                "confidence": new_state.confidence,
                "parse_success": parse_result.success if parse_result else False,
                "raw_output": parse_result.raw_output if parse_result else "",
                "true_confidence": true_confidence
            })

            print(f"  Step {new_state.step}: decision={decision}, confidence={new_state.confidence:.2f}")

            state = new_state
            state_history.append(state.to_dict())

            # Check stopping conditions
            if state.step >= config.max_steps:
                stop_reason = "max_steps_reached"
                break
            if decision == "STOP":
                stop_reason = "model_decided_stop"
                break
            if state.confidence >= config.confidence_threshold:
                stop_reason = "confidence_threshold_reached"
                break

        elapsed = time.time() - start_time

        result = {
            "id": problem_id,
            "problem": problem_text,
            "condition": args.condition,
            "final_solution": state.current_solution,
            "final_confidence": state.confidence,
            "total_steps": state.step,
            "stop_reason": stop_reason,
            "elapsed_time": elapsed,
            "steps": step_logs,
            "answer": prob_data.get("answer"),  # ground truth if available
        }
        all_results.append(result)

        print(f"  -> Completed in {state.step} steps, confidence={state.confidence:.2f}, reason={stop_reason}")

    # Save results
    with open(args.output, 'w') as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    print()
    print("=" * 60)
    print("ABLATION COMPLETE")
    print("=" * 60)
    print(f"Condition: {args.condition}")
    print(f"Problems: {len(all_results)}")
    print(f"Results saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
