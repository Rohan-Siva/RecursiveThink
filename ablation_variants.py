"""
Extended Ablation Study: Test 6 different prompt variants to compare performance.

Variants:
1. "own" - Model is told it's building on its OWN previous reasoning (self-attribution)
2. "anonymous" - Model builds on "the" previous solution (no self-attribution)
3. "step_by_step" - Emphasizes step-by-step thinking with explicit instructions
4. "minimal" - Minimal system prompt, just JSON output requirements
5. "expert" - Framed as an expert mathematician/problem solver
6. "chain_of_thought" - Explicit chain-of-thought prompting style
"""

import argparse
import sys
import json
import time
from dotenv import load_dotenv

load_dotenv()

# ===== VARIANT 1: OWN (Self-attributed) =====
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

# ===== VARIANT 2: ANONYMOUS (No self-attribution) =====
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

# ===== VARIANT 3: STEP_BY_STEP (Explicit step-by-step instructions) =====
SYSTEM_PROMPT_STEP_BY_STEP = """You are a methodical problem solver that thinks step-by-step.

## How to Think
1. First, carefully read the problem and current state
2. Identify what you know and what you need to find
3. Break down the problem into smaller sub-problems
4. Solve each sub-problem systematically
5. Verify your answer at each step

## Input Format
- **Problem**: The question to solve
- **State**: Step number, current solution, open questions, confidence

## Output Format
Respond with ONLY valid JSON:
{
  "analysis": "Step 1: [what you observe]\nStep 2: [what you calculate/deduce]\nStep 3: [your conclusion]",
  "decision": "CONTINUE or STOP",
  "updated_state": {
    "current_solution": "Your solution with \\\\boxed{answer}",
    "open_questions": "What needs more work",
    "confidence": 0.0 to 1.0
  }
}

## Decision: CONTINUE if more steps needed, STOP when confident and complete.
For math, always include \\\\boxed{answer} with just the number.
"""

# ===== VARIANT 4: MINIMAL (Bare minimum instructions) =====
SYSTEM_PROMPT_MINIMAL = """Solve the problem iteratively. Output JSON only:
{
  "analysis": "your reasoning",
  "decision": "CONTINUE or STOP",
  "updated_state": {
    "current_solution": "solution with \\\\boxed{answer}",
    "open_questions": "uncertainties",
    "confidence": 0.0-1.0
  }
}
STOP when confident (>=0.9). Math answers in \\\\boxed{}.
"""

# ===== VARIANT 5: EXPERT (Expert persona) =====
SYSTEM_PROMPT_EXPERT = """You are a world-class expert problem solver with decades of experience.

Your expertise allows you to:
- Quickly identify the core of any problem
- Apply advanced techniques efficiently
- Catch subtle errors that others miss
- Provide rigorous, complete solutions

## Your Process
As an expert, you iteratively refine solutions. You receive the problem and current state, then provide your expert analysis.

## Output (Expert Analysis)
Provide your expert evaluation as JSON:
{
  "analysis": "Your expert analysis: key insights, techniques applied, verification",
  "decision": "CONTINUE (if you see room for improvement) or STOP (if solution is complete)",
  "updated_state": {
    "current_solution": "Your refined solution with final answer in \\\\boxed{answer}",
    "open_questions": "Any remaining considerations",
    "confidence": 0.0 to 1.0 (your expert confidence level)
  }
}

For mathematical problems, always express the final answer as \\\\boxed{value}.
STOP when your expert assessment reaches >= 0.9 confidence with a complete solution.
"""

# ===== VARIANT 6: CHAIN_OF_THOUGHT (Explicit CoT prompting) =====
SYSTEM_PROMPT_CHAIN_OF_THOUGHT = """Let's solve this problem using chain-of-thought reasoning.

Think carefully through each logical step:
- What information is given?
- What is being asked?
- What approach should I take?
- Let me work through this systematically...
- Wait, let me verify this...
- Therefore, the answer is...

## Input
You'll receive a problem and the current solving state with any previous progress.

## Output
Think through your reasoning chain, then output ONLY this JSON:
{
  "analysis": "Let me think through this... [your chain of thought reasoning]",
  "decision": "CONTINUE or STOP",
  "updated_state": {
    "current_solution": "Based on my reasoning, the solution is... Answer: \\\\boxed{value}",
    "open_questions": "Things to verify or consider",
    "confidence": 0.0 to 1.0
  }
}

Show your thinking process explicitly in the analysis. For math, box the final numerical answer: \\\\boxed{42}
STOP when you've thoroughly reasoned through and are confident (>=0.9).
"""


PROMPT_VARIANTS = {
    "own": {
        "system": SYSTEM_PROMPT_OWN,
        "description": "Self-attributed: model knows it's building on ITS OWN solution",
        "user_template": "Problem: {problem}. State (Step {step}): Your Solution: {solution}, Your Questions: {questions}, Your Confidence: {confidence:.2f}. Provide your next reasoning step as JSON."
    },
    "anonymous": {
        "system": SYSTEM_PROMPT_ANONYMOUS,
        "description": "Anonymous: no self-attribution, building on 'the' solution",
        "user_template": "Problem: {problem}. State (Step {step}): Solution: {solution}, Questions: {questions}, Confidence: {confidence:.2f}. Provide the next reasoning step as JSON."
    },
    "step_by_step": {
        "system": SYSTEM_PROMPT_STEP_BY_STEP,
        "description": "Explicit step-by-step thinking instructions",
        "user_template": "Problem: {problem}\n\nCurrent State:\n- Step: {step}\n- Solution: {solution}\n- Open Questions: {questions}\n- Confidence: {confidence:.2f}\n\nThink step-by-step and provide your JSON output."
    },
    "minimal": {
        "system": SYSTEM_PROMPT_MINIMAL,
        "description": "Minimal prompt - bare essentials only",
        "user_template": "Problem: {problem} | Step {step} | Solution: {solution} | Confidence: {confidence:.2f}"
    },
    "expert": {
        "system": SYSTEM_PROMPT_EXPERT,
        "description": "Expert persona - framed as world-class problem solver",
        "user_template": "As an expert, analyze this:\n\nPROBLEM: {problem}\n\nCURRENT STATE:\n- Iteration: {step}\n- Current Solution: {solution}\n- Open Questions: {questions}\n- Confidence: {confidence:.2f}\n\nProvide your expert JSON analysis."
    },
    "chain_of_thought": {
        "system": SYSTEM_PROMPT_CHAIN_OF_THOUGHT,
        "description": "Chain-of-thought prompting style",
        "user_template": "Problem to solve: {problem}\n\nCurrent progress (Step {step}):\n- Solution so far: {solution}\n- Questions: {questions}\n- Confidence: {confidence:.2f}\n\nLet's think through this step by step. Output your JSON response."
    }
}


def build_prompt(state, variant_name: str) -> tuple:
    """Build system and user prompts for a given variant."""
    variant = PROMPT_VARIANTS[variant_name]
    sol = state.current_solution or "(none yet)"
    q = state.open_questions or "(none identified)"
    
    user = variant["user_template"].format(
        problem=state.problem,
        step=state.step,
        solution=sol,
        questions=q,
        confidence=state.confidence
    )
    return variant["system"], user


def run_ablation(args, variant_name: str, problems: list):
    """Run ablation study for a single variant."""
    from state import ThoughtState
    from model import create_model
    from controller import ControllerConfig
    from parser import parse_model_output
    
    variant = PROMPT_VARIANTS[variant_name]
    print(f"\n{'='*60}")
    print(f"VARIANT: {variant_name.upper()}")
    print(f"Description: {variant['description']}")
    print(f"{'='*60}")
    
    # Initialize model
    try:
        model = create_model(provider=args.provider, model_name=args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return []
    
    config = ControllerConfig(
        max_steps=args.max_steps,
        confidence_threshold=args.threshold,
        use_critic=args.critic
    )
    
    all_results = []
    
    for prob_data in problems:
        problem_text = prob_data.get("problem") or prob_data.get("question_content") or prob_data.get("text")
        problem_id = prob_data.get("id", "unknown")
        expected_answer = prob_data.get("answer")
        
        print(f"\n  Problem [{problem_id}]: {problem_text[:60]}...")
        
        state = ThoughtState(problem=problem_text)
        step_logs = []
        start_time = time.time()
        stop_reason = None
        
        while True:
            system_prompt, user_prompt = build_prompt(state, variant_name)
            
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
                    print(f"    Parse retry {retry + 1}")
            
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
                "analysis": parse_result.analysis if parse_result and parse_result.success else "",
            })
            
            print(f"    Step {new_state.step}: conf={new_state.confidence:.2f}, decision={decision}")
            
            state = new_state
            
            # Check stopping conditions
            if state.step >= config.max_steps:
                stop_reason = "max_steps"
                break
            if decision == "STOP":
                stop_reason = "model_stop"
                break
            if state.confidence >= config.confidence_threshold:
                stop_reason = "confidence"
                break
        
        elapsed = time.time() - start_time
        
        result = {
            "id": problem_id,
            "problem": problem_text,
            "variant": variant_name,
            "final_solution": state.current_solution,
            "final_confidence": state.confidence,
            "total_steps": state.step,
            "stop_reason": stop_reason,
            "elapsed_time": round(elapsed, 2),
            "expected_answer": expected_answer,
            "steps": step_logs,
        }
        all_results.append(result)
        
        print(f"    -> Done: {state.step} steps, conf={state.confidence:.2f}, reason={stop_reason}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Extended Ablation Study: Test multiple prompt variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Prompt Variants:
  own            - Self-attributed (model's OWN solution)
  anonymous      - No self-attribution (THE solution)
  step_by_step   - Explicit step-by-step instructions
  minimal        - Minimal prompt, bare essentials
  expert         - Expert persona framing
  chain_of_thought - Explicit CoT prompting

Examples:
  python ablation_variants.py -p "Solve: 2x + 5 = 15" --variants own anonymous
  python ablation_variants.py -p "Solve: 2x + 5 = 15" --all-variants
  python ablation_variants.py -f problems.jsonl --all-variants --output results.jsonl
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
        "--variants",
        type=str,
        nargs="+",
        choices=list(PROMPT_VARIANTS.keys()),
        help="Specific variants to test"
    )
    
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Test all 6 variants"
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
        default="ablation_results.jsonl",
        help="Output log file (default: ablation_results.jsonl)"
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
    
    if not args.variants and not args.all_variants:
        print("Error: Must provide --variants or --all-variants")
        sys.exit(1)
    
    # Determine which variants to test
    variants_to_test = list(PROMPT_VARIANTS.keys()) if args.all_variants else args.variants
    
    print("=" * 70)
    print("EXTENDED ABLATION STUDY: Prompt Variant Comparison")
    print("=" * 70)
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model or '(default)'}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Confidence Threshold: {args.threshold}")
    print(f"Variants to test: {', '.join(variants_to_test)}")
    print(f"Output: {args.output}")
    print("=" * 70)
    
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
    
    print(f"\nLoaded {len(problems)} problem(s)")
    
    # Run ablation for each variant
    all_results = []
    variant_summaries = []
    
    for variant_name in variants_to_test:
        results = run_ablation(args, variant_name, problems)
        all_results.extend(results)
        
        # Compute summary stats for this variant
        if results:
            avg_steps = sum(r["total_steps"] for r in results) / len(results)
            avg_conf = sum(r["final_confidence"] for r in results) / len(results)
            avg_time = sum(r["elapsed_time"] for r in results) / len(results)
            variant_summaries.append({
                "variant": variant_name,
                "avg_steps": round(avg_steps, 2),
                "avg_confidence": round(avg_conf, 3),
                "avg_time": round(avg_time, 2),
                "num_problems": len(results)
            })
    
    # Save detailed results
    with open(args.output, 'w') as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    
    # Print summary comparison
    print("\n" + "=" * 70)
    print("ABLATION STUDY SUMMARY")
    print("=" * 70)
    print(f"{'Variant':<18} {'Avg Steps':>10} {'Avg Conf':>10} {'Avg Time':>10}")
    print("-" * 70)
    for s in variant_summaries:
        print(f"{s['variant']:<18} {s['avg_steps']:>10.2f} {s['avg_confidence']:>10.3f} {s['avg_time']:>10.2f}s")
    print("=" * 70)
    print(f"Total results saved to: {args.output}")
    
    # Save summary
    summary_file = args.output.replace('.jsonl', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump({
            "variants_tested": variants_to_test,
            "num_problems": len(problems),
            "provider": args.provider,
            "model": args.model,
            "max_steps": args.max_steps,
            "threshold": args.threshold,
            "variant_summaries": variant_summaries
        }, f, indent=2)
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
