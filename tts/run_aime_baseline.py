import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import datasets
from tqdm import tqdm

from model import create_model
from tts.math_grader import extract_answer, grade


def run_baseline_problem(
    problem: str,
    answer: str,
    question_id: str,
    model,
    system_prompt: str,
    k: int,
) -> dict:

    samples = []
    correct_count = 0
    start_time = time.time()

    for attempt in range(k):
        try:
            response = model.generate(system_prompt=system_prompt, user_prompt=problem)
            raw_output = response.text
            
            model_raw_answer = extract_answer(raw_output)
            is_correct = False
            if model_raw_answer is not None:
                is_correct = grade(model_raw_answer, str(answer))
            
            if is_correct:
                correct_count += 1
                
            samples.append({
                "attempt": attempt,
                "model_output": raw_output,
                "model_raw_answer": model_raw_answer,
                "is_correct": is_correct,
            })
        except Exception as e:
            samples.append({
                "attempt": attempt,
                "error": str(e),
                "is_correct": False,
            })
    
    elapsed = time.time() - start_time
    
    any_correct = correct_count > 0
    
    return {
        "question_id": question_id,
        "question_content": problem,
        "answer": answer,
        "samples": samples,
        "correct_count": correct_count,
        "k": k,
        "pass_at_k": any_correct,
        "elapsed_time": round(elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Baseline Pass@k on AIME 2025"
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
        "-k", "--samples",
        type=int,
        default=1,
        help="Number of samples per problem (default: 1)"
    )
    parser.add_argument(
        "--num-problems",
        type=int,
        default=None,
        help="Number of problems to run (default: all)"
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting problem index (default: 0)"
    )
    parser.add_argument(
        "--exp-id",
        type=int,
        default=1,
        help="Experiment ID"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory"
    )

    args = parser.parse_args()

    print("Loading AIME 2025 dataset...")
    data = datasets.load_dataset("yentinglin/aime_2025")["train"]

    questions = data["problem"]
    answers = data["answer"]

    end_idx = len(questions) if args.num_problems is None else min(args.start_idx + args.num_problems, len(questions))
    questions = questions[args.start_idx:end_idx]
    answers = answers[args.start_idx:end_idx]
    question_ids = [f"id_{i:02}" for i in range(args.start_idx + 1, end_idx + 1)]

    print(f"Running on {len(questions)} problems")

    print(f"Initializing {args.provider} model...")
    model = create_model(
        provider=args.provider,
        model_name=args.model
    )

    system_prompt = """You are a math problem solver. Solve the problem step by step.

CRITICAL: You MUST put your final numerical answer inside \\boxed{} format. 
For example: \\boxed{42} or \\boxed{123}

The answer inside the box should be ONLY the final number, nothing else."""

    model_name = args.model or ("gemini-2.5-flash" if args.provider == "gemini" else "mistral-large-latest")
    model_save_name = model_name.replace("/", "_").replace("-", "_")

    output_dir = Path(args.output_dir)
    results_dir = output_dir / f"aime_baseline_{model_save_name}"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BASELINE AIME 2025 Evaluation")
    print("=" * 60)
    print(f"Provider: {args.provider}")
    print(f"Model: {model_name}")
    print(f"Samples per problem (k): {args.samples}")
    print("=" * 60)

    all_results = []
    pass_count = 0

    for question, answer, question_id in tqdm(
        zip(questions, answers, question_ids),
        total=len(questions),
        desc="Evaluating"
    ):
        print(f"\n--- {question_id} ---")

        result = run_baseline_problem(
            problem=question,
            answer=answer,
            question_id=question_id,
            model=model,
            system_prompt=system_prompt,
            k=args.samples,
        )

        all_results.append(result)

        if result["pass_at_k"]:
            pass_count += 1
            status = "PASS"
        else:
            status = "FAIL"

        best_answer = None
        for s in result["samples"]:
            if s.get("is_correct"):
                best_answer = s.get("model_raw_answer")
                break
            if best_answer is None:
                best_answer = s.get("model_raw_answer")
                
        print(f"  Answer: {best_answer} | Expected: {answer} | {status} ({result['correct_count']}/{args.samples})")

    accuracy = pass_count / len(questions) if questions else 0

    output_file = results_dir / f"aime2025_baseline_{model_save_name}_exp{args.exp_id}_k{args.samples}.json"

    summary = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "provider": args.provider,
            "model": model_name,
            "k": args.samples,
            "exp_id": args.exp_id,
        },
        "summary": {
            "total_problems": len(questions),
            "pass_at_k": pass_count,
            "accuracy": round(accuracy, 4),
        },
        "results": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("BASELINE EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Pass@{args.samples}: {pass_count}/{len(questions)} = {accuracy:.2%}")
    print(f"Results saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
