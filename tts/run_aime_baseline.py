import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import datasets

from model import create_model
from tts.math_grader import extract_answer, grade


def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    import itertools

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


def run_baseline_problem(
    problem: str,
    answer: str,
    question_id: str,
    model,
    system_prompt: str,
    k: int,
    save_callback=None,
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
                "attempt": attempt + 1,
                "model_output": raw_output,
                "model_raw_answer": model_raw_answer,
                "is_correct": is_correct,
            })
        except Exception as e:
            samples.append({
                "attempt": attempt + 1,
                "error": str(e),
                "is_correct": False,
            })

        # Print progress for each attempt
        print(f"    Attempt {attempt + 1}/{k}: {correct_count}/{attempt + 1} correct", end="")
        if samples[-1].get("is_correct"):
            print(" ✓")
        else:
            print(f" (got: {samples[-1].get('model_raw_answer', 'error')}, expected: {answer})")

        # Save intermediate result after each attempt
        if save_callback:
            partial_result = {
                "question_id": question_id,
                "question_content": problem,
                "answer": answer,
                "samples": samples.copy(),
                "correct_count": correct_count,
                "attempts_completed": attempt + 1,
                "k": k,
                "pass_at_k": correct_count > 0,
                "in_progress": attempt + 1 < k,
            }
            save_callback(partial_result)

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
    correct_counts = []  # Track correct count for each problem for pass@k estimation

    # Incremental save file
    output_file = results_dir / f"aime2025_baseline_{model_save_name}_exp{args.exp_id}_k{args.samples}.json"

    def save_intermediate(partial_result):
        """Save intermediate results after each attempt."""

        # Build current results list with partial current problem
        results_to_save = all_results.copy()
        results_to_save.append(partial_result)

        # Calculate current stats
        current_pass_count = pass_count + (1 if partial_result["pass_at_k"] else 0)
        current_correct_counts = correct_counts.copy()
        current_correct_counts.append(partial_result["correct_count"])

        partial_summary = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "provider": args.provider,
                "model": model_name,
                "k": args.samples,
                "exp_id": args.exp_id,
                "status": "in_progress",
            },
            "summary": {
                "completed_problems": len(all_results),
                "current_problem": partial_result["question_id"],
                "current_attempts": f"{partial_result['attempts_completed']}/{args.samples}",
                "current_correct": f"{partial_result['correct_count']}/{partial_result['attempts_completed']}",
                "total_problems": len(questions),
                "pass_at_k_count": current_pass_count,
                "pass_at_k_accuracy": round(current_pass_count / len(results_to_save), 4) if results_to_save else 0,
            },
            "per_problem_correct_counts": current_correct_counts,
            "results": results_to_save,
        }
        with open(output_file, "w") as f:
            json.dump(partial_summary, f, indent=2)

    for idx, (question, answer, question_id) in enumerate(
        zip(questions, answers, question_ids)
    ):
        print(f"\n{'=' * 40}")
        print(f"Problem {idx + 1}/{len(questions)}: {question_id}")
        print(f"{'=' * 40}")

        result = run_baseline_problem(
            problem=question,
            answer=answer,
            question_id=question_id,
            model=model,
            system_prompt=system_prompt,
            k=args.samples,
            save_callback=save_intermediate,
        )

        all_results.append(result)
        correct_counts.append(result["correct_count"])

        if result["pass_at_k"]:
            pass_count += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"\n  >> {question_id} FINAL: {result['correct_count']}/{args.samples} correct | {status}")

        # Save after each completed problem
        partial_summary = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "provider": args.provider,
                "model": model_name,
                "k": args.samples,
                "exp_id": args.exp_id,
                "status": "in_progress",
            },
            "summary": {
                "completed_problems": len(all_results),
                "total_problems": len(questions),
                "pass_at_k_count": pass_count,
                "pass_at_k_accuracy": round(pass_count / len(all_results), 4) if all_results else 0,
            },
            "per_problem_correct_counts": correct_counts,
            "results": all_results,
        }
        with open(output_file, "w") as f:
            json.dump(partial_summary, f, indent=2)

    accuracy = pass_count / len(questions) if questions else 0

    # Compute pass@k estimates using the statistical estimator
    correct_counts_arr = np.array(correct_counts)
    pass_at_1 = estimate_pass_at_k(args.samples, correct_counts_arr, 1).mean()
    pass_at_k = estimate_pass_at_k(args.samples, correct_counts_arr, args.samples).mean()

    # Also compute pass@k for intermediate k values if k >= 4
    pass_at_estimates = {"pass@1": round(float(pass_at_1), 4)}
    if args.samples >= 4:
        pass_at_4 = estimate_pass_at_k(args.samples, correct_counts_arr, 4).mean()
        pass_at_estimates["pass@4"] = round(float(pass_at_4), 4)
    if args.samples >= 8:
        pass_at_8 = estimate_pass_at_k(args.samples, correct_counts_arr, 8).mean()
        pass_at_estimates["pass@8"] = round(float(pass_at_8), 4)
    pass_at_estimates[f"pass@{args.samples}"] = round(float(pass_at_k), 4)

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
            "pass_at_k_count": pass_count,
            "pass_at_k_accuracy": round(accuracy, 4),
            "pass_at_k_estimates": pass_at_estimates,
        },
        "per_problem_correct_counts": correct_counts,
        "per_problem_summary": [
            {"question_id": r["question_id"], "correct": r["correct_count"], "total": args.samples}
            for r in all_results
        ],
        "results": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("BASELINE EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Total Problems: {len(questions)}")
    print(f"\nPer-problem correct counts (out of {args.samples}):")
    for r in all_results:
        status = "✓" if r["pass_at_k"] else "✗"
        print(f"  {r['question_id']}: {r['correct_count']}/{args.samples} {status}")
    print(f"\nPass@k Estimates (statistical):")
    for k_name, k_val in pass_at_estimates.items():
        print(f"  {k_name}: {k_val:.2%}")
    print(f"\nRaw Pass@{args.samples}: {pass_count}/{len(questions)} = {accuracy:.2%}")
    print(f"\nResults saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
