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
from branching_controller import BranchingController, BranchingConfig
from logger import ReasoningLogger
from tts.math_grader import extract_answer, grade
from uncertainty import extract_boxed_answer


def run_single_problem(
    problem: str,
    answer: str,
    question_id: str,
    model,
    config: BranchingConfig,
    log_dir: Path,
) -> dict:

    log_path = log_dir / f"{question_id}.jsonl"
    logger = ReasoningLogger(log_path=str(log_path))

    controller = BranchingController(
        model=model,
        config=config,
        logger=logger,
    )

    start_time = time.time()

    try:
        final_state, subgoals = controller.run(problem)
        elapsed = time.time() - start_time

        solution = final_state.current_solution
        model_raw_answer = extract_answer(solution)

        is_correct = False
        if model_raw_answer is not None:
            is_correct = grade(model_raw_answer, str(answer))

        branched_subgoals = [sg for sg in subgoals if sg.branches]

        result = {
            "question_id": question_id,
            "question_content": problem,
            "answer": answer,
            "model_output": solution,
            "model_raw_answer": model_raw_answer,
            "is_correct": is_correct,
            "total_subgoals": len(subgoals),
            "subgoals_branched": len(branched_subgoals),
            "final_confidence": final_state.confidence,
            "api_calls": controller.api_calls,
            "elapsed_time": round(elapsed, 2),
            "log_file": str(log_path),
            "subgoals": [sg.to_dict() for sg in subgoals],
        }

    except Exception as e:
        elapsed = time.time() - start_time
        result = {
            "question_id": question_id,
            "question_content": problem,
            "answer": answer,
            "model_output": None,
            "model_raw_answer": None,
            "is_correct": False,
            "error": str(e),
            "elapsed_time": round(elapsed, 2),
            "log_file": str(log_path),
        }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run Branching RecursiveThink on AIME 2025 dataset"
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["mistral", "gemini"],
        default="mistral",
        help="LLM provider (default: mistral)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name override"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Uncertainty samples per subgoal (default: 5)"
    )
    parser.add_argument(
        "--branch-threshold",
        type=float,
        default=0.8,
        help="Confidence threshold below which to branch (default: 0.8)"
    )
    parser.add_argument(
        "--branch-steps",
        type=int,
        default=2,
        help="Verification steps per branch (default: 2)"
    )
    parser.add_argument(
        "--max-branches",
        type=int,
        default=3,
        help="Max branches per uncertain subgoal (default: 3)"
    )
    parser.add_argument(
        "--num-problems",
        type=int,
        default=None,
        help="Number of problems to run (default: all 30)"
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
        help="Experiment ID for output naming (default: 1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory (default: results)"
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

    print(f"Running on {len(questions)} problems (idx {args.start_idx} to {end_idx - 1})")

    print(f"Initializing {args.provider} model...")
    model = create_model(
        provider=args.provider,
        model_name=args.model
    )

    config = BranchingConfig(
        branch_threshold=args.branch_threshold,
        num_samples=args.num_samples,
        max_branches=args.max_branches,
        branch_verification_steps=args.branch_steps,
        final_check_samples=args.num_samples,
    )

    model_name = args.model or ("gemini-2.5-flash" if args.provider == "gemini" else "mistral-large-latest")
    model_save_name = model_name.replace("/", "_").replace("-", "_")

    output_dir = Path(args.output_dir)
    results_dir = output_dir / f"aime_branching_{model_save_name}"
    logs_dir = results_dir / f"logs_exp{args.exp_id}"

    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Branching RecursiveThink AIME 2025 Evaluation")
    print("=" * 60)
    print(f"Provider: {args.provider}")
    print(f"Model: {model_name}")
    print(f"Samples: {args.num_samples}")
    print(f"Branch Threshold: {args.branch_threshold}")
    print(f"Branch Steps: {args.branch_steps}")
    print(f"Max Branches: {args.max_branches}")
    print("=" * 60)

    all_results = []
    correct_count = 0
    total_api_calls = 0

    for question, answer, question_id in tqdm(
        zip(questions, answers, question_ids),
        total=len(questions),
        desc="Evaluating"
    ):
        print(f"\n--- {question_id} ---")

        result = run_single_problem(
            problem=question,
            answer=answer,
            question_id=question_id,
            model=model,
            config=config,
            log_dir=logs_dir,
        )

        all_results.append(result)

        if result["is_correct"]:
            correct_count += 1
            status = "CORRECT"
        else:
            status = "INCORRECT"

        api = result.get("api_calls", "N/A")
        total_api_calls += api if isinstance(api, int) else 0
        print(f"  Answer: {result['model_raw_answer']} | Expected: {answer} | {status}")
        print(f"  Subgoals: {result.get('total_subgoals', 'N/A')} | "
              f"Branched: {result.get('subgoals_branched', 'N/A')} | "
              f"API Calls: {api}")

        partial_accuracy = correct_count / len(all_results) if all_results else 0
        partial_file = results_dir / f"aime2025_branching_{model_save_name}_exp{args.exp_id}_partial.json"
        partial_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "provider": args.provider,
                "model": model_name,
                "num_samples": args.num_samples,
                "branch_threshold": args.branch_threshold,
                "branch_steps": args.branch_steps,
                "max_branches": args.max_branches,
                "exp_id": args.exp_id,
                "status": "in_progress",
                "completed": len(all_results),
                "total_planned": len(questions),
            },
            "summary": {
                "total_problems": len(all_results),
                "correct": correct_count,
                "accuracy": round(partial_accuracy, 4),
                "total_api_calls": total_api_calls,
            },
            "results": all_results,
        }
        with open(partial_file, "w") as f:
            json.dump(partial_data, f, indent=2)

    accuracy = correct_count / len(questions) if questions else 0

    output_file = results_dir / f"aime2025_branching_{model_save_name}_exp{args.exp_id}.json"

    summary = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "provider": args.provider,
            "model": model_name,
            "num_samples": args.num_samples,
            "branch_threshold": args.branch_threshold,
            "branch_steps": args.branch_steps,
            "max_branches": args.max_branches,
            "exp_id": args.exp_id,
        },
        "summary": {
            "total_problems": len(questions),
            "correct": correct_count,
            "accuracy": round(accuracy, 4),
            "total_api_calls": total_api_calls,
        },
        "results": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Accuracy: {correct_count}/{len(questions)} = {accuracy:.2%}")
    print(f"Total API Calls: {total_api_calls}")
    print(f"Results saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
