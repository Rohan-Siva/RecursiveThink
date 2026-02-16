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

from state import ThoughtState
from model import create_model
from controller import Controller, ControllerConfig
from logger import ReasoningLogger
from tts.math_grader import extract_answer, grade


def run_single_problem(
    problem: str,
    answer: str,
    question_id: str,
    model,
    config: ControllerConfig,
    log_dir: Path,
) -> dict:

    log_path = log_dir / f"{question_id}.jsonl"
    logger = ReasoningLogger(log_path=str(log_path))

    controller = Controller(
        model=model,
        config=config,
        logger=logger
    )

    start_time = time.time()

    try:
        final_state = controller.run(problem)
        elapsed = time.time() - start_time

        # solution extraction from state
        solution = final_state.current_solution
        model_raw_answer = extract_answer(solution)

        # grading
        is_correct = False
        if model_raw_answer is not None:
            is_correct = grade(model_raw_answer, str(answer))

        result = {
            "question_id": question_id,
            "question_content": problem,
            "answer": answer,
            "model_output": solution,
            "model_raw_answer": model_raw_answer,
            "is_correct": is_correct,
            "total_steps": final_state.step,
            "final_confidence": final_state.confidence,
            "open_questions": final_state.open_questions,
            "elapsed_time": round(elapsed, 2),
            "log_file": str(log_path),
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
        description="Run RecursiveThink on AIME 2025 dataset"
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
        help="Maximum reasoning steps per problem (default: 10)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Confidence threshold to stop (default: 0.9)"
    )
    parser.add_argument(
        "--critic",
        action="store_true",
        help="Enable critic mode for stopping decisions"
    )
    parser.add_argument(
        "--uncertainty",
        action="store_true",
        help="Enable sampling-based uncertainty: measure confidence via response consistency"
    )
    parser.add_argument(
        "--uncertainty-samples",
        type=int,
        default=5,
        help="Number of samples for uncertainty estimation (default: 5)"
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

    config = ControllerConfig(
        max_steps=args.max_steps,
        confidence_threshold=args.threshold,
        use_critic=args.critic,
        use_uncertainty=args.uncertainty,
        uncertainty_samples=args.uncertainty_samples,
        answer_extractor=extract_answer,
    )

    model_name = args.model or ("gemini-2.5-flash" if args.provider == "gemini" else "mistral-large-latest")
    model_save_name = model_name.replace("/", "_").replace("-", "_")

    output_dir = Path(args.output_dir)
    results_dir = output_dir / f"aime_recursive_{model_save_name}"
    logs_dir = results_dir / f"logs_exp{args.exp_id}"

    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RecursiveThink AIME 2025 Evaluation")
    print("=" * 60)
    print(f"Provider: {args.provider}")
    print(f"Model: {model_name}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Confidence Threshold: {args.threshold}")
    print(f"Critic Mode: {'enabled' if args.critic else 'disabled'}")
    print(f"Uncertainty Mode: {'enabled' if args.uncertainty else 'disabled'}")
    if args.uncertainty:
        print(f"Uncertainty Samples: {args.uncertainty_samples}")
    print("=" * 60)

    all_results = []
    correct_count = 0

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

        print(f"  Answer: {result['model_raw_answer']} | Expected: {answer} | {status}")
        print(f"  Steps: {result.get('total_steps', 'N/A')} | Confidence: {result.get('final_confidence', 'N/A')}")

    accuracy = correct_count / len(questions) if questions else 0 # acc count

    output_file = results_dir / f"aime2025_recursive_{model_save_name}_exp{args.exp_id}_steps{args.max_steps}.json"

    summary = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "provider": args.provider,
            "model": model_name,
            "max_steps": args.max_steps,
            "confidence_threshold": args.threshold,
            "critic_mode": args.critic,
            "uncertainty_mode": args.uncertainty,
            "uncertainty_samples": args.uncertainty_samples if args.uncertainty else None,
            "exp_id": args.exp_id,
        },
        "summary": {
            "total_problems": len(questions),
            "correct": correct_count,
            "accuracy": round(accuracy, 4),
        },
        "results": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Accuracy: {correct_count}/{len(questions)} = {accuracy:.2%}")
    print(f"Results saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
