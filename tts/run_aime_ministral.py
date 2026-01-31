"""
Run Ministral models on AIME 2025 benchmark.

Reproduces results from "Ministral 3" paper (arxiv 2601.08584):
- Ministral-3-3B-Reasoning: 72.1% on AIME '25
- Ministral-3-8B-Reasoning: higher
- Ministral-3-14B-Reasoning: ~85% on AIME '25

The reasoning models use native extended thinking (trained via SFT -> GRPO -> ODPO).
They must be run locally via vLLM as they're not available via Mistral API.

Usage:
    # Run 8B reasoning model via vLLM (local)
    python tts/run_aime_ministral.py --model ministral-8b-reasoning --backend vllm

    # Run 3B reasoning model
    python tts/run_aime_ministral.py --model ministral-3b-reasoning --backend vllm

    # Run instruct model via Mistral API (for comparison)
    python tts/run_aime_ministral.py --model ministral-8b-instruct --backend api
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

import datasets
from tqdm import tqdm

from tts.math_grader import extract_answer, grade


# Model name mappings
MODEL_CONFIGS = {
    # Reasoning models (local via vLLM)
    "ministral-3b-reasoning": {
        "hf_name": "mistralai/Ministral-3-3B-Reasoning-2512",
        "is_reasoning": True,
        "backend": "vllm",
    },
    "ministral-8b-reasoning": {
        "hf_name": "mistralai/Ministral-3-8B-Reasoning-2512",
        "is_reasoning": True,
        "backend": "vllm",
    },
    "ministral-14b-reasoning": {
        "hf_name": "mistralai/Ministral-3-14B-Reasoning-2512",
        "is_reasoning": True,
        "backend": "vllm",
    },
    # Instruct models (API or local)
    "ministral-3b-instruct": {
        "hf_name": "mistralai/Ministral-3-3B-Instruct-2512",
        "api_name": "ministral-3b-2512",
        "is_reasoning": False,
        "backend": "api",
    },
    "ministral-8b-instruct": {
        "hf_name": "mistralai/Ministral-3-8B-Instruct-2512",
        "api_name": "ministral-8b-2512",
        "is_reasoning": False,
        "backend": "api",
    },
    # Mistral Large (API) - reported ~40% on AIME
    "mistral-large": {
        "hf_name": None,
        "api_name": "mistral-large-latest",
        "is_reasoning": False,
        "backend": "api",
    },
}


@dataclass
class GenerationResult:
    """Result from model generation."""
    output: str
    thinking: Optional[str] = None
    total_tokens: int = 0
    generation_time: float = 0.0


class MinistralVLLM:
    """Wrapper for running Ministral reasoning models via vLLM."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.9,
    ):
        from vllm import LLM, SamplingParams

        config = MODEL_CONFIGS.get(model_name)
        if not config:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_CONFIGS.keys())}")

        self.hf_name = config["hf_name"]
        self.is_reasoning = config["is_reasoning"]
        self.model_name = model_name

        print(f"Loading {self.hf_name} via vLLM...")
        print(f"  Reasoning model: {self.is_reasoning}")
        print(f"  Tensor parallel: {tensor_parallel_size}")
        print(f"  Max model len: {max_model_len}")

        self.llm = LLM(
            model=self.hf_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        print("Model loaded successfully.")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 16384,
        temperature: float = 0.6,
        top_p: float = 0.95,
    ) -> GenerationResult:
        from vllm import SamplingParams

        # Format as chat message
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        start_time = time.time()
        outputs = self.llm.generate([formatted_prompt], sampling_params)
        generation_time = time.time() - start_time

        output_text = outputs[0].outputs[0].text
        total_tokens = len(outputs[0].outputs[0].token_ids)

        # Parse thinking for reasoning models
        thinking = None
        final_output = output_text

        if self.is_reasoning and "[THINK]" in output_text:
            # Extract thinking section
            if "[/THINK]" in output_text:
                parts = output_text.split("[/THINK]", 1)
                thinking_part = parts[0]
                if "[THINK]" in thinking_part:
                    thinking = thinking_part.split("[THINK]", 1)[1].strip()
                final_output = parts[1].strip() if len(parts) > 1 else ""
            else:
                # Thinking not closed, entire output is thinking
                if "[THINK]" in output_text:
                    thinking = output_text.split("[THINK]", 1)[1].strip()
                    final_output = ""

        return GenerationResult(
            output=final_output if final_output else output_text,
            thinking=thinking,
            total_tokens=total_tokens,
            generation_time=generation_time,
        )


class MinistralAPI:
    """Wrapper for Ministral models via Mistral API."""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        from mistralai import Mistral
        from dotenv import load_dotenv

        load_dotenv()

        config = MODEL_CONFIGS.get(model_name)
        if not config:
            raise ValueError(f"Unknown model: {model_name}")

        if "api_name" not in config:
            raise ValueError(f"{model_name} is not available via API. Use --backend vllm")

        self.api_name = config["api_name"]
        self.model_name = model_name
        self.is_reasoning = config["is_reasoning"]

        api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found")

        print(f"Initializing Mistral API client for {self.api_name}...")
        self.client = Mistral(api_key=api_key)
        print("Client ready.")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 8192,
        temperature: float = 0.6,
    ) -> GenerationResult:
        start_time = time.time()

        response = self.client.chat.complete(
            model=self.api_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        generation_time = time.time() - start_time
        output_text = response.choices[0].message.content.strip()
        total_tokens = response.usage.completion_tokens if response.usage else 0

        return GenerationResult(
            output=output_text,
            thinking=None,
            total_tokens=total_tokens,
            generation_time=generation_time,
        )


def create_prompt(problem: str, is_reasoning: bool = True) -> str:
    """Create prompt for AIME problem."""

    if is_reasoning:
        # For reasoning models, let them think naturally
        prompt = f"""Solve this math problem step by step. Think carefully and show your reasoning.

Problem:
{problem}

After your reasoning, provide your final answer as a single integer inside \\boxed{{}}.
For example: \\boxed{{42}}"""
    else:
        # For instruct models, more explicit prompting
        prompt = f"""You are an expert mathematician. Solve this competition math problem step by step.

Problem:
{problem}

Instructions:
1. Break down the problem carefully
2. Show all your work and reasoning
3. Double-check your calculations
4. Put your final numerical answer inside \\boxed{{}}

Your final answer must be a single integer inside \\boxed{{}}."""

    return prompt


def run_single_problem(
    problem: str,
    answer: str,
    question_id: str,
    model,
    is_reasoning: bool,
    k: int = 1,
) -> dict:
    """Run model on a single AIME problem with k samples."""

    prompt = create_prompt(problem, is_reasoning)
    samples = []
    correct_count = 0
    total_tokens = 0
    total_time = 0

    for attempt in range(k):
        try:
            result = model.generate(prompt)

            # Extract answer from output
            output_text = result.output
            model_raw_answer = extract_answer(output_text)

            # If no answer in main output, check thinking (reasoning models sometimes put it there)
            if model_raw_answer is None and result.thinking:
                model_raw_answer = extract_answer(result.thinking)

            is_correct = False
            if model_raw_answer is not None:
                is_correct = grade(model_raw_answer, str(answer))

            if is_correct:
                correct_count += 1

            samples.append({
                "attempt": attempt,
                "output": output_text,
                "thinking": result.thinking,
                "model_raw_answer": model_raw_answer,
                "is_correct": is_correct,
                "tokens": result.total_tokens,
                "time": round(result.generation_time, 2),
            })

            total_tokens += result.total_tokens
            total_time += result.generation_time

        except Exception as e:
            samples.append({
                "attempt": attempt,
                "error": str(e),
                "is_correct": False,
            })

    return {
        "question_id": question_id,
        "problem": problem,
        "answer": answer,
        "samples": samples,
        "correct_count": correct_count,
        "k": k,
        "pass_at_k": correct_count > 0,
        "total_tokens": total_tokens,
        "total_time": round(total_time, 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Ministral models on AIME 2025 (reproducing arxiv 2601.08584)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ministral-8b-reasoning",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to use (default: ministral-8b-reasoning)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "api"],
        default=None,
        help="Backend to use (default: auto based on model)"
    )
    parser.add_argument(
        "-k", "--samples",
        type=int,
        default=1,
        help="Number of samples per problem for pass@k (default: 1)"
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
        "--max-tokens",
        type=int,
        default=16384,
        help="Max tokens for generation (default: 16384)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.6)"
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM (default: 1)"
    )
    parser.add_argument(
        "--exp-id",
        type=int,
        default=1,
        help="Experiment ID (default: 1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory (default: results)"
    )

    args = parser.parse_args()

    # Get model config
    config = MODEL_CONFIGS[args.model]
    backend = args.backend or config["backend"]
    is_reasoning = config["is_reasoning"]

    # Validate backend choice
    if backend == "api" and "api_name" not in config:
        print(f"ERROR: {args.model} is not available via API.")
        print("Reasoning models must be run locally via vLLM.")
        print("Use: --backend vllm")
        sys.exit(1)

    print("=" * 70)
    print("MINISTRAL AIME 2025 EVALUATION")
    print("Reproducing results from arxiv 2601.08584 (Ministral 3 paper)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Backend: {backend}")
    print(f"Is Reasoning Model: {is_reasoning}")
    print(f"Samples per problem (k): {args.samples}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print("=" * 70)

    # Load dataset
    print("\nLoading AIME 2025 dataset...")
    data = datasets.load_dataset("yentinglin/aime_2025")["train"]

    questions = data["problem"]
    answers = data["answer"]

    end_idx = len(questions) if args.num_problems is None else min(args.start_idx + args.num_problems, len(questions))
    questions = questions[args.start_idx:end_idx]
    answers = answers[args.start_idx:end_idx]
    question_ids = [f"aime25_{i:02}" for i in range(args.start_idx + 1, end_idx + 1)]

    print(f"Running on {len(questions)} problems (idx {args.start_idx} to {end_idx - 1})")

    # Initialize model
    print(f"\nInitializing model via {backend}...")
    if backend == "vllm":
        model = MinistralVLLM(
            model_name=args.model,
            tensor_parallel_size=args.tensor_parallel,
        )
    else:
        model = MinistralAPI(model_name=args.model)

    # Setup output directory
    model_save_name = args.model.replace("-", "_")
    output_dir = Path(args.output_dir)
    results_dir = output_dir / f"ministral_{model_save_name}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    print("\nStarting evaluation...")
    all_results = []
    pass_count = 0
    total_tokens = 0

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
            is_reasoning=is_reasoning,
            k=args.samples,
        )

        all_results.append(result)
        total_tokens += result["total_tokens"]

        if result["pass_at_k"]:
            pass_count += 1
            status = "PASS"
        else:
            status = "FAIL"

        # Get best answer to display
        best_answer = None
        for s in result["samples"]:
            if s.get("is_correct"):
                best_answer = s.get("model_raw_answer")
                break
            if best_answer is None:
                best_answer = s.get("model_raw_answer")

        print(f"  Answer: {best_answer} | Expected: {answer} | {status}")
        print(f"  Correct: {result['correct_count']}/{args.samples} | Tokens: {result['total_tokens']}")

    # Calculate final metrics
    accuracy = pass_count / len(questions) if questions else 0

    # Save results
    output_file = results_dir / f"aime2025_{model_save_name}_exp{args.exp_id}_k{args.samples}.json"

    summary = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "hf_model": config["hf_name"],
            "backend": backend,
            "is_reasoning": is_reasoning,
            "k": args.samples,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "exp_id": args.exp_id,
            "paper_reference": "arxiv 2601.08584 (Ministral 3)",
        },
        "summary": {
            "total_problems": len(questions),
            "pass_at_k": pass_count,
            "accuracy": round(accuracy, 4),
            "total_tokens": total_tokens,
            "avg_tokens_per_problem": round(total_tokens / len(questions), 1) if questions else 0,
        },
        "paper_comparison": {
            "note": "Paper reported results for AIME '25",
            "ministral_3b_reasoning": "72.1%",
            "ministral_14b_reasoning": "~85%",
        },
        "results": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Pass@{args.samples}: {pass_count}/{len(questions)} = {accuracy:.1%}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Results saved to: {output_file}")
    print("=" * 70)

    # Compare to paper
    print("\nPaper comparison (arxiv 2601.08584):")
    print("  Ministral-3-3B-Reasoning: 72.1% on AIME '25")
    print("  Ministral-3-14B-Reasoning: ~85% on AIME '25")
    print(f"  Your result ({args.model}): {accuracy:.1%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
