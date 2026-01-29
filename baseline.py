whaimport argparse
import sys
import json
import os
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

from model import create_model

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run baseline Pass@k evaluation for generic problems."
    )
    parser.add_argument(
        "-p", "--problem",
        type=str,
        help="Single problem string to evaluate."
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        help="Input JSONL file containing problems. Must have 'problem' field (and optional 'answer')."
    )
    parser.add_argument(
        "-k", "--samples",
        type=int,
        default=8,
        help="Number of samples (k) per problem. Default: 8"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="gemini",
        choices=["gemini", "mistral"],
        help="LLM provider. Default: gemini"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name override."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers. Default: 4"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="baseline_results.jsonl",
        help="Output JSONL file."
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful assistant. Please answer the user's question. Think step by step.",
        help="System prompt to use."
    )
    return parser.parse_args()


def solve_single_attempt(model, system_prompt: str, problem: str, attempt_id: int) -> Dict[str, Any]:
    """Generates a single response for a problem."""
    try:
        response = model.generate(system_prompt=system_prompt, user_prompt=problem)
        return {
            "attempt_id": attempt_id,
            "response": response.text,
            "error": None
        }
    except Exception as e:
        return {
            "attempt_id": attempt_id,
            "response": None,
            "error": str(e)
        }


def process_problem(
    problem_data: Dict[str, Any],
    k: int,
    provider: str,
    model_name: str,
    system_prompt: str,
    workers: int
) -> Dict[str, Any]:
    """
    Process a single problem k times. 
    Note: We instantiate the model inside here or pass a thread-safe client.
    The wrappers in model.py initialize their own clients. 
    Ideally, we'd reuse the client, but for simplicity and safety across threads, 
    we can try to share one model instance if the client is thread-safe (Gemini/Mistral usually are),
    OR we instantiate one per thread if needed. 
    
    Given the current model.py structure, let's create one global model instance 
    to pass around, assuming client thread-safety.
    """
    
   
    
    problem_text = problem_data.get("problem") or problem_data.get("question_content")
    if not problem_text:
        return {"error": "No problem text found", "raw_data": problem_data}

    
    
    pass 
    # Logic is moved to main to handle both cases efficiently.


def main():
    args = parse_args()
    
    # 1. Load Data
    problems = []
    if args.problem:
        problems.append({"id": "custom_0", "problem": args.problem})
    elif args.file:
        with open(args.file, 'r') as f:
            for line in f:
                if line.strip():
                    problems.append(json.loads(line))
    else:
        print("Error: Must provide -p or -f")
        sys.exit(1)
        
    print(f"Loaded {len(problems)} problem(s).")
    print(f"Configuration: Provider={args.provider}, K={args.samples}, Workers={args.workers}")

    # 2. Initialize Model (assuming thread safety for API clients)
    try:
        model = create_model(provider=args.provider, model_name=args.model)
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        sys.exit(1)

    # 3. Execution
    results = []
    
    # Helper to run k samples for one problem
    def run_samples_for_problem(prob_item):
        p_text = prob_item.get("problem") or prob_item.get("question_content") or prob_item.get("text")
        if not p_text:
            return {**prob_item, "error": "Invalid format"}
            
        # We define a function for a single sample generation to be submitted to executor
        def generate_sample(idx):
            return solve_single_attempt(model, args.system_prompt, p_text, idx)

        # Use a local executor for the k samples if we are processing a single problem,
        # otherwise we might just do them strictly sequentially or depend on the outer loop.
        # But to be robust: let's just iterate k times. 
        # API latency is the bottleneck. 
        
        samples = []
        # If specific problem count is low (like 1), we want parallel samples.
        # If problem count is high, we want parallel problems.
        # Let's blindly use ThreadPool for everything by flattening the tasks?
        # Simpler: Iterate k times.
        
        item_results = []
        for i in range(args.samples):
            item_results.append(generate_sample(i))
            
        prob_item["samples"] = item_results
        return prob_item

    # If we have 1 problem, run samples in parallel
    if len(problems) == 1:
        print("Single problem mode: Parallelizing samples.")
        prob = problems[0]
        p_text = prob.get("problem")
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(solve_single_attempt, model, args.system_prompt, p_text, i)
                for i in range(args.samples)
            ]
            samples = []
            for f in tqdm(as_completed(futures), total=args.samples, desc="Generating samples"):
                samples.append(f.result())
        
        # Sort by attempt_id
        samples.sort(key=lambda x: x['attempt_id'])
        prob["samples"] = samples
        results.append(prob)
        
    else:
        print("Batch mode: Parallelizing problems (sequential samples per problem).")
        # For multiple problems, parallelize at the problem level
        # logic: each worker grabs a problem and does k samples sequentially (or could parallelize internal, but keep simple)
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_samples_for_problem, p): p for p in problems}
            
            for f in tqdm(as_completed(futures), total=len(problems), desc="Processing problems"):
                results.append(f.result())

    # 4. Save Results
    with open(args.output, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    print(f"\nCompleted. Results saved to {args.output}")

    # 5. Basic Report (if ground truth exists)
    # Just checking the structure of valid answers if available
    # We can't easily auto-grade free text without a parser, but we can report completion.
    
    total_samples = len(results) * args.samples
    print(f"Total Generations: {total_samples}")


if __name__ == "__main__":
    main()
