import argparse
import sys
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Agentic Recursive Reasoning System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py -p "What is the capital of France?"
  python main.py -p "Solve: 2x + 5 = 15" --max-steps 5
  python main.py -p "Explain recursion" --provider gemini
  python main.py -p "Complex math" --provider mistral --model mistral-large-latest

Providers:
  mistral - Requires MISTRAL_API_KEY env var
  gemini  - Requires GEMINI_API_KEY env var (uses thinking by default)
        """
    )
    
    parser.add_argument(
        "-p", "--problem",
        type=str,
        required=True,
        help="The problem to solve"
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        choices=["mistral", "gemini"],
        default="gemini",
        help="LLM provider to use (default: gemini)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name override (default: provider's default model)"
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
        "--log-file",
        type=str,
        default="reasoning.jsonl",
        help="Output log file path (default: reasoning.jsonl)"
    )
    
    parser.add_argument(
        "--critic",
        action="store_true",
        help="Enable critic mode: use LLM to evaluate stopping instead of confidence threshold"
    )

    parser.add_argument(
        "--uncertainty",
        action="store_true",
        help="Enable sampling-based uncertainty: measure confidence via response consistency instead of self-reported scores"
    )

    parser.add_argument(
        "--uncertainty-samples",
        type=int,
        default=5,
        help="Number of samples for uncertainty estimation (default: 5)"
    )

    parser.add_argument(
        "--branching",
        action="store_true",
        help="Enable uncertainty-guided branching: decompose into subgoals, branch on uncertain ones"
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

    args = parser.parse_args()
    
    from state import ThoughtState
    from model import create_model
    from controller import Controller, ControllerConfig
    from logger import ReasoningLogger
    
    print("=" * 60)
    print("Agentic Recursive Reasoning System")
    print("=" * 60)
    print(f"Problem: {args.problem}")
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model or '(default)'}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Confidence Threshold: {args.threshold}")
    print(f"Critic Mode: {'enabled' if args.critic else 'disabled'}")
    print(f"Uncertainty Mode: {'enabled' if args.uncertainty else 'disabled'}")
    if args.uncertainty:
        print(f"Uncertainty Samples: {args.uncertainty_samples}")
    print(f"Branching Mode: {'enabled' if args.branching else 'disabled'}")
    if args.branching:
        print(f"Branch Threshold: {args.branch_threshold}")
        print(f"Branch Verification Steps: {args.branch_steps}")
    print("=" * 60)
    print()
    
    try:
        model = create_model(
            provider=args.provider,
            model_name=args.model
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    logger = ReasoningLogger(log_path=args.log_file)

    if args.branching:
        from branching_controller import BranchingController, BranchingConfig

        branch_config = BranchingConfig(
            branch_threshold=args.branch_threshold,
            num_samples=args.uncertainty_samples,
            max_branches=3,
            branch_verification_steps=args.branch_steps,
            final_check_samples=args.uncertainty_samples,
        )

        controller = BranchingController(
            model=model,
            config=branch_config,
            logger=logger,
        )

        print("Starting branching recursive reasoning...")
        print("-" * 40)

        final_state, subgoals = controller.run(args.problem)

        print("-" * 40)
        print()
        print("=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)
        print(final_state.current_solution)
        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total Subgoals: {len(subgoals)}")
        print(f"Final Confidence: {final_state.confidence:.2f}")
        print(f"API Calls: {controller.api_calls}")
        branched = [sg for sg in subgoals if sg.branches]
        print(f"Subgoals Branched: {len(branched)}/{len(subgoals)}")
        for sg in subgoals:
            marker = " ⚡" if sg.branches else ""
            print(f"  {sg.subgoal_id}. conf={sg.confidence:.2f}{marker} {sg.description[:60]}")
        print(f"Log File: {args.log_file}")
        print("=" * 60)

    else:
        config = ControllerConfig(
            max_steps=args.max_steps,
            confidence_threshold=args.threshold,
            use_critic=args.critic,
            use_uncertainty=args.uncertainty,
            uncertainty_samples=args.uncertainty_samples,
        )

        controller = Controller(
            model=model,
            config=config,
            logger=logger
        )

        print("Starting recursive reasoning...")
        print("-" * 40)

        final_state = controller.run(args.problem)

        print("-" * 40)
        print()
        print("=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)
        print(final_state.current_solution)
        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total Steps: {final_state.step}")
        print(f"Final Confidence: {final_state.confidence:.2f}")
        open_q = final_state.open_questions or "None"
        print(f"Open Questions: {open_q}")
        print(f"Log File: {args.log_file}")
        print("=" * 60)


if __name__ == "__main__":
    main()

