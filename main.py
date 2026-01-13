"""
main.py - CLI Entry Point for Recursive Reasoning System
"""
import argparse
import sys


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Agentic Recursive Reasoning System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py -p "What is the capital of France?"
  python main.py -p "Solve: 2x + 5 = 15" --max-steps 5
  python main.py -p "Explain quantum entanglement" --model mistralai/Mistral-7B-Instruct-v0.1

Requirements:
  pip install torch transformers
        """
    )
    
    parser.add_argument(
        "-p", "--problem",
        type=str,
        required=True,
        help="The problem to solve"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/phi-2",
        help="HuggingFace model name (default: microsoft/phi-2)"
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
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="Device to run on (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    # Lazy imports to allow --help without dependencies
    from state import ThoughtState
    from model import ModelWrapper
    from controller import Controller, ControllerConfig
    from logger import ReasoningLogger
    
    print("=" * 60)
    print("Agentic Recursive Reasoning System")
    print("=" * 60)
    print(f"Problem: {args.problem}")
    print(f"Model: {args.model}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Confidence Threshold: {args.threshold}")
    print("=" * 60)
    print()
    
    # Initialize components
    try:
        model = ModelWrapper(
            model_name=args.model,
            device=args.device
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    config = ControllerConfig(
        max_steps=args.max_steps,
        confidence_threshold=args.threshold
    )
    
    logger = ReasoningLogger(log_path=args.log_file)
    
    controller = Controller(
        model=model,
        config=config,
        logger=logger
    )
    
    # Run reasoning
    print("Starting recursive reasoning...")
    print("-" * 40)
    
    final_state = controller.run(args.problem)
    
    # Print results
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
