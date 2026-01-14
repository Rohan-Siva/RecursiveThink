# Agentic Recursive Reasoning System

A research-grade, minimal, extensible Python codebase for agentic recursive reasoning using Mistral API.

## Overview

This system implements inference-time recursive reasoning where:
- A frozen LLM is repeatedly called
- Each call updates an explicit "thought state"
- The model proposes whether to CONTINUE or STOP
- An external controller enforces safety limits
- Structured JSON outputs ensure parseable responses

## Architecture

```
main.py          CLI entry point
    |
controller.py    Recursive reasoning loop
    |
    +-- model.py      Mistral API wrapper
    +-- prompt.py     System prompt templates
    +-- parser.py     JSON output validation
    +-- state.py      ThoughtState dataclass
    +-- logger.py     JSONL logging
```

## Installation

```bash
pip install mistralai
export MISTRAL_API_KEY="your-api-key"
```

## Usage

```bash
# Basic usage
python main.py -p "What is 2 + 2?"

# With custom settings
python main.py -p "Explain recursion" --max-steps 5 --threshold 0.85

# Use different Mistral model
python main.py -p "Solve x^2 = 16" --model mistral-small-latest
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| -p, --problem | (required) | Problem to solve |
| --model | mistral-large-latest | Mistral model name |
| --max-steps | 10 | Maximum reasoning steps |
| --threshold | 0.9 | Confidence threshold to stop |
| --log-file | reasoning.jsonl | Output log path |

## Stopping Conditions

The controller stops when ANY condition is met:
1. Model outputs decision="STOP"
2. Confidence >= threshold
3. Step count >= max_steps
4. State stops changing (stagnation)
5. Loop detected (repeating states)

## Output Format

Model outputs JSON:
```json
{
  "analysis": "Reasoning for this step",
  "decision": "CONTINUE or STOP",
  "updated_state": {
    "current_solution": "Updated solution",
    "open_questions": "Remaining questions",
    "confidence": 0.0-1.0
  }
}
```

## Log Format (JSONL)

Each line in the log file is a JSON object:
- Step logs: timestamp, state_before, model_output, decision, state_after
- Summary: total_steps, stop_reason, elapsed_time, final_state

## Extension Points

- **New models**: Modify model.py to support different backends
- **Custom prompts**: Edit prompt.py templates
- **New stopping conditions**: Add to controller._should_stop()
- **Different output schemas**: Modify parser.py validation

## Design Principles

- No training/fine-tuning (inference only)
- No LangChain or agent frameworks
- Structured JSON outputs (no free-form parsing)
- Controller never trusts model blindly
- All steps logged for inspection
