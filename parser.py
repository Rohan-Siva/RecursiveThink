"""
parser.py - JSON Output Parser and Validator
"""
import json
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ParseResult:
    """Result of parsing model output."""
    success: bool
    analysis: Optional[str] = None
    decision: Optional[str] = None
    updated_state: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    raw_output: str = ""


def extract_json(text: str) -> Optional[str]:
    """
    Extract JSON object from potentially messy model output.
    
    Args:
        text: Raw model output
        
    Returns:
        Extracted JSON string or None if not found
    """
    # Try to find JSON between curly braces
    # Handle potential markdown code blocks
    text = re.sub(r"```json?\n?", "", text)
    text = re.sub(r"```", "", text)
    
    # Find the outermost curly braces
    start = text.find("{")
    if start == -1:
        return None
    
    # Count braces to find matching close
    depth = 0
    for i, char in enumerate(text[start:], start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    
    return None


def validate_schema(data: Dict[str, Any]) -> Optional[str]:
    """
    Validate that parsed JSON matches expected schema.
    
    Returns:
        Error message if invalid, None if valid
    """
    required_fields = ["analysis", "decision", "updated_state"]
    for field in required_fields:
        if field not in data:
            return f"Missing required field: {field}"
    
    # Validate decision
    if data["decision"] not in ["CONTINUE", "STOP"]:
        return f"Invalid decision: {data['decision']} (must be CONTINUE or STOP)"
    
    # Validate updated_state
    state = data["updated_state"]
    if not isinstance(state, dict):
        return "updated_state must be a dictionary"
    
    state_fields = ["current_solution", "open_questions", "confidence"]
    for field in state_fields:
        if field not in state:
            return f"Missing field in updated_state: {field}"
    
    # Validate confidence range
    try:
        conf = float(state["confidence"])
        if not 0.0 <= conf <= 1.0:
            return f"Confidence must be between 0.0 and 1.0, got {conf}"
    except (ValueError, TypeError):
        return f"Invalid confidence value: {state['confidence']}"
    
    return None


def parse_model_output(raw: str) -> ParseResult:
    """
    Parse and validate model output.
    
    Args:
        raw: Raw model output string
        
    Returns:
        ParseResult with success status and parsed data or error
    """
    if not raw or not raw.strip():
        return ParseResult(
            success=False,
            error="Empty model output",
            raw_output=raw
        )
    
    # Extract JSON from output
    json_str = extract_json(raw)
    if not json_str:
        return ParseResult(
            success=False,
            error="No JSON object found in output",
            raw_output=raw
        )
    
    # Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return ParseResult(
            success=False,
            error=f"JSON parse error: {str(e)}",
            raw_output=raw
        )
    
    # Validate schema
    schema_error = validate_schema(data)
    if schema_error:
        return ParseResult(
            success=False,
            error=schema_error,
            raw_output=raw
        )
    
    # Success
    return ParseResult(
        success=True,
        analysis=str(data["analysis"]),
        decision=data["decision"],
        updated_state=data["updated_state"],
        raw_output=raw
    )
