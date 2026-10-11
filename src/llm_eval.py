"""
src/llm_eval.py

Validation and evaluation helpers for LLM-based short-answer grading responses.
Validates structured JSON output from direct and rubric prompts:
- score must be a number in [0, 5]
- for rubric prompt, each criterion score must also be in [0, 5]
- invalid outputs produce NaN scores / parse failure indications
"""

from typing import Any, Dict, List, Optional
import numpy as np

RUBRIC_CRITERIA: List[str] = [
    "correctness",
    "completeness",
    "relevance",
    "understanding",
    "clarity",
]


def is_valid_score(val: Any) -> bool:
    """Checks if a value is a valid numeric score in [0, 5]."""
    if val is None or isinstance(val, bool):
        return False
    try:
        f = float(val)
        return not np.isnan(f) and 0.0 <= f <= 5.0
    except (ValueError, TypeError):
        return False


def validate_direct_output(data: Any) -> Dict[str, Any]:
    """
    Validates output from direct grading prompts.
    
    Expected format:
    {
      "reasoning": str,
      "score": number in [0, 5]
    }
    
    Returns
    -------
    dict with normalized fields:
      - 'is_valid': bool
      - 'score': float (or np.nan if invalid)
      - 'reasoning': str
      - 'error': str (if invalid, else None)
    """
    if not isinstance(data, dict) or "error" in data:
        return {
            "is_valid": False,
            "score": np.nan,
            "reasoning": "",
            "error": data.get("error", "Output is not a valid dict") if isinstance(data, dict) else "Output is not a dict"
        }
    
    score_raw = data.get("score")
    if not is_valid_score(score_raw):
        return {
            "is_valid": False,
            "score": np.nan,
            "reasoning": str(data.get("reasoning", "")),
            "error": f"Invalid score value: {score_raw}"
        }
        
    return {
        "is_valid": True,
        "score": float(score_raw),
        "reasoning": str(data.get("reasoning", "")),
        "error": None
    }


def validate_rubric_output(data: Any) -> Dict[str, Any]:
    """
    Validates output from rubric grading prompts.
    
    Expected format:
    {
      "criterion_scores": {
        "correctness": 0-5,
        "completeness": 0-5,
        "relevance": 0-5,
        "understanding": 0-5,
        "clarity": 0-5
      },
      "strengths": list,
      "missing_concepts": list,
      "errors": list,
      "feedback": str,
      "reasoning": str,
      "score": 0-5
    }
    
    Returns
    -------
    dict with normalized fields:
      - 'is_valid': bool
      - 'score': float (or np.nan if invalid)
      - 'criterion_scores': dict mapping criterion -> float (or np.nan)
      - 'strengths': list
      - 'missing_concepts': list
      - 'errors': list
      - 'feedback': str
      - 'reasoning': str
      - 'error': str (if invalid, else None)
    """
    if not isinstance(data, dict) or "error" in data:
        return {
            "is_valid": False,
            "score": np.nan,
            "criterion_scores": {c: np.nan for c in RUBRIC_CRITERIA},
            "strengths": [],
            "missing_concepts": [],
            "errors": [],
            "feedback": "",
            "reasoning": "",
            "error": data.get("error", "Output is not a valid dict") if isinstance(data, dict) else "Output is not a dict"
        }

    # Validate overall score
    score_raw = data.get("score")
    if not is_valid_score(score_raw):
        return {
            "is_valid": False,
            "score": np.nan,
            "criterion_scores": {c: np.nan for c in RUBRIC_CRITERIA},
            "strengths": list(data.get("strengths", [])),
            "missing_concepts": list(data.get("missing_concepts", [])),
            "errors": list(data.get("errors", [])),
            "feedback": str(data.get("feedback", "")),
            "reasoning": str(data.get("reasoning", "")),
            "error": f"Invalid overall score value: {score_raw}"
        }

    # Validate criterion_scores
    crit_dict = data.get("criterion_scores")
    if not isinstance(crit_dict, dict):
        return {
            "is_valid": False,
            "score": np.nan,
            "criterion_scores": {c: np.nan for c in RUBRIC_CRITERIA},
            "strengths": list(data.get("strengths", [])),
            "missing_concepts": list(data.get("missing_concepts", [])),
            "errors": list(data.get("errors", [])),
            "feedback": str(data.get("feedback", "")),
            "reasoning": str(data.get("reasoning", "")),
            "error": "Missing or non-dict criterion_scores"
        }

    parsed_crit = {}
    for crit in RUBRIC_CRITERIA:
        c_val = crit_dict.get(crit)
        if not is_valid_score(c_val):
            return {
                "is_valid": False,
                "score": np.nan,
                "criterion_scores": {c: np.nan for c in RUBRIC_CRITERIA},
                "strengths": list(data.get("strengths", [])),
                "missing_concepts": list(data.get("missing_concepts", [])),
                "errors": list(data.get("errors", [])),
                "feedback": str(data.get("feedback", "")),
                "reasoning": str(data.get("reasoning", "")),
                "error": f"Invalid or missing criterion score for '{crit}': {c_val}"
            }
        parsed_crit[crit] = float(c_val)

    return {
        "is_valid": True,
        "score": float(score_raw),
        "criterion_scores": parsed_crit,
        "strengths": list(data.get("strengths", [])),
        "missing_concepts": list(data.get("missing_concepts", [])),
        "errors": list(data.get("errors", [])),
        "feedback": str(data.get("feedback", "")),
        "reasoning": str(data.get("reasoning", "")),
        "error": None
    }


def validate_response(data: Any, prompt_type: str = "direct") -> Dict[str, Any]:
    """
    Unified validator dispatching to prompt-specific validation based on prompt_type.
    
    Parameters
    ----------
    data : Parsed response dict from LLM.
    prompt_type : str ('direct' or 'rubric', case-insensitive, or version name like 'direct_v1', 'rubric_v1').
    
    Returns
    -------
    dict with validation metadata and extracted score (or np.nan on failure).
    """
    ptype = prompt_type.lower()
    if "rubric" in ptype:
        return validate_rubric_output(data)
    else:
        return validate_direct_output(data)
