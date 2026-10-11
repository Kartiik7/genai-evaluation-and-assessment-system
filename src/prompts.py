"""
src/prompts.py

Utilities for loading and rendering prompt templates with version tracking.
Templates reside in the prompts/ directory (e.g. direct_v1.txt, rubric_v1.txt).
"""

from pathlib import Path
from typing import Tuple

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str) -> str:
    """
    Loads a prompt template string by name.
    
    Parameters
    ----------
    name : str
        Template name, with or without '.txt' extension (e.g. 'direct_v1' or 'direct_v1.txt').
        
    Returns
    -------
    str : Raw prompt template text.
    """
    clean_name = name if name.endswith(".txt") else f"{name}.txt"
    path = PROMPTS_DIR / clean_name
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found at {path}")
    return path.read_text(encoding="utf-8")


def render(name: str, **fields) -> Tuple[str, str]:
    """
    Renders a prompt template with keyword fields.
    
    Parameters
    ----------
    name : str
        Template name (e.g. 'direct_v1').
    **fields : Keyword arguments matching {placeholders} in template.
    
    Returns
    -------
    Tuple[str, str] : (rendered_prompt, prompt_version_string)
        prompt_version_string is the base name without extension (e.g. 'direct_v1').
    """
    version = name[:-4] if name.endswith(".txt") else name
    template = load_prompt(name)
    rendered = template
    for key, value in fields.items():
        rendered = rendered.replace(f"{{{key}}}", str(value))
    return rendered, version

