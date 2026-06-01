"""Object-label utilities for prompt grounding."""

from __future__ import annotations

import re
from typing import Iterable, Optional

CANONICAL_LABELS = [
    "banana",
    "meat_can",
    "coke_can",
    "strawberry",
    "hammer",
    "book",
    "eraser",
    "soap",
    "snack",
    "biscuits",
    "glue",
    "mustard_bottle",
    "sticky_notes",
]

ALIASES = {
    "banana": ["banana", "bananas"],
    "meat_can": ["meat can", "meat_can", "can of meat", "meat"],
    "coke_can": ["coke can", "coke_can", "coca cola", "cola can", "soda can", "coke"],
    "strawberry": ["strawberry", "strawberries"],
    "hammer": ["hammer"],
    "book": ["book"],
    "eraser": ["eraser", "rubber"],
    "soap": ["soap", "soap bar"],
    "snack": ["snack", "snacks"],
    "biscuits": ["biscuits", "biscuit box", "biscuits box", "biscuit"],
    "glue": ["glue", "glue stick"],
    "mustard_bottle": ["mustard bottle", "mustard_bottle", "mustard"],
    "sticky_notes": ["sticky notes", "sticky_notes", "post it", "post-it"],
}


def normalize_label(text: str) -> str:
    """Normalize free-form model output into a canonical class name when possible."""
    if not text:
        return "object"
    t = re.sub(r"[_\-]+", " ", text.lower()).strip()
    for canonical, aliases in ALIASES.items():
        if t == canonical.replace("_", " "):
            return canonical
        for alias in aliases:
            if t == alias.lower():
                return canonical
    return re.sub(r"\s+", "_", t)


def extract_target_label(prompt: str, default: str = "object") -> str:
    """Extract the requested object class from a command-like prompt."""
    if not prompt:
        return default
    text = prompt.lower().replace("_", " ")
    # Prefer longest aliases first, otherwise "can" would be ambiguous.
    candidates = []
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            candidates.append((canonical, alias.lower()))
    candidates.sort(key=lambda x: len(x[1]), reverse=True)

    for canonical, alias in candidates:
        if re.search(rf"\b{re.escape(alias)}\b", text):
            return canonical
    return default


def labels_for_detector(target: str) -> list[str]:
    """Prompts used by zero-shot detectors."""
    canonical = normalize_label(target)
    if canonical in ALIASES:
        readable = canonical.replace("_", " ")
        return [readable, f"a photo of a {readable}", f"the {readable}"]
    return [target, f"a photo of a {target}"]
