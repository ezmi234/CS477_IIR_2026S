"""Object-label and prompt-grounding utilities for the vision pipeline.

The competition commands use human-facing names such as "coke can" or
"meat can", while zero-shot detectors respond better to visual descriptions.
Keep coke/meat can prompts class-specific in normal runtime: coke is a red
cylindrical drink can, while meat is a rectangular spam-like food can.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


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

# Aliases used to understand the instructor command.  Keep the most specific
# phrases in each list because extraction searches longer aliases first.
ALIASES: dict[str, list[str]] = {
    "banana": [
        "yellow banana",
        "banana",
        "bananas",
    ],
    "meat_can": [
        "rectangular meat can",
        "square meat can",
        "spam can",
        "spam-like can",
        "meat can",
        "meat_can",
        "rectangular food can",
        "box shaped tin of meat",
        "box-shaped tin of meat",
        "small rectangular canned meat",
        "canned meat",
        "meat tin",
        "food tin",
        "meat",
    ],
    "coke_can": [
        "red cylindrical soda can",
        "red cola can",
        "red coke can",
        "red drink can",
        "red beverage can",
        "cylindrical red can",
        "coke can",
        "coca cola can",
        "cola can",
        "red soda can",
        "soda can",
        "red can",
        "tin can",
        "drink can",
        "beverage can",
        "can of cola",
        "coke",
        "cola",
    ],
    "strawberry": [
        "red strawberry",
        "strawberry",
        "strawberries",
    ],
    "hammer": [
        "hammer with long handle",
        "hammer with wooden handle",
        "tool with long stick handle",
        "long handled hammer",
        "hammer head and stick handle",
        "metal hammer with handle",
        "hammer tool",
        "long handle",
        "stick handle",
        "tool handle",
        "tool hammer",
        "metal hammer",
        "hammer",
    ],
    "book": [
        "book",
        "notebook",
    ],
    "eraser": [
        "eraser",
        "rubber eraser",
        "rubber",
    ],
    "soap": [
        "soap bar",
        "bar of soap",
        "soap",
    ],
    "snack": [
        "snack bag",
        "snack package",
        "snacks",
        "snack",
    ],
    "biscuits": [
        "biscuits box",
        "biscuit box",
        "box of biscuits",
        "biscuits",
        "biscuit",
    ],
    "glue": [
        "glue stick",
        "glue bottle",
        "glue",
    ],
    "mustard_bottle": [
        "mustard bottle",
        "mustard_bottle",
        "yellow bottle",
        "mustard",
    ],
    "sticky_notes": [
        "sticky notes",
        "sticky_notes",
        "post it notes",
        "post-it notes",
        "post it",
        "post-it",
    ],
}

# Aliases sent to the zero-shot detector.  These can be broader than the command
# aliases, but avoid the generic word "object" because it causes useless boxes.
# Normal target detection intentionally avoids generic can prompts for
# coke_can/meat_can; generic prompts live in RELAXED_DETECTOR_ALIASES and are
# marked as fail-open/fallback by the vision server.
DETECTOR_ALIASES: dict[str, list[str]] = {
    "banana": ["banana", "yellow banana", "curved yellow banana"],
    "meat_can": [
        "rectangular meat can",
        "square meat can",
        "spam can",
        "spam-like can",
        "rectangular food can",
        "box-shaped tin of meat",
        "small rectangular canned meat",
    ],
    "coke_can": [
        "red cylindrical soda can",
        "red cola can",
        "red coke can",
        "red drink can",
        "red beverage can",
        "cylindrical red can",
        "red soda can",
        "cola can",
        "coca cola can",
    ],
    "strawberry": ["strawberry", "red strawberry"],
    "hammer": [
        "hammer with long handle",
        "hammer with wooden handle",
        "tool with long stick handle",
        "long handled hammer",
        "hammer head and stick handle",
        "metal hammer with handle",
    ],
    "book": ["book", "notebook"],
    "eraser": ["eraser", "rubber eraser"],
    "soap": ["soap", "soap bar", "bar of soap"],
    "snack": ["snack", "snack bag", "snack package"],
    "biscuits": ["biscuits", "biscuit box", "box of biscuits"],
    "glue": ["glue", "glue stick"],
    "mustard_bottle": ["mustard bottle", "yellow bottle", "mustard"],
    "sticky_notes": ["sticky notes", "post it notes", "post-it notes"],
    "object": ["object", "small object", "item on table", "thing on table"],
}

NON_STRICT_EXTRA_DETECTOR_ALIASES: dict[str, list[str]] = {
    "meat_can": ["meat can", "canned meat", "meat tin", "food tin", "food can", "tin can"],
    "coke_can": ["coke can", "red can", "cola can", "soda can", "drink can", "beverage can"],
    "hammer": ["hammer", "hammer tool", "long handle", "stick handle", "tool handle"],
}

RELAXED_DETECTOR_ALIASES: dict[str, list[str]] = {
    "banana": ["yellow banana", "curved yellow object", "yellow object", "banana"],
    "meat_can": [
        "rectangular meat can",
        "spam can",
        "rectangular can",
        "square can",
        "canned meat",
        "food tin",
        "food can",
        "tin can",
        "small can",
    ],
    "coke_can": [
        "red cylindrical soda can",
        "red coke can",
        "red can",
        "drink can",
        "soda can",
        "cola can",
        "beverage can",
        "red cylindrical object",
    ],
    "strawberry": ["red fruit", "small red object", "strawberry", "red round object"],
    "hammer": [
        "hammer with long handle",
        "long handled hammer",
        "hammer head and stick handle",
        "hammer tool",
        "long handle",
        "stick handle",
        "tool handle",
        "tool",
        "metal object",
    ],
    "object": ["object", "small object", "item on table", "thing on table"],
}

# Penalty applied to weaker/generic aliases.  It is used only for ranking; the
# raw model score is still published for debugging.
ALIAS_RANK_WEIGHT: dict[str, float] = {
    "can": 0.60,
    "small can": 0.70,
    "small tin": 0.70,
    "tin can": 0.78,
    "metal can": 0.72,
    "food can": 0.78,
    "drink can": 0.80,
    "beverage can": 0.80,
    "coke": 0.90,
    "cola": 0.95,
    "rectangular meat can": 1.08,
    "square meat can": 1.05,
    "spam can": 1.08,
    "spam-like can": 1.08,
    "rectangular food can": 1.04,
    "red cylindrical soda can": 1.10,
    "red cola can": 1.08,
    "red coke can": 1.08,
    "red drink can": 1.05,
    "red beverage can": 1.05,
    "cylindrical red can": 1.08,
    "hammer with long handle": 1.10,
    "hammer with wooden handle": 1.08,
    "tool with long stick handle": 1.08,
    "long handled hammer": 1.08,
    "hammer head and stick handle": 1.10,
    "metal hammer with handle": 1.06,
    "hammer tool": 0.92,
    "long handle": 0.82,
    "stick handle": 0.82,
    "tool handle": 0.78,
    "tool": 0.60,
    "metal object": 0.55,
}


@dataclass(frozen=True)
class DetectorQuery:
    """Text query submitted to a zero-shot detector."""

    text: str
    canonical: str
    rank_weight: float = 1.0


def _clean_text(text: str) -> str:
    """Normalize text while preserving useful spaces for phrase matching."""
    text = (text or "").lower()
    text = text.replace("_", " ").replace("-", " ")
    # Strip common prompt wrappers returned by OWL-ViT labels.
    wrappers = [
        r"^a\s+photo\s+of\s+an?\s+",
        r"^a\s+picture\s+of\s+an?\s+",
        r"^an?\s+image\s+of\s+an?\s+",
        r"^the\s+",
        r"^an?\s+",
    ]
    for pattern in wrappers:
        text = re.sub(pattern, "", text)
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _all_alias_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for canonical, aliases in ALIASES.items():
        pairs.append((canonical, canonical.replace("_", " ")))
        for alias in aliases:
            pairs.append((canonical, _clean_text(alias)))
        for alias in DETECTOR_ALIASES.get(canonical, []):
            pairs.append((canonical, _clean_text(alias)))
    # Longest first: "red can" should be checked before any generic "can".
    pairs.sort(key=lambda item: len(item[1]), reverse=True)
    return pairs


def normalize_label(text: str) -> str:
    """Normalize model output or user text into a canonical class name.

    Examples:
        "a photo of a red can" -> "coke_can"
        "tin can" -> "meat_can"
        "yellow banana" -> "banana"
    """
    t = _clean_text(text)
    if not t:
        return "object"

    for canonical, alias in _all_alias_pairs():
        if t == alias:
            return canonical

    # Fallback: if a detector returns a longer phrase containing a known alias.
    for canonical, alias in _all_alias_pairs():
        if len(alias) >= 4 and re.search(rf"\b{re.escape(alias)}\b", t):
            return canonical

    return re.sub(r"\s+", "_", t)


def extract_target_label(prompt: str, default: str = "object") -> str:
    """Extract the requested object class from a command-like prompt."""
    if not prompt:
        return default
    text = _clean_text(prompt)

    # Common tutorial wording: "Detect a banana and return pose".
    text = re.sub(r"\b(detect|find|pick|move|return|pose|coordinate|coordinates)\b", " ", text)
    text = re.sub(r"\b(to|into|onto|on|in|and|please)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    for canonical, alias in _all_alias_pairs():
        if re.search(rf"\b{re.escape(alias)}\b", text):
            return canonical
    return default


def detector_queries_for_target(
    target: str,
    include_templates: bool = True,
    stage: str | None = None,
    strict_semantic_prompts: bool = True,
) -> list[DetectorQuery]:
    """Return zero-shot detector queries for a canonical target.

    The returned query text is intentionally human/visual, while `canonical`
    preserves the object class requested by the task planner.  This prevents
    debug logs from showing generic labels such as "a_photo_of_a_object".
    """
    canonical = normalize_label(target)
    stage_name = str(stage or "normal_target_detection").strip().lower()
    fallback_stage = stage_name in {
        "relaxed_alias_detection",
        "relaxed",
        "generic_object_proposal",
        "generic",
        "depth_cluster_fallback",
    }
    if stage_name in {"relaxed_alias_detection", "relaxed"}:
        aliases = RELAXED_DETECTOR_ALIASES.get(canonical, DETECTOR_ALIASES.get(canonical))
    elif stage_name in {"generic_object_proposal", "generic"}:
        aliases = RELAXED_DETECTOR_ALIASES.get("object", DETECTOR_ALIASES.get("object"))
        canonical = "object"
    else:
        aliases = list(DETECTOR_ALIASES.get(canonical) or [])
        if not strict_semantic_prompts:
            aliases.extend(NON_STRICT_EXTRA_DETECTOR_ALIASES.get(canonical, []))
    if aliases is None:
        aliases = [_clean_text(target)] if target else []

    seen: set[str] = set()
    out: list[DetectorQuery] = []
    for alias in aliases:
        alias = _clean_text(alias)
        if not alias or alias in seen or alias == "object":
            continue
        seen.add(alias)
        weight = ALIAS_RANK_WEIGHT.get(alias, 1.0)
        if fallback_stage and alias in {"tin can", "small can", "metal can", "food can"}:
            weight *= 0.70
        out.append(DetectorQuery(alias, canonical, weight))
        if include_templates:
            templated = f"a photo of a {alias}"
            if templated not in seen:
                seen.add(templated)
                out.append(DetectorQuery(templated, canonical, weight * 0.95))
    return out


def labels_for_detector(target: str) -> list[str]:
    """Backward-compatible list of detector query strings."""
    return [q.text for q in detector_queries_for_target(target)]


def canonical_aliases_for_log(target: str) -> list[str]:
    """Readable alias list used in logs and README/debug output."""
    return [q.text for q in detector_queries_for_target(target, include_templates=False)]
