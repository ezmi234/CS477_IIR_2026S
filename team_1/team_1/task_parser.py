import re

from .config import DESTINATION_ALIASES, OBJECT_ALIASES
from .models import PickPlaceTask


def _normalized_words(text):
    return re.sub(r'[^a-z0-9_ ]+', ' ', text.lower())


def _alias_map(alias_groups):
    aliases = {}
    for canonical, names in alias_groups.items():
        for name in names:
            aliases[_normalized_words(name).strip()] = canonical
    return aliases


def _ordered_alias_hits(text, alias_groups):
    alias_to_name = _alias_map(alias_groups)
    hits = []
    for alias, canonical in sorted(alias_to_name.items(), key=lambda item: len(item[0]), reverse=True):
        if not alias:
            continue
        for match in re.finditer(r'\b' + re.escape(alias) + r'\b', text):
            hits.append((match.start(), match.end(), canonical, len(alias)))

    hits.sort(key=lambda item: (item[0], -item[3]))
    selected = []
    occupied_until = -1
    for start, end, canonical, _ in hits:
        if start < occupied_until:
            continue
        selected.append((start, end, canonical))
        occupied_until = end
    return selected


def _strip_task_fillers(text):
    return re.sub(
        r'\b(move|put|place|pick|and|then|a|an|the|to|into|onto|in|on|please)\b',
        ' ',
        text,
    )


def parse_task_command(text):
    """Convert a natural-language command into ordered (object, destination) tasks."""
    tasks = []
    clauses = [c.strip() for c in re.split(r'[.;]', text) if c.strip()]

    for clause in clauses:
        normalized = _normalized_words(clause)
        destination_hits = _ordered_alias_hits(normalized, DESTINATION_ALIASES)
        if not destination_hits:
            continue

        previous_destination_end = 0
        for destination_start, destination_end, destination in destination_hits:
            object_region = normalized[previous_destination_end:destination_start]
            object_region = _strip_task_fillers(object_region)
            object_hits = _ordered_alias_hits(object_region, OBJECT_ALIASES)
            tasks.extend(
                PickPlaceTask(object_name, destination, source_command=text)
                for _, _, object_name in object_hits
            )
            previous_destination_end = destination_end

    return tasks


def prioritize_tasks(tasks):
    # Preserve command order. Reordering can make the arm disturb nearby objects
    # before their queued pick, which makes multi-task debugging ambiguous.
    return list(tasks)
