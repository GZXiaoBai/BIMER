from __future__ import annotations

import re

_USED_SWAP_PATTERN = re.compile(r"\bused\s*=\s*([0-9]+(?:\.[0-9]+)?)([MG])\b")


def parse_swap_megabytes(swap_usage: str) -> float:
    match = _USED_SWAP_PATTERN.search(swap_usage)
    if match is None:
        raise ValueError(f"unable to parse macOS swap usage: {swap_usage!r}")
    value = float(match.group(1))
    if match.group(2) == "G":
        value *= 1024.0
    return value


def evaluate_system_swap(
    before: str,
    after: str,
    *,
    max_initial_mb: float = 256.0,
    max_increase_mb: float = 0.0,
) -> dict[str, float | bool]:
    before_used = parse_swap_megabytes(before)
    after_used = parse_swap_megabytes(after)
    increase = round(after_used - before_used, 3)
    clean_start = before_used <= max_initial_mb
    unchanged = increase <= max_increase_mb
    return {
        "before_used_mb": before_used,
        "after_used_mb": after_used,
        "increase_mb": increase,
        "clean_start": clean_start,
        "unchanged": unchanged,
        "passed": clean_start and unchanged,
    }
