"""Model-free embedding port contract for extracted content helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol


class EmbeddingPort(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one vector per input text, preserving order."""


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float | None:
    """Return cosine similarity, or None for malformed/non-comparable vectors."""

    if (
        not isinstance(left, Sequence)
        or not isinstance(right, Sequence)
        or isinstance(left, (str, bytes, bytearray))
        or isinstance(right, (str, bytes, bytearray))
        or len(left) != len(right)
        or len(left) == 0
    ):
        return None
    try:
        left_values = tuple(float(value) for value in left)
        right_values = tuple(float(value) for value in right)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in left_values + right_values):
        return None
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0 or right_norm == 0:
        return None
    result = sum(
        left_value * right_value
        for left_value, right_value in zip(left_values, right_values, strict=True)
    ) / (left_norm * right_norm)
    return result if math.isfinite(result) else None
