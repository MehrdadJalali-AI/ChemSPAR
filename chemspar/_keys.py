"""Canonical keys for periodic contact instances (copied unchanged from the paper code, ``periodic.canonical``)."""
from __future__ import annotations

from typing import Tuple

Img = Tuple[int, ...]  # three integers (a, b, c)
Key = Tuple[int, int, Img]


def canonical(i: int, j: int, img: Img) -> Key:
    """Canonical key of the undirected contact between atom ``i`` (home cell) and atom ``j`` in image ``img``.

    ``(i, j, img)`` with ``i < j``; ``(j, i, -img)`` is the same instance. A self-image contact is ``(i, i, img)``
    with ``img`` lexicographically greater than ``(0, 0, 0)``.
    """
    img = tuple(int(x) for x in img)
    neg = tuple(-x for x in img)
    if i < j:
        return (i, j, img)
    if i > j:
        return (j, i, neg)
    return (i, i, img if img > (0, 0, 0) else neg)
