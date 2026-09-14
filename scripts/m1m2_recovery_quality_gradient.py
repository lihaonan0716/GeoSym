"""Deterministic M0 coordinate degradation for M1/M2 recovery experiments."""

from __future__ import annotations

import hashlib
import inspect
import random
from typing import Iterable

Point = tuple[int, int]


QUALITY_LEVELS = [
    {"level": "q0_native", "drop_fraction": 0.0, "jitter": 0, "false_points": 0},
    {"level": "q1_moderate_degradation", "drop_fraction": 0.2, "jitter": 1, "false_points": 1},
    {"level": "q2_strong_degradation", "drop_fraction": 0.4, "jitter": 2, "false_points": 2},
]


def _seed(sample_id: str, level: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}:{level}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def degrade_coords(coords: Iterable[tuple[int, int]], *, sample_id: str, level: str, seed: int = 20260911) -> list[Point]:
    """Apply deterministic input-side degradation without ground truth access."""
    spec = next((item for item in QUALITY_LEVELS if item["level"] == level), None)
    if spec is None:
        raise ValueError(f"Unknown quality level: {level}")
    points = sorted({(int(x), int(y)) for x, y in coords})
    rng = random.Random(_seed(sample_id, level, seed))
    if spec["drop_fraction"]:
        keep_count = max(2, round(len(points) * (1.0 - float(spec["drop_fraction"]))))
        points = sorted(rng.sample(points, min(len(points), keep_count)))
    jitter = int(spec["jitter"])
    if jitter:
        points = [(x + rng.randint(-jitter, jitter), y + rng.randint(-jitter, jitter)) for x, y in points]
    false_points = int(spec["false_points"])
    for _ in range(false_points):
        points.append((rng.randint(-8, 8), rng.randint(-8, 8)))
    return sorted(set(points))


def assert_degradation_does_not_accept_ground_truth() -> None:
    signature = inspect.signature(degrade_coords)
    forbidden = {"gt", "ground_truth", "truth", "answer", "label"}
    assert forbidden.isdisjoint(signature.parameters)
