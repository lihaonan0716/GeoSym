"""
Plain-table baseline for coordinate injection.

Reuses the same high-quality coordinates extracted by GeoSym but injects
them as a plain, unstructured list — no confidence framing, no derived
chart facts, no structured section headers.

This baseline isolates the value of GeoSym's *structured prompt engineering*:
if structured fusion outperforms plain-table injection, the formatting and
fact organization matter in addition to the coordinates themselves.
"""

from __future__ import annotations

from typing import List, Tuple


class PlainTableBaseline:
    """Format GeoSym-extracted coordinates as a plain, unstructured text list."""

    def format_for_prompt(self, coords: List[Tuple[int, int]]) -> str:
        """Return a minimal comma-separated coordinate string."""
        if not coords:
            return "(no points detected)"
        return ", ".join(f"({x}, {y})" for x, y in sorted(coords, key=lambda p: (p[0], p[1])))
