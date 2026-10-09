"""
Shot rules shared by the history backfill and the nightly ESPN reader, so both produce identical
rows: what kind of shot (2, 3 or free throw), how far, and which zone.

Zones: 2s are rim / short / mid / long; 3s are corner / above (the break) / heave; free throws FT.
"""
from __future__ import annotations

import math
import re

RIM_WORDS = re.compile(r"Dunk|Layup|Tip|Alley Oop|Putback|Finger Roll")
FEET = re.compile(r"(\d+)-foot")
HOOP_Y = 4.0  # ESPN's raw y coordinate of the rim (fitted to the shot distances in the text)
NO_COORD = -1000  # ESPN uses huge negative numbers for "no coordinate"


def kind_of(type_text: str, text: str, points_attempted: int | None, scoring: bool, score_value: int,
            x: float | None, y: float | None) -> str:
    """'FT', '2' or '3'. Uses ESPN's pointsAttempted when it has one; otherwise reads the shot."""
    if (type_text or "").startswith("Free Throw"):
        return "FT"
    if points_attempted in (2, 3):
        return str(points_attempted)
    if scoring:
        return "3" if score_value == 3 else "2"
    t = text or ""
    if "three point" in t.lower():
        return "3"
    m = FEET.search(t)
    if m:
        return "3" if int(m.group(1)) >= 23 else "2"
    if x is not None and y is not None and x > NO_COORD:
        d = math.hypot(x - 25, y - HOOP_Y)
        if d >= 23.5 or (y <= 14 and abs(x - 25) >= 21.5):
            return "3"
    return "2"


def distance(text: str, x: float | None, y: float | None) -> float | None:
    m = FEET.search(text or "")
    if m:
        return float(m.group(1))
    if x is None or y is None or x <= NO_COORD:
        return None
    return math.hypot(x - 25, y - HOOP_Y)


def zone_of(kind: str, dist: float | None, type_text: str, y: float | None) -> str:
    if kind == "FT":
        return "FT"
    d = dist if dist is not None else float("nan")
    if kind == "2":
        if RIM_WORDS.search(type_text or "") or d <= 4:
            return "rim"
        if d <= 10:
            return "short"
        if d <= 16:
            return "mid"
        return "long"
    if "Heave" in (type_text or "") or d >= 36:
        return "heave"
    if y is not None and y <= 14 and d < 23.6:
        return "corner"
    return "above"
