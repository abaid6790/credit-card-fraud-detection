"""
Converts a model's fraud probability into a 0-100 risk score and a risk level
label. This is a deterministic linear mapping of the actual model output —
never a random or arbitrary number.
"""
from __future__ import annotations


def probability_to_risk(probability: float) -> tuple[int, str]:
    score = round(max(0.0, min(1.0, probability)) * 100)
    if score <= 20:
        level = "VERY_LOW"
    elif score <= 40:
        level = "LOW"
    elif score <= 60:
        level = "MODERATE"
    elif score <= 80:
        level = "HIGH"
    else:
        level = "CRITICAL"
    return score, level
