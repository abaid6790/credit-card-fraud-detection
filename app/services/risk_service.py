"""Thin wrapper over ml.risk so the app layer doesn't import ml internals everywhere."""
from ml.risk import probability_to_risk

RISK_LEVEL_LABELS = {
    "VERY_LOW": "Very Low",
    "LOW": "Low",
    "MODERATE": "Moderate",
    "HIGH": "High",
    "CRITICAL": "Critical",
}


def score_and_level(probability: float) -> tuple[int, str]:
    return probability_to_risk(probability)


def display_label(risk_level: str) -> str:
    return RISK_LEVEL_LABELS.get(risk_level, risk_level.title())
