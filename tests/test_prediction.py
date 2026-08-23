from ml.risk import probability_to_risk


def test_risk_score_boundaries():
    assert probability_to_risk(0.0) == (0, "VERY_LOW")
    assert probability_to_risk(0.2) == (20, "VERY_LOW")
    assert probability_to_risk(0.5) == (50, "MODERATE")
    assert probability_to_risk(0.95) == (95, "CRITICAL")
    assert probability_to_risk(1.0) == (100, "CRITICAL")


def test_risk_score_is_deterministic():
    # Same probability must always produce the same score - never random.
    results = {probability_to_risk(0.7273) for _ in range(20)}
    assert len(results) == 1


def test_risk_score_clamped_to_valid_range():
    score, _ = probability_to_risk(1.5)
    assert score == 100
    score, _ = probability_to_risk(-0.5)
    assert score == 0
