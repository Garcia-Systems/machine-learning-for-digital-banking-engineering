from harbor_ml import find_threshold_violations


def test_normal_observation_has_no_violations() -> None:
    observation = {"api_latency_ms": 180, "error_rate": 0.004}
    thresholds = {"api_latency_ms": 500, "error_rate": 0.05}

    assert find_threshold_violations(observation, thresholds) == []


def test_incident_observation_reports_each_exceeded_threshold() -> None:
    observation = {"api_latency_ms": 2400, "error_rate": 0.087}
    thresholds = {"api_latency_ms": 500, "error_rate": 0.05}

    assert find_threshold_violations(observation, thresholds) == [
        "api_latency_ms: observed 2400 exceeds threshold 500",
        "error_rate: observed 0.087 exceeds threshold 0.05",
    ]


def test_missing_metric_is_not_treated_as_a_violation() -> None:
    assert find_threshold_violations({}, {"api_latency_ms": 500}) == []

