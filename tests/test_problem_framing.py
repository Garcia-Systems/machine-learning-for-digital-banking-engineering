from pathlib import Path

import pytest

from harbor_ml import (
    FUTURE_LATENCY,
    HARBOR_PROBLEMS,
    INCIDENT_CLASSIFICATION,
    REQUEST_FAILURE,
    TELEMETRY_ANOMALY,
    MLProblem,
    ProblemType,
    load_request_outcomes,
)

FIXTURE = Path(__file__).parents[1] / "data/harbor_request_outcomes.csv"


def make_problem(**changes: object) -> MLProblem:
    values = {
        "name": "Example",
        "engineering_question": "Will it fail?",
        "problem_type": ProblemType.BINARY_CLASSIFICATION,
        "features": ("signal",),
        "target": "failed",
    }
    values.update(changes)
    return MLProblem(**values)  # type: ignore[arg-type]


def test_defined_problems_are_valid_and_cover_expected_types() -> None:
    assert HARBOR_PROBLEMS == (
        REQUEST_FAILURE,
        INCIDENT_CLASSIFICATION,
        FUTURE_LATENCY,
        TELEMETRY_ANOMALY,
    )
    assert {problem.problem_type for problem in HARBOR_PROBLEMS} == {
        ProblemType.BINARY_CLASSIFICATION,
        ProblemType.MULTICLASS_CLASSIFICATION,
        ProblemType.REGRESSION,
        ProblemType.ANOMALY_DETECTION,
    }
    for problem in HARBOR_PROBLEMS:
        assert problem.validate() is None


def test_problem_type_distinguishes_supervised_behavior() -> None:
    assert ProblemType.BINARY_CLASSIFICATION.is_supervised
    assert ProblemType.MULTICLASS_CLASSIFICATION.is_supervised
    assert ProblemType.REGRESSION.is_supervised
    assert not ProblemType.ANOMALY_DETECTION.is_supervised
    assert not ProblemType.CLUSTERING.is_supervised


def test_supervised_problem_requires_target() -> None:
    with pytest.raises(ValueError, match="supervised problems require a target"):
        make_problem(target=None)


def test_unsupervised_problem_rejects_target() -> None:
    with pytest.raises(ValueError, match="must not define a target"):
        make_problem(problem_type=ProblemType.CLUSTERING)


def test_duplicate_features_are_rejected() -> None:
    with pytest.raises(ValueError, match="feature names must be unique"):
        make_problem(features=("signal", "signal"))


def test_target_in_features_is_rejected_as_obvious_leakage() -> None:
    with pytest.raises(
        ValueError,
        match="target 'failed' cannot also appear in the feature set",
    ):
        make_problem(features=("signal", "failed"))


def test_empty_features_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one feature is required"):
        make_problem(features=())


def test_fixture_has_expected_typed_educational_structure() -> None:
    outcomes = load_request_outcomes(FIXTURE)

    assert len(outcomes) == 30
    assert outcomes[0].timestamp.isoformat() == "2026-08-12T09:00:00+00:00"
    assert outcomes[-1].timestamp > outcomes[0].timestamp
    assert {outcome.request_failed for outcome in outcomes} == {0, 1}
    assert all(isinstance(outcome.vendor_latency_ms, int) for outcome in outcomes)
    assert all(outcome.retry_count >= 0 for outcome in outcomes)


def test_fixture_loader_rejects_invalid_binary_label(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(
        "timestamp,vendor_latency_ms,queue_depth,db_connections,retry_count,"
        "request_failed\n2026-08-12T09:00:00Z,220,12,31,0,2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="request_failed must be 0 or 1"):
        load_request_outcomes(path)
