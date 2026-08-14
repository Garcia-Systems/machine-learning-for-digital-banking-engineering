from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from harbor_ml.capstone_incident_classifier import (
    CAPSTONE_INCIDENT_CLASSES, CAPSTONE_INCIDENT_FEATURES, PROHIBITED_CLASSIFIER_FIELDS,
    create_capstone_incident_metadata,
    evaluate_capstone_incident_classifier, load_capstone_classification_data,
    load_capstone_incident_artifact, load_chapter_26_timeline,
    predict_incident_probabilities, save_capstone_incident_artifact,
    score_capstone_incident_timeline, split_capstone_incident_data,
    train_capstone_incident_classifier, validate_incident_labels,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def trained():
    data = load_capstone_classification_data(ROOT / "data/harbor_incident_classes.csv")
    split = split_capstone_incident_data(data)
    model = train_capstone_incident_classifier(split)
    return data, split, model


def test_taxonomy_label_validation_and_feature_leakage():
    assert CAPSTONE_INCIDENT_CLASSES == ("normal", "vendor_degradation",
        "database_pressure", "traffic_spike", "application_regression")
    validate_incident_labels(CAPSTONE_INCIDENT_CLASSES)
    with pytest.raises(ValueError, match="unexpected incident_type.*network_routing_failure"):
        validate_incident_labels((*CAPSTONE_INCIDENT_CLASSES, "network_routing_failure"))
    assert CAPSTONE_INCIDENT_FEATURES == ("api_latency_ms", "error_rate",
        "db_connections", "queue_depth", "vendor_latency_ms", "requests_per_minute")
    assert not set(CAPSTONE_INCIDENT_FEATURES) & PROHIBITED_CLASSIFIER_FIELDS
    assert not {"incident_type", "phase", "confirmed_cause", "trace_diagnosis"} & set(CAPSTONE_INCIDENT_FEATURES)


def test_data_and_split_are_deterministic_and_represent_every_class(trained):
    data, split, _ = trained
    repeated = split_capstone_incident_data(data)
    assert data.features.shape == (300, len(CAPSTONE_INCIDENT_FEATURES))
    assert np.array_equal(split.train_indices, repeated.train_indices)
    assert np.array_equal(split.test_indices, repeated.test_indices)
    assert set(split.y_train) == set(split.y_test) == set(CAPSTONE_INCIDENT_CLASSES)
    assert split.strategy == "deterministic_stratified_random_split"


def test_model_probability_mapping_ambiguity_and_validation(trained):
    _, split, model = trained
    result = predict_incident_probabilities(model, split.X_test[0])
    raw = model.predict_proba(split.X_test[:1])[0]
    assert result.predicted_class in CAPSTONE_INCIDENT_CLASSES
    assert result.probabilities == pytest.approx(dict(zip(model.classes_, raw, strict=True)))
    assert np.isfinite(list(result.probabilities.values())).all()
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    ranked = sorted(result.probabilities.values(), reverse=True)
    assert result.probability_gap == pytest.approx(ranked[0] - ranked[1])
    assert predict_incident_probabilities(model, split.X_test[0], ambiguity_gap=1).ambiguous
    with pytest.raises(ValueError, match="feature contract mismatch"):
        predict_incident_probabilities(model, {"incident_type": 1})


def test_held_out_metrics_have_declared_shape_and_macro_f1(trained):
    _, split, model = trained
    evaluation = evaluate_capstone_incident_classifier(model, split)
    assert evaluation.confusion_matrix.shape == (5, 5)
    assert set(evaluation.per_class) == set(CAPSTONE_INCIDENT_CLASSES)
    assert sum(metric.support for metric in evaluation.per_class.values()) == len(split.y_test)
    assert evaluation.macro_f1 == pytest.approx(
        np.mean([evaluation.per_class[label].f1 for label in CAPSTONE_INCIDENT_CLASSES]))
    assert 0 <= evaluation.baseline_accuracy <= evaluation.accuracy <= 1


def test_timeline_scoring_is_model_derived_and_keeps_editorial_fields_separate(trained):
    _, _, model = trained
    rows = load_chapter_26_timeline(ROOT / "data/harbor_capstone_incident.csv")
    results = score_capstone_incident_timeline(model, rows)
    assert len(results) == len(rows) == 20
    for row, result in zip(rows, results, strict=True):
        direct = predict_incident_probabilities(model,
            {name: float(getattr(row, name)) for name in CAPSTONE_INCIDENT_FEATURES})
        assert result.prediction.probabilities == pytest.approx(direct.probabilities)
        assert result.editorial_class in CAPSTONE_INCIDENT_CLASSES


def test_artifact_metadata_and_probability_round_trip(trained, tmp_path):
    data, split, model = trained
    evaluation = evaluate_capstone_incident_classifier(model, split)
    metadata = create_capstone_incident_metadata(data, split, evaluation,
        now=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc))
    model_path, metadata_path = save_capstone_incident_artifact(model, metadata, tmp_path)
    reloaded = load_capstone_incident_artifact(model_path)
    assert np.allclose(model.predict_proba(split.X_test), reloaded.predict_proba(split.X_test))
    assert metadata_path.exists() and metadata.dataset_sha256 == data.dataset_sha256
    assert metadata.features == CAPSTONE_INCIDENT_FEATURES
    assert metadata.class_taxonomy == CAPSTONE_INCIDENT_CLASSES
    assert metadata.ambiguity_gap_threshold == 0.10
    assert metadata.model_version.startswith("harbor-capstone-incident-")
