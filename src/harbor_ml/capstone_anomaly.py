"""Reproducible healthy-baseline anomaly training for Chapter 28."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from .capstone_dataset import DATASET_NAME, DATASET_VERSION, CapstoneSources

MODEL_NAME = "harbor-capstone-anomaly"
CAPSTONE_ANOMALY_FEATURES = (
    "api_latency_ms", "error_rate", "db_connections", "queue_depth",
    "vendor_latency_ms", "requests_per_minute", "retry_count",
)
PROHIBITED_ANOMALY_FIELDS = frozenset({
    "incident_type", "request_failed", "future_trace_duration", "final_status",
    "diagnosis", "incident_phase", "member_id", "account_number", "email",
    "name", "social_security_number",
})
DEFAULT_CONTAMINATION = 0.05
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_ESTIMATORS = 100
# The committed teaching fixture has exactly three pre-incident observations. This
# is a fixture-specific guard, not production sizing advice.
MINIMUM_BASELINE_ROWS = 3


@dataclass(frozen=True)
class CapstoneAnomalyObservation:
    timestamp: datetime
    features: Mapping[str, float]
    phase: str


@dataclass(frozen=True)
class CapstoneAnomalyResult:
    timestamp: datetime
    anomaly_score: float
    is_anomaly: bool
    phase: str


@dataclass(frozen=True)
class DetectionBehavior:
    healthy_eval_rows: int
    healthy_flagged_rows: int
    healthy_anomaly_rate: float
    first_anomaly_timestamp: datetime | None
    reference_incident_timestamp: datetime | None
    lead_time_seconds: float | None


@dataclass(frozen=True)
class CapstoneAnomalyMetadata:
    model_name: str
    model_version: str
    model_type: str
    trained_at: str
    dataset_name: str
    dataset_version: str
    dataset_sha256: str
    baseline_start: str
    baseline_end: str
    baseline_rows: int
    features: tuple[str, ...]
    contamination: float
    random_state: int
    n_estimators: int
    training_dependency_versions: Mapping[str, str]
    training_metadata: Mapping[str, object]
    synthetic_evaluation_metadata: Mapping[str, object]


def build_capstone_anomaly_timeline(sources: CapstoneSources) -> tuple[CapstoneAnomalyObservation, ...]:
    """Create the point-in-time contract from Chapter 27's validated source views."""
    database = {row.timestamp: row for row in sources.database}
    vendor = {row.timestamp: row for row in sources.vendor}
    observations = []
    for app in sources.application:
        db, integration = database[app.timestamp], vendor[app.timestamp]
        minute = app.timestamp.minute
        phase = ("healthy" if minute <= 4 else "early_degradation" if minute <= 10
                 else "vendor_degradation" if minute <= 18 else "compound_pressure"
                 if minute <= 30 else "recovery")
        features = {
            "api_latency_ms": float(app.api_latency_ms), "error_rate": float(app.error_rate),
            "db_connections": float(db.db_connections), "queue_depth": float(app.queue_depth),
            "vendor_latency_ms": float(integration.vendor_latency_ms),
            "requests_per_minute": float(app.requests_per_minute),
            "retry_count": float(integration.retry_count),
        }
        observations.append(CapstoneAnomalyObservation(app.timestamp, features, phase))
    return tuple(observations)


def select_anomaly_baseline(observations: Sequence[CapstoneAnomalyObservation], *,
                            baseline_end: datetime | None = None) -> tuple[CapstoneAnomalyObservation, ...]:
    """Select only the explicit synthetic healthy phase, optionally bounded by time."""
    return tuple(row for row in observations if row.phase == "healthy" and
                 (baseline_end is None or row.timestamp <= baseline_end))


def build_anomaly_feature_matrix(observations: Sequence[CapstoneAnomalyObservation]) -> np.ndarray:
    """Select only the ordered prediction-time feature contract (never labels)."""
    rows = []
    for observation in observations:
        if set(observation.features) != set(CAPSTONE_ANOMALY_FEATURES):
            raise ValueError("anomaly feature contract mismatch")
        rows.append([float(observation.features[name]) for name in CAPSTONE_ANOMALY_FEATURES])
    return np.asarray(rows, dtype=np.float64).reshape(len(rows), len(CAPSTONE_ANOMALY_FEATURES))


def validate_anomaly_baseline(baseline: Sequence[CapstoneAnomalyObservation], *,
                              evaluation: Sequence[CapstoneAnomalyObservation] = (),
                              minimum_rows: int = MINIMUM_BASELINE_ROWS) -> None:
    if len(baseline) < minimum_rows:
        raise ValueError(f"anomaly baseline must contain at least {minimum_rows} healthy observations")
    if any(row.phase != "healthy" for row in baseline):
        raise ValueError("anomaly baseline contains a known degraded observation")
    if PROHIBITED_ANOMALY_FIELDS.intersection(CAPSTONE_ANOMALY_FEATURES):
        raise ValueError("anomaly feature contract contains prohibited fields")
    matrix = build_anomaly_feature_matrix(baseline)
    if not np.isfinite(matrix).all():
        raise ValueError("anomaly baseline features must be finite")
    if evaluation and max(row.timestamp for row in baseline) >= min(row.timestamp for row in evaluation):
        raise ValueError("baseline timestamps must precede incident evaluation timestamps")


def build_capstone_anomaly_detector(*, contamination: float = DEFAULT_CONTAMINATION,
                                     random_state: int = DEFAULT_RANDOM_STATE,
                                     n_estimators: int = DEFAULT_N_ESTIMATORS) -> Pipeline:
    if not 0 < contamination <= 0.5:
        raise ValueError("contamination must be in (0, 0.5]")
    return Pipeline([("detector", IsolationForest(n_estimators=n_estimators,
        contamination=contamination, random_state=random_state))])


def train_capstone_anomaly_detector(baseline: Sequence[CapstoneAnomalyObservation], **config) -> Pipeline:
    validate_anomaly_baseline(baseline)
    return build_capstone_anomaly_detector(**config).fit(build_anomaly_feature_matrix(baseline))


def score_capstone_timeline(model: Pipeline, observations: Sequence[CapstoneAnomalyObservation]
                             ) -> tuple[CapstoneAnomalyResult, ...]:
    matrix = build_anomaly_feature_matrix(observations)
    raw = model.decision_function(matrix)
    predictions = model.predict(matrix)
    return tuple(CapstoneAnomalyResult(row.timestamp, -float(score), prediction == -1, row.phase)
                 for row, score, prediction in zip(observations, raw, predictions))


def evaluate_detection_behavior(results: Sequence[CapstoneAnomalyResult], *,
                                reference_incident_timestamp: datetime | None = None) -> DetectionBehavior:
    # Initial ``healthy`` rows are the fitted baseline, not evaluation. The fixture
    # has no later steady-state holdout, so recovery is the clearly named proxy.
    healthy = [row for row in results if row.phase == "recovery"]
    flagged = int(sum(bool(row.is_anomaly) for row in healthy))
    first = next((row.timestamp for row in results
                  if row.phase != "healthy" and row.is_anomaly), None)
    lead = ((reference_incident_timestamp - first).total_seconds()
            if first is not None and reference_incident_timestamp is not None else None)
    return DetectionBehavior(len(healthy), flagged, flagged / len(healthy) if healthy else 0.0,
                             first, reference_incident_timestamp, lead)


def fingerprint_anomaly_dataset(observations: Sequence[CapstoneAnomalyObservation]) -> str:
    payload = [{"timestamp": row.timestamp.isoformat(), "features": dict(row.features),
                "phase": row.phase} for row in observations]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def create_anomaly_metadata(baseline: Sequence[CapstoneAnomalyObservation],
                            timeline: Sequence[CapstoneAnomalyObservation], behavior: DetectionBehavior,
                            *, contamination: float = DEFAULT_CONTAMINATION,
                            random_state: int = DEFAULT_RANDOM_STATE,
                            n_estimators: int = DEFAULT_N_ESTIMATORS,
                            now: Callable[[], datetime] | None = None) -> CapstoneAnomalyMetadata:
    instant = (now or (lambda: datetime.now(timezone.utc)))()
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("training timestamp must be timezone-aware")
    digest = fingerprint_anomaly_dataset(timeline)
    return CapstoneAnomalyMetadata(
        MODEL_NAME, f"{MODEL_NAME}-{digest[:8]}", "IsolationForest",
        instant.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"), DATASET_NAME,
        DATASET_VERSION, digest, baseline[0].timestamp.isoformat(), baseline[-1].timestamp.isoformat(),
        len(baseline), CAPSTONE_ANOMALY_FEATURES, contamination, random_state, n_estimators,
        {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
         "joblib": joblib.__version__},
        {"baseline_rule": "phase == healthy", "score_orientation": "-decision_function; higher is unusual"},
        {"healthy_eval_rows": behavior.healthy_eval_rows,
         "healthy_anomaly_rate": behavior.healthy_anomaly_rate,
         "first_anomaly_timestamp": behavior.first_anomaly_timestamp.isoformat()
         if behavior.first_anomaly_timestamp else None,
         "reference_incident_timestamp": behavior.reference_incident_timestamp.isoformat()
         if behavior.reference_incident_timestamp else None, "lead_time_seconds": behavior.lead_time_seconds},
    )


def save_capstone_anomaly_artifact(model: Pipeline, metadata: CapstoneAnomalyMetadata,
                                    output_dir: str | Path) -> tuple[Path, Path]:
    directory = Path(output_dir); directory.mkdir(parents=True, exist_ok=True)
    model_path, metadata_path = directory / "model.joblib", directory / "metadata.json"
    joblib.dump(model, model_path)
    metadata_path.write_text(json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return model_path, metadata_path


def load_capstone_anomaly_artifact(path: str | Path) -> Pipeline:
    model = joblib.load(Path(path))
    if not isinstance(model, Pipeline) or "detector" not in model.named_steps:
        raise TypeError("trusted artifact does not contain the capstone anomaly pipeline")
    return model
