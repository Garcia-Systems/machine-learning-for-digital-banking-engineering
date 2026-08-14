"""Reproducible multi-class incident-pattern training for Chapter 29."""

from __future__ import annotations

import json
import platform
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import joblib
import numpy as np
import sklearn
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .capstone_dataset import DATASET_NAME, DATASET_VERSION, sha256_file
from .capstone_incident import CapstoneTelemetry, classify_incident_phase, load_capstone_incident
from .incident_classifier import INCIDENT_CLASSES, load_incident_dataset

MODEL_NAME = "harbor-capstone-incident"
TARGET = "incident_type"
CAPSTONE_INCIDENT_CLASSES = INCIDENT_CLASSES
# These are the trustworthy Chapter 27 raw telemetry fields that are also available
# throughout the labeled Chapter 5 history. Rolling fields are intentionally omitted:
# the older fixture cannot reconstruct them consistently.
CAPSTONE_INCIDENT_FEATURES = (
    "api_latency_ms", "error_rate", "db_connections", "queue_depth",
    "vendor_latency_ms", "requests_per_minute",
)
PROHIBITED_CLASSIFIER_FIELDS = frozenset({
    TARGET, "request_failed", "confirmed_diagnosis", "confirmed_cause",
    "trace_final_duration", "trace_diagnosis", "future_status", "final_status",
    "incident_phase", "phase", "member_id", "account_number", "email", "name",
    "social_security_number",
})
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.25
DEFAULT_AMBIGUITY_GAP = 0.10


@dataclass(frozen=True)
class CapstoneClassificationData:
    timestamps: tuple[datetime, ...]
    features: NDArray[np.float64]
    targets: NDArray[np.str_]
    dataset_path: Path
    dataset_sha256: str


@dataclass(frozen=True)
class CapstoneIncidentSplit:
    X_train: NDArray[np.float64]
    X_test: NDArray[np.float64]
    y_train: NDArray[np.str_]
    y_test: NDArray[np.str_]
    train_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    strategy: str


@dataclass(frozen=True)
class PerClassMetric:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class CapstoneIncidentEvaluation:
    accuracy: float
    baseline_accuracy: float
    macro_f1: float
    weighted_f1: float
    predictions: NDArray[np.str_]
    confusion_matrix: NDArray[np.int64]
    per_class: Mapping[str, PerClassMetric]


@dataclass(frozen=True)
class CapstoneIncidentPrediction:
    predicted_class: str
    probabilities: dict[str, float]
    ambiguous: bool
    top_probability: float
    second_probability: float
    probability_gap: float


@dataclass(frozen=True)
class TimelineIncidentResult:
    timestamp: datetime
    editorial_phase: str
    editorial_class: str
    prediction: CapstoneIncidentPrediction


@dataclass(frozen=True)
class CapstoneIncidentMetadata:
    model_name: str
    model_version: str
    model_type: str
    trained_at: str
    dataset_name: str
    dataset_version: str
    dataset_sha256: str
    features: tuple[str, ...]
    target: str
    class_taxonomy: tuple[str, ...]
    train_rows: int
    test_rows: int
    class_counts: Mapping[str, int]
    training_class_counts: Mapping[str, int]
    split_strategy: str
    test_size: float
    random_state: int
    ambiguity_gap_threshold: float
    accuracy: float
    baseline_accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_metrics: Mapping[str, Mapping[str, float | int]]
    scikit_learn_version: str
    python_version: str


def validate_incident_labels(labels: Sequence[str] | NDArray[np.str_]) -> None:
    unexpected = sorted(set(str(label) for label in labels) - set(CAPSTONE_INCIDENT_CLASSES))
    if unexpected:
        raise ValueError("unexpected incident_type labels: " + ", ".join(unexpected))
    if len(labels) == 0:
        raise ValueError("incident_type labels cannot be empty")


def load_capstone_classification_data(path: str | Path) -> CapstoneClassificationData:
    """Load labeled history using Chapter 27's prediction-time raw feature contract."""
    dataset = Path(path)
    observations = load_incident_dataset(dataset)
    targets = np.asarray([row.incident_type for row in observations], dtype=np.str_)
    validate_incident_labels(targets)
    features = np.asarray([[float(getattr(row, name)) for name in CAPSTONE_INCIDENT_FEATURES]
                           for row in observations], dtype=np.float64)
    if not np.isfinite(features).all():
        raise ValueError("classifier features must be finite")
    if set(CAPSTONE_INCIDENT_FEATURES) & PROHIBITED_CLASSIFIER_FIELDS:
        raise ValueError("classifier feature contract contains prohibited fields")
    return CapstoneClassificationData(tuple(row.timestamp for row in observations), features,
                                      targets, dataset, sha256_file(dataset))


def split_capstone_incident_data(data: CapstoneClassificationData, *,
                                 test_size: float = DEFAULT_TEST_SIZE,
                                 random_state: int = DEFAULT_RANDOM_STATE) -> CapstoneIncidentSplit:
    """Use a disclosed stratified split because strict time ordering is class-fragile."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    indices = np.arange(len(data.targets), dtype=np.int64)
    train, test = train_test_split(indices, test_size=test_size, random_state=random_state,
                                   stratify=data.targets)
    for selected, name in ((train, "training"), (test, "test")):
        if set(data.targets[selected]) != set(CAPSTONE_INCIDENT_CLASSES):
            raise ValueError(f"{name} split does not represent every canonical class")
    return CapstoneIncidentSplit(data.features[train], data.features[test], data.targets[train],
                                 data.targets[test], train, test,
                                 "deterministic_stratified_random_split")


def build_capstone_incident_pipeline(*, random_state: int = DEFAULT_RANDOM_STATE) -> Pipeline:
    # In scikit-learn 1.9 LogisticRegression handles multiclass automatically;
    # specifying the removed/deprecated multi_class parameter would be incorrect.
    return Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(
        solver="lbfgs", max_iter=1_000, random_state=random_state))])


def train_capstone_incident_classifier(split: CapstoneIncidentSplit, *,
                                       random_state: int = DEFAULT_RANDOM_STATE) -> Pipeline:
    validate_incident_labels(split.y_train)
    return build_capstone_incident_pipeline(random_state=random_state).fit(split.X_train,
                                                                           split.y_train)


def predict_incident_probabilities(model: Pipeline, features: Sequence[float] | Mapping[str, float], *,
                                   ambiguity_gap: float = DEFAULT_AMBIGUITY_GAP
                                   ) -> CapstoneIncidentPrediction:
    if not 0 <= ambiguity_gap <= 1:
        raise ValueError("ambiguity_gap must be between 0 and 1")
    if isinstance(features, Mapping):
        missing = [name for name in CAPSTONE_INCIDENT_FEATURES if name not in features]
        unexpected = [name for name in features if name not in CAPSTONE_INCIDENT_FEATURES]
        if missing or unexpected:
            raise ValueError("classifier feature contract mismatch")
        values = [features[name] for name in CAPSTONE_INCIDENT_FEATURES]
    else:
        values = list(features)
    row = np.asarray([values], dtype=np.float64)
    if row.shape != (1, len(CAPSTONE_INCIDENT_FEATURES)) or not np.isfinite(row).all():
        raise ValueError("classifier row must contain the ordered finite feature contract")
    scores = model.predict_proba(row)[0]
    mapped = {str(label): float(score) for label, score in
              zip(model.classes_, scores, strict=True)}
    ranked = sorted(mapped.items(), key=lambda item: (-item[1], item[0]))
    gap = ranked[0][1] - ranked[1][1]
    return CapstoneIncidentPrediction(ranked[0][0], mapped, gap < ambiguity_gap,
                                      ranked[0][1], ranked[1][1], gap)


def evaluate_capstone_incident_classifier(model: Pipeline, split: CapstoneIncidentSplit
                                            ) -> CapstoneIncidentEvaluation:
    predictions = np.asarray(model.predict(split.X_test), dtype=np.str_)
    precision, recall, f1, support = precision_recall_fscore_support(
        split.y_test, predictions, labels=CAPSTONE_INCIDENT_CLASSES, zero_division=0)
    metrics = {label: PerClassMetric(float(precision[i]), float(recall[i]), float(f1[i]),
                                     int(support[i]))
               for i, label in enumerate(CAPSTONE_INCIDENT_CLASSES)}
    majority = Counter(split.y_train).most_common(1)[0][0]
    baseline = np.full(split.y_test.shape, majority, dtype=split.y_test.dtype)
    return CapstoneIncidentEvaluation(
        float(accuracy_score(split.y_test, predictions)),
        float(accuracy_score(split.y_test, baseline)),
        float(f1_score(split.y_test, predictions, labels=CAPSTONE_INCIDENT_CLASSES,
                       average="macro", zero_division=0)),
        float(f1_score(split.y_test, predictions, labels=CAPSTONE_INCIDENT_CLASSES,
                       average="weighted", zero_division=0)), predictions,
        np.asarray(confusion_matrix(split.y_test, predictions,
                                   labels=CAPSTONE_INCIDENT_CLASSES), dtype=np.int64), metrics)


def score_capstone_incident_timeline(model: Pipeline, rows: Sequence[CapstoneTelemetry], *,
                                     ambiguity_gap: float = DEFAULT_AMBIGUITY_GAP
                                     ) -> tuple[TimelineIncidentResult, ...]:
    results = []
    for row in rows:
        features = {name: float(getattr(row, name)) for name in CAPSTONE_INCIDENT_FEATURES}
        # Evaluation-only editorial class means initiating scenario, never an input.
        editorial = "normal" if row.timestamp.minute <= 4 or row.timestamp.minute >= 32 else "vendor_degradation"
        results.append(TimelineIncidentResult(row.timestamp, classify_incident_phase(row.timestamp),
            editorial, predict_incident_probabilities(model, features, ambiguity_gap=ambiguity_gap)))
    return tuple(results)


def create_capstone_incident_metadata(data: CapstoneClassificationData,
                                      split: CapstoneIncidentSplit,
                                      evaluation: CapstoneIncidentEvaluation, *,
                                      ambiguity_gap: float = DEFAULT_AMBIGUITY_GAP,
                                      random_state: int = DEFAULT_RANDOM_STATE,
                                      test_size: float = DEFAULT_TEST_SIZE,
                                      now: Callable[[], datetime] | None = None
                                      ) -> CapstoneIncidentMetadata:
    instant = (now or (lambda: datetime.now(timezone.utc)))()
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("training timestamp must be timezone-aware")
    class_counts = Counter(data.targets)
    training_counts = Counter(split.y_train)
    return CapstoneIncidentMetadata(MODEL_NAME, f"{MODEL_NAME}-{data.dataset_sha256[:8]}",
        "StandardScaler+LogisticRegression", instant.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        DATASET_NAME, DATASET_VERSION, data.dataset_sha256, CAPSTONE_INCIDENT_FEATURES, TARGET,
        CAPSTONE_INCIDENT_CLASSES, len(split.y_train), len(split.y_test),
        {label: int(class_counts[label]) for label in CAPSTONE_INCIDENT_CLASSES},
        {label: int(training_counts[label]) for label in CAPSTONE_INCIDENT_CLASSES},
        split.strategy, test_size, random_state, ambiguity_gap, evaluation.accuracy,
        evaluation.baseline_accuracy, evaluation.macro_f1, evaluation.weighted_f1,
        {label: asdict(evaluation.per_class[label]) for label in CAPSTONE_INCIDENT_CLASSES},
        sklearn.__version__, platform.python_version())


def save_capstone_incident_artifact(model: Pipeline, metadata: CapstoneIncidentMetadata,
                                    output_dir: str | Path) -> tuple[Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    model_path, metadata_path = directory / "model.joblib", directory / "metadata.json"
    joblib.dump(model, model_path)
    metadata_path.write_text(json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    return model_path, metadata_path


def load_capstone_incident_artifact(path: str | Path) -> Pipeline:
    model = joblib.load(Path(path))
    if not isinstance(model, Pipeline):
        raise TypeError("trusted artifact does not contain a scikit-learn Pipeline")
    return model


def load_chapter_26_timeline(path: str | Path) -> list[CapstoneTelemetry]:
    return load_capstone_incident(path)
