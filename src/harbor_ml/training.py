"""Repeatable artifact training for Harbor's integration-failure model."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from .integration_failure_model import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PREDICTION_FEATURES,
    TARGET,
    build_integration_features,
    build_integration_pipeline,
    build_integration_targets,
    load_integration_requests,
    split_integration_dataset,
    train_integration_model,
)

MODEL_NAME = "harbor-integration-failure"
MINIMUM_TRAINING_ROWS = 20
PROHIBITED_LEAKAGE_COLUMNS = frozenset(
    {TARGET, "final_http_status", "failure_reason", "response_duration_ms"}
)
REQUIRED_COLUMNS = ("timestamp", *PREDICTION_FEATURES, TARGET)


@dataclass(frozen=True)
class TrainingConfig:
    """The small set of choices that materially controls this teaching run."""

    test_size: float = 0.25
    random_state: int = 42
    classification_threshold: float = 0.50
    max_iter: int = 1_000

    def __post_init__(self) -> None:
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if not 0 <= self.classification_threshold <= 1:
            raise ValueError("classification_threshold must be between 0 and 1")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")


@dataclass(frozen=True)
class EvaluationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    baseline_accuracy: float


@dataclass(frozen=True)
class TrainingMetadata:
    model_name: str
    model_version: str
    trained_at: str
    dataset_path: str
    dataset_sha256: str
    dataset_rows: int
    training_rows: int
    test_rows: int
    numerical_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    target: str
    test_size: float
    random_state: int
    classification_threshold: float
    max_iter: int
    model_type: str
    python_version: str
    scikit_learn_version: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class TrainingResult:
    pipeline: Pipeline
    metrics: EvaluationMetrics
    metadata: TrainingMetadata
    training_rows: int
    test_rows: int
    dataset_sha256: str


def calculate_file_sha256(path: str | Path) -> str:
    """Identify the exact file bytes used by a run; this is not a quality proof."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(64 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_training_dataset(path: str | Path, *, minimum_rows: int = MINIMUM_TRAINING_ROWS):
    """Validate the focused Chapter 16 CSV contract before any fitting occurs."""
    dataset = Path(path)
    with dataset.open(newline="", encoding="utf-8") as source:
        reader = csv.reader(source)
        header = next(reader, None)
    if not header:
        raise ValueError("training dataset must contain a header and observations")
    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates:
        raise ValueError("duplicate dataset columns: " + ", ".join(duplicates))
    missing = [name for name in REQUIRED_COLUMNS if name not in header]
    if missing:
        raise ValueError("missing required training columns: " + ", ".join(missing))
    leaked = PROHIBITED_LEAKAGE_COLUMNS.intersection(PREDICTION_FEATURES)
    if leaked:
        raise ValueError("declared features contain post-outcome leakage: " + ", ".join(sorted(leaked)))

    observations = load_integration_requests(dataset)
    if len(observations) < minimum_rows:
        raise ValueError(f"training dataset must contain at least {minimum_rows} observations")
    targets = {row.request_failed for row in observations}
    if targets != {0, 1}:
        raise ValueError("training target must represent both binary classes")
    return observations


def evaluate_trained_model(
    pipeline: Pipeline, X_test, y_test, *, threshold: float
) -> EvaluationMetrics:
    probabilities = pipeline.predict_proba(X_test)[:, list(pipeline.classes_).index(1)]
    predictions = (probabilities >= threshold).astype(np.int64)
    majority = int(np.mean(y_test) >= 0.5)
    metrics = EvaluationMetrics(
        accuracy=float(accuracy_score(y_test, predictions)),
        precision=float(precision_score(y_test, predictions, zero_division=0)),
        recall=float(recall_score(y_test, predictions, zero_division=0)),
        f1=float(f1_score(y_test, predictions, zero_division=0)),
        baseline_accuracy=float(accuracy_score(y_test, np.full(len(y_test), majority))),
    )
    if not all(isfinite(value) for value in asdict(metrics).values()):
        raise ValueError("evaluation gate rejected non-finite metrics")
    return metrics


def train_integration_failure_model(
    dataset_path: str | Path,
    config: TrainingConfig = TrainingConfig(),
    *,
    now: Callable[[], datetime] | None = None,
) -> TrainingResult:
    """Validate, fingerprint, split, fit, and evaluate without writing output."""
    path = Path(dataset_path)
    observations = validate_training_dataset(path)
    dataset_sha256 = calculate_file_sha256(path)
    X = build_integration_features(observations)
    y = build_integration_targets(observations)
    split = split_integration_dataset(
        X, y, test_size=config.test_size, random_state=config.random_state
    )
    pipeline = train_integration_model(
        build_integration_pipeline(random_state=config.random_state, max_iter=config.max_iter),
        split.X_train,
        split.y_train,
    )
    metrics = evaluate_trained_model(
        pipeline, split.X_test, split.y_test, threshold=config.classification_threshold
    )
    instant = (now or (lambda: datetime.now(timezone.utc)))()
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("training timestamp must be timezone-aware")
    metadata = TrainingMetadata(
        model_name=MODEL_NAME,
        model_version=f"{MODEL_NAME}-{dataset_sha256[:8]}",
        trained_at=instant.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        dataset_path=str(path),
        dataset_sha256=dataset_sha256,
        dataset_rows=len(observations),
        training_rows=len(split.y_train),
        test_rows=len(split.y_test),
        numerical_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        target=TARGET,
        test_size=config.test_size,
        random_state=config.random_state,
        classification_threshold=config.classification_threshold,
        max_iter=config.max_iter,
        model_type=type(pipeline.named_steps["classifier"]).__name__,
        python_version=platform.python_version(),
        scikit_learn_version=sklearn.__version__,
        metrics=asdict(metrics),
    )
    return TrainingResult(
        pipeline, metrics, metadata, len(split.y_train), len(split.y_test), dataset_sha256
    )


def save_model_artifact(pipeline: Pipeline, output_dir: str | Path) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model.joblib"
    joblib.dump(pipeline, path)
    return path


def save_training_metadata(metadata: TrainingMetadata, output_dir: str | Path) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "metadata.json"
    path.write_text(json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_trusted_model_artifact(path: str | Path) -> Pipeline:
    """Load only a local artifact whose provenance Harbor already controls."""
    model = joblib.load(Path(path))
    if not isinstance(model, Pipeline):
        raise TypeError("trusted artifact does not contain a scikit-learn Pipeline")
    return model
