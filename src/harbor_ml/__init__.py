"""Educational machine-learning engineering components for the Harbor examples."""

from .thresholds import Observation, Thresholds, find_threshold_violations
from .telemetry import (
    TelemetryObservation,
    TelemetrySummary,
    load_telemetry,
    summarize_telemetry,
)
from .problem_framing import (
    FUTURE_LATENCY,
    HARBOR_PROBLEMS,
    INCIDENT_CLASSIFICATION,
    REQUEST_FAILURE,
    TELEMETRY_ANOMALY,
    MLProblem,
    ProblemType,
    RequestOutcome,
    load_request_outcomes,
)
from .request_failure_model import (
    FEATURE_NAMES,
    RANDOM_STATE,
    DatasetSplit,
    EvaluationResult,
    FailurePrediction,
    build_feature_matrix,
    build_model,
    build_target_vector,
    evaluate_model,
    predict_request_failure,
    split_dataset,
    train_model,
)

__all__ = [
    "Observation",
    "FUTURE_LATENCY",
    "HARBOR_PROBLEMS",
    "INCIDENT_CLASSIFICATION",
    "MLProblem",
    "ProblemType",
    "REQUEST_FAILURE",
    "RequestOutcome",
    "TELEMETRY_ANOMALY",
    "TelemetryObservation",
    "TelemetrySummary",
    "Thresholds",
    "find_threshold_violations",
    "FEATURE_NAMES",
    "RANDOM_STATE",
    "DatasetSplit",
    "EvaluationResult",
    "FailurePrediction",
    "build_feature_matrix",
    "build_model",
    "build_target_vector",
    "evaluate_model",
    "load_telemetry",
    "load_request_outcomes",
    "predict_request_failure",
    "split_dataset",
    "summarize_telemetry",
    "train_model",
]

# Chapter 4 names are explicit to avoid ambiguity with Chapter 3's features.
from .anomaly_detection import (
    ANOMALY_CONTAMINATION,
    ANOMALY_FEATURE_NAMES,
    ANOMALY_RANDOM_STATE,
    AnomalyResult,
    AnomalyScenario,
    build_anomaly_detector,
    build_anomaly_features,
    load_anomaly_scenarios,
    load_normal_telemetry,
    observation_features,
    score_observation,
    train_anomaly_detector,
)

__all__ += [
    "ANOMALY_CONTAMINATION",
    "ANOMALY_FEATURE_NAMES",
    "ANOMALY_RANDOM_STATE",
    "AnomalyResult",
    "AnomalyScenario",
    "build_anomaly_detector",
    "build_anomaly_features",
    "load_anomaly_scenarios",
    "load_normal_telemetry",
    "observation_features",
    "score_observation",
    "train_anomaly_detector",
]
