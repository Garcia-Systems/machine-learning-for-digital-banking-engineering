"""Run Chapter 23's technical slice evaluation laboratory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harbor_ml.fairness_slices import evaluate_technical_slices
from harbor_ml.integration_failure_model import (build_integration_features,
    load_integration_requests)
from harbor_ml.training import train_integration_failure_model


def _metric(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.3f}"


def main() -> None:
    path = ROOT / "data/harbor_integration_requests.csv"
    observations = load_integration_requests(path)
    trained = train_integration_failure_model(path)
    probabilities = trained.pipeline.predict_proba(
        build_integration_features(observations))[:, 1].tolist()
    slices = evaluate_technical_slices(observations, probabilities)

    print("Harbor Federal Credit Union\nTechnical Slice Evaluation")
    print("context,value,count,base_rate,precision,recall,FPR,FNR,support_status")
    for item in slices:
        print(",".join((item.feature, item.value, str(item.support),
            _metric(item.base_rate), _metric(item.precision), _metric(item.recall),
            _metric(item.false_positive_rate), _metric(item.false_negative_rate),
            "LOW SUPPORT" if item.low_support else "sufficient")))
    print("\nThese technical slices diagnose model behavior; they do not measure member worth,"
          " establish financial eligibility, or authorize action.")


if __name__ == "__main__":
    main()
