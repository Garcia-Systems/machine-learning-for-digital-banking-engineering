"""Display Chapter 2's definitions and intentional leakage example."""

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from harbor_ml import HARBOR_PROBLEMS, MLProblem, ProblemType  # noqa: E402


def display_problem(problem: MLProblem) -> None:
    """Print and explicitly validate one problem definition."""
    print(f"Problem: {problem.name}")
    print(f"Type: {problem.problem_type.value.replace('_', ' ')}\n")
    print("Question:")
    print(f"{problem.engineering_question}\n")
    print("Features:")
    for feature in problem.features:
        print(f"- {feature}")
    print("\nTarget:")
    print(problem.target if problem.target is not None else "None (unsupervised)")
    problem.validate()
    print("\nValidation:\nOK\n")


def main() -> None:
    """Present valid definitions, then catch expected target leakage."""
    print("Harbor Federal Credit Union — ML Problem Framing\n")
    for problem in HARBOR_PROBLEMS:
        display_problem(problem)

    print("Leakage demonstration (intentionally invalid teaching material)\n")
    print("Features:\n- vendor_latency_ms\n- request_failed")
    print("\nTarget:\nrequest_failed\n")
    print("Validation:")
    try:
        MLProblem(
            name="Invalid request failure prediction",
            engineering_question="Will this vendor-backed request fail?",
            problem_type=ProblemType.BINARY_CLASSIFICATION,
            features=("vendor_latency_ms", "request_failed"),
            target="request_failed",
        )
    except ValueError as error:
        print("FAILED\n\nReason:")
        print(error)


if __name__ == "__main__":
    main()
