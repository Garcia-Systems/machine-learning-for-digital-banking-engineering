"""Run the Chapter 11 member behavior segmentation laboratory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harbor_ml.member_behavior import group_events_by_session, load_member_events
from harbor_ml.member_segmentation import (
    SEGMENTATION_FEATURES,
    BehavioralSession,
    assign_clusters,
    build_behavioral_sessions,
    build_segmentation_features,
    build_segmentation_pipeline,
    calculate_inertia_by_k,
    inverse_transformed_centers,
    summarize_clusters,
    train_segmentation_model,
)

def description(means: dict[str, float]) -> str:
    """Create a neutral analyst description from a summary, not a model target."""
    if (means["transfer_count"] < 1 and means["statement_view_count"] < 1
            and means["search_count"] + means["help_event_count"] < 1.5):
        return "short/light-activity sessions"
    candidates = {
        "account-view activity": means["account_view_count"],
        "transfer-focused activity": means["transfer_count"] / 3,
        "statement-view activity": means["statement_view_count"] * 2,
        "search/help activity": means["search_count"] + means["help_event_count"],
        "verification activity": means["verification_event_count"],
    }
    return max(candidates, key=candidates.get)


def main() -> None:
    events = load_member_events(ROOT / "data/harbor_member_events.csv")
    rows = build_behavioral_sessions(group_events_by_session(events))
    X = build_segmentation_features(rows)
    print("Harbor Federal Credit Union\nMember Behavior Segmentation Laboratory")
    print(f"\nBehavioral sessions: {len(rows)}\n\nFeatures:")
    for feature in SEGMENTATION_FEATURES:
        print(f"- {feature}")
    print("\nTarget: none (unsupervised learning)\n\nK comparison")
    for k, inertia in calculate_inertia_by_k(X, (2, 3, 4, 5)).items():
        print(f"k={k} inertia={inertia:.2f}")

    print("\nTraining KMeans with k=4... (a teaching choice, not definitive)")
    model = train_segmentation_model(build_segmentation_pipeline(4), X)
    print("Model trained.")
    assignments = assign_clusters(model, rows)
    summaries = summarize_clusters(rows, assignments)
    analyst_descriptions = {}
    for summary in summaries:
        label = description(dict(summary.means))
        analyst_descriptions[summary.cluster_id] = label
        print(f"\nCluster {summary.cluster_id}\nSessions: {summary.session_count}")
        print(f"Analyst description: {label}")
        print("Average behavior:")
        for feature, value in summary.means.items():
            print(f"  {feature}: {value:.2f}")

    print("\nCenters inverse-transformed to behavioral units")
    for cluster_id, center in enumerate(inverse_transformed_centers(model)):
        print(f"\nCluster {cluster_id} center")
        for feature, value in zip(SEGMENTATION_FEATURES, center, strict=True):
            print(f"  {feature}: {value:.2f}")

    new_sessions = [
        BehavioralSession("session-9001", 50, 5, 4, 0, 0, 0, 0, 0),
        BehavioralSession("session-9002", 180, 12, 2, 0, 5, 0, 0, 0),
    ]
    print("\nNew fictional session assignments (computed by the fitted pipeline)")
    for row, cluster_id in zip(new_sessions, assign_clusters(model, new_sessions), strict=True):
        print(f"{row.session_id}: cluster={cluster_id}; analyst description={analyst_descriptions[int(cluster_id)]}")
    print("\nCluster IDs are arbitrary, distances are not probabilities, and descriptions")
    print("are analyst interpretations of synthetic session behavior—not member identities.")


if __name__ == "__main__":
    main()
