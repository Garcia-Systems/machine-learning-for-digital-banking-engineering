from pathlib import Path
import re

import pytest

from harbor_ml.data_security import SensitiveFieldError, validate_prediction_payload_fields
from harbor_ml.dashboard.service import build_capstone_dashboard
from harbor_ml.operations import run_operating_laboratory

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def report():
    return run_operating_laboratory(ROOT)


def test_master_lab_assembles_independent_models_and_responsible_ml(report):
    assert report.dataset_rows > 0 and report.timeline_rows > 0
    assert report.health_status == "ok"
    assert len(report.inventory) == 3
    assert len({item.name for item in report.inventory}) == 3
    assert len({item.version for item in report.inventory}) == 3
    assert all(item.dataset_hash != "unknown" and item.model_type for item in report.inventory)
    assert report.explanation_delta == pytest.approx(0, abs=1e-12)
    assert report.slices and {item.feature for item in report.slices} == {"vendor", "endpoint"}
    assert report.monitoring_periods == ("A", "B", "C", "D")


def test_failures_are_contained_and_rollback_restores_version(report):
    assert report.partial_outage and report.complete_outage
    assert report.deterministic_operation_preserved
    assert report.stale_recognized
    integration = next(item for item in report.inventory if item.name == "harbor-integration-failure")
    assert report.rollback_version == integration.version


def test_review_keeps_prediction_separate_from_outcome(report):
    assert report.review_prediction == .91
    assert report.review_outcome == "resolved_no_issue"


def test_sensitive_field_is_rejected():
    with pytest.raises(SensitiveFieldError):
        validate_prediction_payload_fields({"access_token": "fictional"})


def test_contents_has_exactly_34_ordered_entries_and_chapter_33_is_final():
    contents = (ROOT / "CONTENTS.md").read_text(encoding="utf-8")
    numbers = [int(value) for value in re.findall(r"^(\d+)\. ", contents, re.MULTILINE)]
    assert numbers == list(range(34))
    assert "chapter-33-operating-the-intelligent-digital-credit-union.md" in contents
    assert not list((ROOT / "book").rglob("chapter-34-*.md"))


def test_final_documentation_navigation_and_boundaries():
    chapter = ROOT / "book/part-07-capstone/chapter-33-operating-the-intelligent-digital-credit-union.md"
    assert chapter.exists()
    text = chapter.read_text(encoding="utf-8")
    assert "What comes next: Chapter 34" not in text
    assert "Where to go from here" in text
    assert "MODEL PREDICTION remains historically unchanged" in text
    assert "Harbor Federal Credit Union" in text
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "34 numbered chapters (0–33)" in readme
    assert "chapter_33_operating_harbor.py" in readme
    part = (ROOT / "book/part-07-capstone/README.md").read_text(encoding="utf-8")
    assert part.index("Chapter 32") < part.index("Chapter 33")


def test_historical_dashboard_boundary_is_documented_and_regression_covered():
    service, rows = build_capstone_dashboard(ROOT)
    before_trace = next(row for row in rows if row.timestamp.minute == 28)
    at_trace = next(row for row in rows if row.timestamp.minute == 30)
    assert service.build_snapshot(before_trace).confirmed_evidence == ()
    assert service.build_snapshot(at_trace).confirmed_evidence
    text = (ROOT / "book/part-07-capstone/chapter-33-operating-the-intelligent-digital-credit-union.md").read_text()
    assert "Historical playback must never reveal that trace early" in text
