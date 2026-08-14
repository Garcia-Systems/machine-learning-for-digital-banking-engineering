from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from harbor_ml.capstone_dataset import (
    MODEL_FEATURES, PROHIBITED_FIELDS, ApplicationMetric, CapstoneSources,
    VendorMetric, build_capstone_dataset, build_capstone_observation, build_label,
    load_capstone_sources, parse_timestamp, select_latest_before, select_window,
    validate_temporal_integrity, write_capstone_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
AT_1016 = datetime(2026, 8, 12, 10, 16, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def sources():
    return load_capstone_sources(ROOT / "data")


def test_source_fixture_loading_parsing_and_sorting(sources):
    assert len(sources.application) == len(sources.database) == len(sources.vendor) == 20
    assert len(sources.traces) == 8
    assert len(sources.outcomes) == 30
    assert list(sources.application) == sorted(sources.application, key=lambda row: row.timestamp)
    assert parse_timestamp("2026-08-12T10:16:00Z") == AT_1016
    with pytest.raises(ValueError, match="timezone-aware"):
        parse_timestamp("2026-08-12T10:16:00")


def test_latest_before_never_selects_future_and_enforces_freshness(sources):
    selected = select_latest_before(sources.vendor, AT_1016 + timedelta(seconds=30),
                                    source_name="vendor_metrics", maximum_age=timedelta(minutes=2))
    assert selected.timestamp == AT_1016
    with pytest.raises(ValueError, match="too stale"):
        select_latest_before(tuple(row for row in sources.vendor if row.timestamp <= AT_1016),
                             AT_1016 + timedelta(minutes=3),
                             source_name="vendor_metrics", maximum_age=timedelta(minutes=2))


def test_rolling_window_is_inclusive_and_excludes_future(sources):
    window = select_window(sources.vendor, AT_1016)
    assert window[0].timestamp == datetime(2026, 8, 12, 10, 12, tzinfo=timezone.utc)
    assert window[-1].timestamp == AT_1016
    assert all(AT_1016 - timedelta(minutes=5) <= row.timestamp <= AT_1016 for row in window)
    example = build_capstone_observation(sources, AT_1016)
    expected = sum(row.retry_count for row in window)
    assert example.features["retry_count_5m"] == expected
    now = next(row for row in sources.application if row.timestamp == AT_1016)
    anchor = max((row for row in sources.application
                  if row.timestamp <= AT_1016 - timedelta(minutes=5)), key=lambda row: row.timestamp)
    assert example.features["queue_growth_5m"] == now.queue_depth - anchor.queue_depth


def test_stale_source_is_rejected_and_missing_is_not_zero(sources):
    stale = replace(sources, vendor=tuple(row for row in sources.vendor
                                         if row.timestamp <= AT_1016 - timedelta(minutes=5)))
    with pytest.raises(ValueError, match="vendor_metrics too stale"):
        build_capstone_observation(stale, AT_1016)
    # An observed zero remains an ordinary numeric fact; absence causes rejection above.
    zero = VendorMetric(AT_1016, "ClearVerify", 1280, .02, 0)
    observed = build_capstone_observation(replace(sources, vendor=(zero,)), AT_1016)
    assert observed.features["retry_count"] == 0


def test_feature_and_label_builders_are_separate_and_leakage_safe(sources):
    before = build_capstone_observation(sources, AT_1016)
    first, second = sources.outcomes[0], replace(sources.outcomes[0], request_failed=1)
    observation_time = first.created_at
    assert build_label(first, observation_time) != build_label(second, observation_time)
    after_a = build_capstone_observation(replace(sources, outcomes=(first,)), AT_1016)
    after_b = build_capstone_observation(replace(sources, outcomes=(second,)), AT_1016)
    assert before.features == after_a.features == after_b.features
    assert before.incident_type not in before.features
    assert not ({"duration_ms", "status", "current_request_vendor_duration_ms"} & before.features.keys())


def test_conflicting_duplicate_timestamp_fails(tmp_path):
    source = ROOT / "data/harbor_capstone_incident.csv"
    lines = source.read_text(encoding="utf-8").splitlines()
    lines.append(lines[1].replace(",184,", ",999,"))
    (tmp_path / source.name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name in ("harbor_capstone_traces.csv", "harbor_request_outcomes.csv"):
        (tmp_path / name).write_bytes((ROOT / "data" / name).read_bytes())
    # The shared Chapter 26 loader is stricter still: duplicate/out-of-order input is rejected.
    with pytest.raises(ValueError, match="strictly chronological"):
        load_capstone_sources(tmp_path)


def test_dataset_contract_row_count_temporal_integrity_and_privacy(sources):
    examples = build_capstone_dataset(sources)
    assert len(examples) == 17
    assert all(set(row.features) == set(MODEL_FEATURES) for row in examples)
    assert all(selection.source_timestamp <= row.observation_time
               for row in examples for selection in row.source_selections)
    assert not (PROHIBITED_FIELDS & set(MODEL_FEATURES))
    for row in examples:
        validate_temporal_integrity(row)


def test_dataset_fingerprint_is_deterministic(sources, tmp_path):
    examples = build_capstone_dataset(sources)
    first = write_capstone_dataset(examples, tmp_path / "one.csv", tmp_path / "one.json",
                                   sources.source_paths)
    second = write_capstone_dataset(examples, tmp_path / "two.csv", tmp_path / "two.json",
                                    sources.source_paths)
    assert first["output_sha256"] == second["output_sha256"]
    assert (tmp_path / "one.csv").read_bytes() == (tmp_path / "two.csv").read_bytes()
