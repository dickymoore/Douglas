"""Integration tests for deterministic testing and CI simulators."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from douglas.domain.metrics import Coverage, PassFailCounts, VelocityInputs
from douglas.steps.ci import CIStep
from douglas.steps.testing import OfflineTestingConfig, OfflineTestingStep


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_testing_step_deterministic(tmp_path):
    config = OfflineTestingConfig(
        seed=1234,
        suite="unit",
        test_count=8,
        failure_rate=0.25,
        coverage_range=(55.0, 72.0),
    )
    step = OfflineTestingStep(tmp_path, config=config)
    result_first = step.run()

    # Running with the same configuration should be deterministic.
    step_repeat = OfflineTestingStep(tmp_path, config=config)
    result_second = step_repeat.run()

    assert isinstance(result_first.metrics["tests"], PassFailCounts)
    assert isinstance(result_first.metrics["coverage"], Coverage)
    assert isinstance(result_first.metrics["velocity"], VelocityInputs)
    assert result_first.metrics["tests"].to_dict() == result_second.metrics["tests"].to_dict()
    assert result_first.metrics["coverage"].percent == result_second.metrics["coverage"].percent

    # Coverage stays within the requested percentage range.
    assert 55 <= result_first.metrics["coverage"].percent <= 72

    totals: dict[str, dict[str, float]] = {}
    result_first.apply_state(totals)
    assert totals["tests"]["total"] == config.test_count
    assert totals["coverage"]["runs"] == 1.0
    assert totals["coverage"]["percent_total"] == result_first.metrics["coverage"].percent

    report_path, coverage_path, summary_path = result_first.artifacts
    assert report_path.name.startswith("test_report_")
    assert coverage_path.name.startswith("coverage_")
    assert summary_path.name.startswith("test_report_")

    report_payload = _read_json(report_path)
    coverage_payload = _read_json(coverage_path)
    assert report_payload["summary"]["total"] == config.test_count
    assert coverage_payload["coverage"]["percent"] == result_first.metrics["coverage"].percent

    # Replay from recorded artifacts.
    replay = OfflineTestingStep(tmp_path, config=config).run(replay=True)
    assert replay.metrics["tests"].to_dict() == result_first.metrics["tests"].to_dict()
    assert replay.metrics["coverage"].percent == result_first.metrics["coverage"].percent


def test_ci_step_deterministic(tmp_path):
    step = CIStep(
        tmp_path,
        name="lint",
        pipelines=("lint", "typecheck", "security"),
        seed=2024,
        failure_rate=0.2,
    )
    result = step.run()

    repeat = CIStep(
        tmp_path,
        name="lint",
        pipelines=("lint", "typecheck", "security"),
        seed=2024,
        failure_rate=0.2,
    ).run()

    assert isinstance(result.metrics["ci"], PassFailCounts)
    assert isinstance(result.metrics["velocity"], VelocityInputs)
    assert result.metrics["ci"].to_dict() == repeat.metrics["ci"].to_dict()

    totals: dict[str, dict[str, float]] = {}
    result.apply_state(totals)
    assert totals["ci"]["runs"] == len(step.pipelines)
    assert totals["ci"]["failed"] == result.metrics["ci"].failed

    report_path, summary_path = result.artifacts
    assert report_path.name.startswith("ci_")
    assert summary_path.name.startswith("ci_")

    payload = _read_json(report_path)
    assert payload["summary"]["failed"] == result.metrics["ci"].failed
    assert len(payload["pipelines"]) == len(step.pipelines)

    replay = CIStep(
        tmp_path,
        name="lint",
        pipelines=("lint", "typecheck", "security"),
        seed=2024,
        failure_rate=0.2,
    ).run(replay=True)
    assert replay.metrics["ci"].to_dict() == result.metrics["ci"].to_dict()
