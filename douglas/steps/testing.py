"""Deterministic test outcome simulator used for offline runs."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from douglas.domain.metrics import Coverage, PassFailCounts, VelocityInputs
from douglas.steps import StepResult


def _derive_seed(base_seed: int, name: str) -> int:
    material = f"{base_seed}:{name}".encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    return int(digest[:16], 16)


def _normalize_range(values: Sequence[float]) -> tuple[float, float]:
    """Validate and normalize coverage percentage ranges.

    The simulator expects coverage ranges expressed as percentages between
    ``0`` and ``100``. Callers can provide either ``int`` or ``float`` values
    and the order is normalized.
    """

    if len(values) != 2:
        raise ValueError("coverage_range must contain exactly two numbers")
    low, high = sorted(float(v) for v in values)
    if not 0 <= low <= 100 or not 0 <= high <= 100:
        raise ValueError("coverage_range values must be between 0 and 100")
    return low, high


@dataclass
class OfflineTestingConfig:
    """Configuration knobs for the test simulator.

    ``coverage_range`` is expressed as a pair of percentages between ``0`` and
    ``100`` (for example ``(65.0, 90.0)``).
    """

    seed: int
    suite: str = "unit"
    test_count: int = 12
    failure_rate: float = 0.2
    coverage_range: tuple[float, float] = (65.0, 90.0)

    def derived_seed(self) -> int:
        return _derive_seed(self.seed, self.suite)


class OfflineTestingStep:
    """Simulate a unit test run for the deterministic offline workflow."""

    def __init__(
        self,
        project_root: Path | str,
        config: OfflineTestingConfig | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.config = config or OfflineTestingConfig(seed=0)
        if self.config.test_count <= 0:
            raise ValueError("test_count must be positive")
        if not 0 <= self.config.failure_rate <= 1:
            raise ValueError("failure_rate must be between 0 and 1")
        self.coverage_range = _normalize_range(self.config.coverage_range)
        self._state_dir = self.project_root / ".douglas" / "state"
        self._ci_dir = self.project_root / "ai-inbox" / "ci"

    # Paths -----------------------------------------------------------------
    def _slug(self) -> str:
        return f"{self.config.derived_seed():016x}"[:8]

    def _report_path(self) -> Path:
        return self._state_dir / f"test_report_{self._slug()}.json"

    def _coverage_path(self) -> Path:
        return self._state_dir / f"coverage_{self._slug()}.json"

    def _summary_path(self) -> Path:
        return self._ci_dir / f"test_report_{self._slug()}.txt"

    # Public API -------------------------------------------------------------
    def run(self, *, replay: bool = False) -> StepResult:
        if replay:
            return self._load_from_artifacts()
        return self._simulate()

    # Internal helpers -------------------------------------------------------
    def _simulate(self) -> StepResult:
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._ci_dir.mkdir(parents=True, exist_ok=True)

        rng = random.Random(self.config.derived_seed())
        tests = []
        passed = failed = skipped = 0
        total_duration = 0.0
        for index in range(self.config.test_count):
            name = f"test_case_{index:03d}"
            roll = rng.random()
            if roll < self.config.failure_rate:
                status = "failed"
                failed += 1
            else:
                status = "passed"
                passed += 1
            duration = round(0.25 + rng.random() * 1.5, 3)
            total_duration += duration
            tests.append({"name": name, "status": status, "duration": duration})

        counts = PassFailCounts(passed=passed, failed=failed, skipped=skipped)
        coverage = self._generate_coverage(rng)
        status = "passed" if counts.failed == 0 else "failed"
        velocity = VelocityInputs(
            test_runs=1,
            tests_total=counts.total,
            tests_failed=counts.failed,
            coverage_points=coverage.percent,
            ci_runs=0,
            ci_failures=0,
        )
        state_deltas = {
            "tests": counts.to_dict(),
            "coverage": {"runs": 1.0, "percent_total": coverage.percent},
            "velocity": velocity.to_dict(),
        }

        report_payload = {
            "suite": self.config.suite,
            "seed": self.config.seed,
            "status": status,
            "summary": counts.to_dict(),
            "tests": tests,
            "duration_seconds": round(total_duration, 3),
            "state_deltas": state_deltas,
        }
        coverage_payload = {
            "suite": self.config.suite,
            "seed": self.config.seed,
            "coverage": coverage.to_dict(),
        }
        summary_lines = [
            f"Test suite: {self.config.suite}",
            f"Seed: {self.config.seed}",
            f"Status: {status}",
            f"Total: {counts.total}",
            f"Passed: {counts.passed}",
            f"Failed: {counts.failed}",
            f"Coverage: {coverage.percent:.2f}%",
        ]

        report_path = self._report_path()
        coverage_path = self._coverage_path()
        summary_path = self._summary_path()

        report_path.write_text(json.dumps(report_payload, indent=2) + "\n", encoding="utf-8")
        coverage_path.write_text(
            json.dumps(coverage_payload, indent=2) + "\n", encoding="utf-8"
        )
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

        return StepResult(
            name="test",
            status=status,
            metrics={
                "tests": counts,
                "coverage": coverage,
                "velocity": velocity,
                "cases": tests,
            },
            artifacts=[report_path, coverage_path, summary_path],
            state_deltas=state_deltas,
        )

    def _generate_coverage(self, rng: random.Random) -> Coverage:
        low, high = self.coverage_range
        percent = rng.uniform(low, high)
        percent = round(percent, 2)
        total_lines = 400 + rng.randrange(200)
        covered = min(total_lines, int(round(total_lines * percent / 100.0)))
        return Coverage(covered=covered, total=total_lines)

    def _load_from_artifacts(self) -> StepResult:
        report_payload = json.loads(self._report_path().read_text(encoding="utf-8"))
        coverage_payload = json.loads(
            self._coverage_path().read_text(encoding="utf-8")
        )
        counts = PassFailCounts.from_mapping(report_payload.get("summary", {}))
        coverage = Coverage.from_mapping(coverage_payload.get("coverage", {}))
        status = report_payload.get("status", "passed" if counts.failed == 0 else "failed")
        state_deltas = report_payload.get("state_deltas") or {
            "tests": counts.to_dict(),
            "coverage": {"runs": 1.0, "percent_total": coverage.percent},
        }
        velocity = VelocityInputs.from_mapping(state_deltas.get("velocity", {}))
        metrics = {
            "tests": counts,
            "coverage": coverage,
            "velocity": velocity,
            "cases": list(report_payload.get("tests", [])),
        }
        return StepResult(
            name="test",
            status=status,
            metrics=metrics,
            artifacts=[self._report_path(), self._coverage_path(), self._summary_path()],
            state_deltas=state_deltas,
        )


__all__ = ["OfflineTestingConfig", "OfflineTestingStep"]
