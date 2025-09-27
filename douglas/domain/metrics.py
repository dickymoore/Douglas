"""Structured metrics used by deterministic offline steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Coverage:
    """Represents a coverage snapshot."""

    covered: int
    total: int

    @property
    def percent(self) -> float:
        if self.total <= 0:
            return 100.0 if self.covered >= 0 else 0.0
        value = (self.covered / self.total) * 100
        return round(value, 2)

    def to_dict(self) -> dict[str, float]:
        return {
            "covered": float(self.covered),
            "total": float(self.total),
            "percent": self.percent,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, float]) -> "Coverage":
        return cls(
            covered=int(round(float(data.get("covered", 0)))),
            total=int(round(float(data.get("total", 0)))),
        )


@dataclass(frozen=True)
class PassFailCounts:
    """Summarises pass/fail/skip counts for a suite of checks."""

    passed: int
    failed: int
    skipped: int = 0

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.skipped

    def to_dict(self) -> dict[str, int]:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "total": self.total,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, float]) -> "PassFailCounts":
        return cls(
            passed=int(round(float(data.get("passed", 0)))),
            failed=int(round(float(data.get("failed", 0)))),
            skipped=int(round(float(data.get("skipped", 0)))),
        )


@dataclass(frozen=True)
class VelocityInputs:
    """Aggregated numeric inputs used by velocity style dashboards."""

    test_runs: int = 0
    tests_total: int = 0
    tests_failed: int = 0
    coverage_points: float = 0.0
    ci_runs: int = 0
    ci_failures: int = 0

    def to_dict(self) -> dict[str, float]:
        return {
            "test_runs": float(self.test_runs),
            "tests_total": float(self.tests_total),
            "tests_failed": float(self.tests_failed),
            "coverage_points": float(self.coverage_points),
            "ci_runs": float(self.ci_runs),
            "ci_failures": float(self.ci_failures),
        }

    def __add__(self, other: "VelocityInputs") -> "VelocityInputs":
        return VelocityInputs(
            test_runs=self.test_runs + other.test_runs,
            tests_total=self.tests_total + other.tests_total,
            tests_failed=self.tests_failed + other.tests_failed,
            coverage_points=self.coverage_points + other.coverage_points,
            ci_runs=self.ci_runs + other.ci_runs,
            ci_failures=self.ci_failures + other.ci_failures,
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, float]) -> "VelocityInputs":
        return cls(
            test_runs=int(round(float(data.get("test_runs", 0)))),
            tests_total=int(round(float(data.get("tests_total", 0)))),
            tests_failed=int(round(float(data.get("tests_failed", 0)))),
            coverage_points=float(data.get("coverage_points", 0.0)),
            ci_runs=int(round(float(data.get("ci_runs", 0)))),
            ci_failures=int(round(float(data.get("ci_failures", 0)))),
        )


__all__ = ["Coverage", "PassFailCounts", "VelocityInputs"]
