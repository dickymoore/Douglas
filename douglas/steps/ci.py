"""Deterministic CI pipeline simulator."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from douglas.domain.metrics import PassFailCounts, VelocityInputs
from douglas.steps import StepResult


from douglas.steps.utils import _derive_seed
@dataclass
class CIPipeline:
    name: str
    status: str
    duration: float

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "duration": round(self.duration, 3),
        }


class CIStep:
    """Simulate CI status checks."""

    def __init__(
        self,
        project_root: Path | str,
        *,
        name: str = "ci",
        pipelines: Iterable[str] | None = None,
        seed: int = 0,
        failure_rate: float = 0.1,
    ) -> None:
        self.project_root = Path(project_root)
        self.name = name
        self.pipelines = tuple(pipelines or ("build", "lint", "deploy"))
        if not self.pipelines:
            raise ValueError("pipelines must not be empty")
        if not 0 <= failure_rate <= 1:
            raise ValueError("failure_rate must be between 0 and 1")
        self.failure_rate = failure_rate
        self.seed = _derive_seed(int(seed), self.name)
        self._state_dir = self.project_root / ".douglas" / "state"
        self._ci_dir = self.project_root / "ai-inbox" / "ci"

    def _slug(self) -> str:
        return f"{self.seed:016x}"[:8]

    def _report_path(self) -> Path:
        return self._state_dir / f"ci_{self._slug()}.json"

    def _summary_path(self) -> Path:
        return self._ci_dir / f"ci_{self._slug()}.txt"

    def run(self, *, replay: bool = False) -> StepResult:
        if replay:
            return self._load_from_artifacts()
        return self._simulate()

    def _simulate(self) -> StepResult:
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._ci_dir.mkdir(parents=True, exist_ok=True)

        rng = random.Random(self.seed)
        pipeline_results: list[CIPipeline] = []
        failed = 0
        for index, pipeline_name in enumerate(self.pipelines):
            roll = rng.random()
            status = "failed" if roll < self.failure_rate else "passed"
            if status == "failed":
                failed += 1
            duration = 2.5 + rng.random() * 3.0 + index * 0.2
            pipeline_results.append(
                CIPipeline(name=pipeline_name, status=status, duration=duration)
            )

        counts = PassFailCounts(
            passed=len(self.pipelines) - failed, failed=failed, skipped=0
        )
        velocity = VelocityInputs(
            ci_runs=len(self.pipelines),
            ci_failures=failed,
        )
        state_deltas = {
            "ci": {"runs": float(len(self.pipelines)), "failed": float(failed)},
            "velocity": velocity.to_dict(),
        }
        status = "passed" if failed == 0 else "failed"

        report_payload = {
            "name": self.name,
            "seed": self.seed,
            "status": status,
            "summary": counts.to_dict(),
            "pipelines": [item.to_dict() for item in pipeline_results],
            "state_deltas": state_deltas,
        }
        summary_lines = [
            f"CI step: {self.name}",
            f"Seed: {self.seed}",
            f"Status: {status}",
            f"Checks: {len(self.pipelines)}",
            f"Failures: {failed}",
        ]
        for item in pipeline_results:
            summary_lines.append(
                f"- {item.name}: {item.status} ({item.duration:.2f}s)"
            )

        report_path = self._report_path()
        summary_path = self._summary_path()
        report_path.write_text(json.dumps(report_payload, indent=2) + "\n", encoding="utf-8")
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

        return StepResult(
            name=self.name,
            status=status,
            metrics={
                "ci": counts,
                "pipelines": [item.to_dict() for item in pipeline_results],
                "velocity": velocity,
            },
            artifacts=[report_path, summary_path],
            state_deltas=state_deltas,
        )

    def _load_from_artifacts(self) -> StepResult:
        payload = json.loads(self._report_path().read_text(encoding="utf-8"))
        counts = PassFailCounts.from_mapping(payload.get("summary", {}))
        state_deltas = payload.get("state_deltas") or {
            "ci": {"runs": float(counts.total), "failed": float(counts.failed)}
        }
        velocity = VelocityInputs.from_mapping(state_deltas.get("velocity", {}))
        status = payload.get("status", "passed" if counts.failed == 0 else "failed")
        return StepResult(
            name=self.name,
            status=status,
            metrics={
                "ci": counts,
                "pipelines": list(payload.get("pipelines", [])),
                "velocity": velocity,
            },
            artifacts=[self._report_path(), self._summary_path()],
            state_deltas=state_deltas,
        )


__all__ = ["CIPipeline", "CIStep"]
