"""Step helpers for Douglas workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, MutableMapping


@dataclass
class StepResult:
    """Structured outcome of running an offline step.

    The object intentionally keeps the surface area minimal so that individual
    step implementations can attach custom metrics while still allowing callers
    to aggregate numeric deltas in a predictable way.
    """

    name: str
    status: str
    metrics: dict[str, object] = field(default_factory=dict)
    artifacts: list[Path] = field(default_factory=list)
    state_deltas: dict[str, Dict[str, float]] = field(default_factory=dict)

    def apply_state(
        self, totals: MutableMapping[str, Dict[str, float]]
    ) -> MutableMapping[str, Dict[str, float]]:
        """Apply the ``state_deltas`` to an aggregate mapping.

        Each delta bucket is merged into ``totals`` by summing numeric values.
        The mapping is returned to make chained usage ergonomic in tests.
        """

        for bucket, delta in self.state_deltas.items():
            bucket_totals = totals.setdefault(bucket, {})
            for key, value in delta.items():
                bucket_totals[key] = bucket_totals.get(key, 0) + float(value)
        return totals


# Import steps after defining StepResult to avoid circular imports
from .delivery import (
    DeliveryContext,
    DeliveryStory,
    StepResult as DeliveryStepResult,
    run_delivery,
)
from .planning import PlanningContext, PlanningStepResult, run_planning
from .sprint_zero import (
    SprintZeroContext,
    StepResult as SprintZeroStepResult,
    run_sprint_zero,
)

# Import CI and Testing modules that depend on the base StepResult
from .ci import CIStep
from .testing import OfflineTestingStep

__all__ = [
    "StepResult",
    "CIStep",
    "OfflineTestingStep",
    "DeliveryContext",
    "DeliveryStory",
    "DeliveryStepResult",
    "run_delivery",
    "PlanningContext",
    "PlanningStepResult",
    "run_planning",
    "SprintZeroContext",
    "SprintZeroStepResult",
    "run_sprint_zero",
]
