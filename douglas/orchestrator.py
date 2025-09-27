"""Lightweight StepResult orchestrator."""

from __future__ import annotations

from pathlib import Path

from douglas.domain.step_result import StepArtifact, StepResult


class StepResultOrchestrator:
    """Persist ``StepResult`` artifacts and append to history."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def apply(self, result: StepResult) -> None:
        """Apply a ``StepResult`` by writing artifacts and updating history."""

        for artifact in result.artifacts:
            self._write_artifact(artifact)
        self._write_state_snapshot(result)
        self._append_history(result)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _write_artifact(self, artifact: StepArtifact) -> None:
        path = self._resolve_path(artifact.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(artifact.content, encoding="utf-8")

    def _write_state_snapshot(self, result: StepResult) -> None:
        state_dir = self.project_root / ".douglas" / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = state_dir / "last_step_result.json"
        snapshot_path.write_text(result.to_json(), encoding="utf-8")

    def _append_history(self, result: StepResult) -> None:
        history_dir = self.project_root / "ai-inbox"
        history_dir.mkdir(parents=True, exist_ok=True)
        history_path = history_dir / "history.jsonl"
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(result.to_json())
            handle.write("\n")

    def _resolve_path(self, relative: str) -> Path:
        if not relative:
            raise ValueError("Artifact path must be non-empty.")
        candidate = (self.project_root / relative).resolve()
        root = self.project_root.resolve()
        if not str(candidate).startswith(str(root)):
            raise ValueError(f"Artifact path escapes project root: {relative}")
        return candidate


__all__ = ["StepResultOrchestrator"]

