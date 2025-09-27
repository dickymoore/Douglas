"""Structured orchestration result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional


_LEGACY_BLOCK_PATTERN = re.compile(
    r"```(?P<path>[^\n`]+)\n(?P<body>.*?)```",
    flags=re.DOTALL | re.MULTILINE,
)


def _now_iso() -> str:
    """Return an ISO-8601 timestamp in UTC."""

    return datetime.now(timezone.utc).isoformat()


def _normalize_prompt(prompt: str) -> str:
    """Collapse whitespace for stable prompt hashing."""

    return " ".join(prompt.strip().split())


@dataclass
class StepArtifact:
    """Represents a file created or modified during a step."""

    path: str
    content: str
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the artifact to a JSON compatible dictionary."""

        return {
            "path": self.path,
            "content": self.content,
            "description": self.description,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StepArtifact":
        """Create an artifact from a mapping."""

        return cls(
            path=str(payload.get("path", "")),
            content=str(payload.get("content", "")),
            description=(
                str(payload["description"])
                if payload.get("description") is not None
                else None
            ),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class StepEvent:
    """Log-like entry emitted while a step executes."""

    message: str
    level: str = "info"
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the event to a dictionary."""

        return {
            "message": self.message,
            "level": self.level,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StepEvent":
        """Hydrate an event from its JSON form."""

        timestamp = payload.get("timestamp")
        return cls(
            message=str(payload.get("message", "")),
            level=str(payload.get("level", "info")),
            timestamp=str(timestamp) if timestamp is not None else None,
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class StepResult:
    """Structured output produced by orchestrated providers."""

    step_name: str
    role: str
    agent: str
    seed: Optional[int] = None
    prompt_hash: Optional[str] = None
    timestamps: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[StepArtifact] = field(default_factory=list)
    state_deltas: Dict[str, Any] = field(default_factory=dict)
    events: List[StepEvent] = field(default_factory=list)
    questions: List[Any] = field(default_factory=list)
    answers: List[Any] = field(default_factory=list)
    ci_reports: List[Any] = field(default_factory=list)
    test_reports: List[Any] = field(default_factory=list)
    retro_notes: List[Any] = field(default_factory=list)
    commits: List[Any] = field(default_factory=list)
    errors: List[Any] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result into a JSON friendly mapping."""

        return {
            "step_name": self.step_name,
            "role": self.role,
            "agent": self.agent,
            "seed": self.seed,
            "prompt_hash": self.prompt_hash,
            "timestamps": dict(self.timestamps),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "state_deltas": dict(self.state_deltas),
            "events": [event.to_dict() for event in self.events],
            "questions": list(self.questions),
            "answers": list(self.answers),
            "ci_reports": list(self.ci_reports),
            "test_reports": list(self.test_reports),
            "retro_notes": list(self.retro_notes),
            "commits": list(self.commits),
            "errors": list(self.errors),
        }

    def to_json(self) -> str:
        """Serialise the result into a JSON string."""

        return json.dumps(self.to_dict(), sort_keys=True)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------
    def with_defaults(
        self,
        *,
        step_name: Optional[str] = None,
        agent: Optional[str] = None,
        role: Optional[str] = None,
        seed: Optional[int] = None,
        prompt_hash: Optional[str] = None,
        timestamps: Optional[Mapping[str, Any]] = None,
    ) -> "StepResult":
        """Populate missing metadata without overwriting explicit values."""

        if step_name and not self.step_name:
            self.step_name = step_name
        if agent and not self.agent:
            self.agent = agent
        if role and not self.role:
            self.role = role
        if seed is not None and self.seed is None:
            self.seed = seed
        if prompt_hash and not self.prompt_hash:
            self.prompt_hash = prompt_hash
        if timestamps:
            merged: MutableMapping[str, Any] = dict(timestamps)
            merged.update(self.timestamps)
            self.timestamps = dict(merged)
        if not self.timestamps:
            self.timestamps = {"completed_at": _now_iso()}
        return self

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StepResult":
        """Instantiate a ``StepResult`` from a mapping."""

        artifacts_payload = payload.get("artifacts") or []
        events_payload = payload.get("events") or []
        return cls(
            step_name=str(payload.get("step_name", "")),
            role=str(payload.get("role", "")),
            agent=str(payload.get("agent", "")),
            seed=(
                int(payload["seed"])
                if isinstance(payload.get("seed"), (int, float))
                else payload.get("seed")
            ),
            prompt_hash=(
                str(payload["prompt_hash"])
                if payload.get("prompt_hash") is not None
                else None
            ),
            timestamps=dict(payload.get("timestamps", {})),
            artifacts=[
                StepArtifact.from_dict(artifact)
                for artifact in artifacts_payload
                if isinstance(artifact, Mapping)
            ],
            state_deltas=dict(payload.get("state_deltas", {})),
            events=[
                StepEvent.from_dict(event)
                for event in events_payload
                if isinstance(event, Mapping)
            ],
            questions=list(payload.get("questions", [])),
            answers=list(payload.get("answers", [])),
            ci_reports=list(payload.get("ci_reports", [])),
            test_reports=list(payload.get("test_reports", [])),
            retro_notes=list(payload.get("retro_notes", [])),
            commits=list(payload.get("commits", [])),
            errors=list(payload.get("errors", [])),
        )

    @classmethod
    def from_json(cls, payload: str) -> "StepResult":
        """Hydrate a ``StepResult`` from a JSON string."""

        data = json.loads(payload)
        if not isinstance(data, Mapping):
            raise TypeError("StepResult JSON must decode to a mapping.")
        return cls.from_dict(data)

    @classmethod
    def ensure(
        cls,
        value: Any,
        *,
        step_name: Optional[str] = None,
        agent: Optional[str] = None,
        role: Optional[str] = None,
        seed: Optional[int] = None,
        prompt_hash: Optional[str] = None,
        prompt: Optional[str] = None,
        timestamps: Optional[Mapping[str, Any]] = None,
    ) -> "StepResult":
        """Coerce provider output into a ``StepResult`` instance."""

        if isinstance(value, StepResult):
            return value.with_defaults(
                step_name=step_name,
                agent=agent,
                role=role,
                seed=seed,
                prompt_hash=prompt_hash,
                timestamps=timestamps,
            )

        if isinstance(value, Mapping):
            return cls.from_dict(value).with_defaults(
                step_name=step_name,
                agent=agent,
                role=role,
                seed=seed,
                prompt_hash=prompt_hash,
                timestamps=timestamps,
            )

        inferred_prompt_hash = prompt_hash
        if inferred_prompt_hash is None and prompt is not None:
            inferred_prompt_hash = cls.compute_prompt_hash(prompt)

        artifacts = list(cls._parse_legacy_artifacts(value))
        events: List[StepEvent] = []
        if isinstance(value, str) and value.strip():
            events.append(
                StepEvent(
                    message=value,
                    level="info",
                    timestamp=_now_iso(),
                    metadata={"source": "legacy_text"},
                )
            )

        result = cls(
            step_name=step_name or "",
            role=role or "",
            agent=agent or "",
            seed=seed,
            prompt_hash=inferred_prompt_hash,
            timestamps=dict(timestamps or {}),
            artifacts=artifacts,
            events=events,
        )
        return result.with_defaults(
            step_name=step_name,
            agent=agent,
            role=role,
            seed=seed,
            prompt_hash=inferred_prompt_hash,
            timestamps=timestamps,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def compute_prompt_hash(prompt: str) -> str:
        """Return a deterministic hash for the given prompt."""

        normalized = _normalize_prompt(prompt)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def default_timestamps() -> Dict[str, str]:
        """Generate a default start/end timestamp pair."""

        started = _now_iso()
        completed = _now_iso()
        return {"started_at": started, "completed_at": completed}

    @classmethod
    def _parse_legacy_artifacts(cls, value: Any) -> Iterable[StepArtifact]:
        """Extract artifacts from legacy triple-backtick blocks."""

        if not isinstance(value, str):
            return []

        artifacts: List[StepArtifact] = []
        for match in _LEGACY_BLOCK_PATTERN.finditer(value):
            path = match.group("path").strip()
            body = match.group("body")
            artifacts.append(
                StepArtifact(
                    path=path,
                    content=body.strip("\n") + "\n" if body else "",
                )
            )
        return artifacts


__all__ = [
    "StepArtifact",
    "StepEvent",
    "StepResult",
]

