"""Planning pipeline that seeds an initial backlog via the configured LLM."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from douglas.logging_utils import get_logger


logger = get_logger(__name__)


@dataclass
class PlanContext:
    project_name: str
    project_description: str
    project_root: Path
    backlog_path: Path
    system_prompt_path: Path
    sprint_index: int
    sprint_day: int
    planning_config: Dict[str, Any]
    llm: Any


@dataclass
class PlanResult:
    created_backlog: bool
    backlog_path: Path
    backlog_data: Optional[Dict[str, Any]]
    raw_response: str
    reason: str = ""

    def epic_count(self) -> int:
        if not self.backlog_data:
            return 0
        epics = self.backlog_data.get("epics")
        return len(epics) if isinstance(epics, list) else 0

    def feature_count(self) -> int:
        if not self.backlog_data:
            return 0
        features = self.backlog_data.get("features")
        return len(features) if isinstance(features, list) else 0


def run_plan(context: PlanContext) -> PlanResult:
    """Run Sprint Zero planning if a backlog does not already exist."""

    backlog_path = context.backlog_path
    backlog_exists = backlog_path.exists() and backlog_path.read_text(encoding="utf-8").strip() != ""
    allow_overwrite = bool(context.planning_config.get("allow_overwrite"))

    if backlog_exists and not allow_overwrite:
        logger.info("Backlog already present at %s; skipping planning step.", backlog_path)
        return PlanResult(False, backlog_path, None, "", reason="existing_backlog")

    llm = context.llm
    if llm is None:
        logger.warning("No LLM provider available for planning; skipping backlog generation.")
        return PlanResult(False, backlog_path, None, "", reason="no_llm")

    system_prompt_text = _read_file(context.system_prompt_path)
    project_description = context.project_description or ""

    prompt = _build_prompt(
        project_name=context.project_name,
        project_description=project_description,
        system_prompt=system_prompt_text,
        planning_config=context.planning_config,
    )

    logger.debug("Planning prompt:\n%s", prompt)

    raw_response = llm.generate_code(prompt)
    logger.debug("Planning raw response:\n%s", raw_response)

    backlog_data = _parse_backlog(raw_response)

    if backlog_data is None:
        logger.warning("Planning response could not be parsed; writing raw output as fallback.")
        backlog_data = {"raw": raw_response}

    backlog_path.parent.mkdir(parents=True, exist_ok=True)
    backlog_path.write_text(
        yaml.safe_dump(backlog_data, sort_keys=False),
        encoding="utf-8",
    )

    epics = backlog_data.get("epics")
    features = backlog_data.get("features")
    epic_count = len(epics) if isinstance(epics, list) else 0
    feature_count = len(features) if isinstance(features, list) else 0

    logger.info(
        "Backlog seeded at %s (%d epics, %d features).",
        backlog_path,
        epic_count,
        feature_count,
    )

    return PlanResult(True, backlog_path, backlog_data, raw_response)


def _read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.info("System prompt file %s not found; continuing without it.", path)
        return ""
    except OSError as exc:
        logger.warning("Unable to read system prompt %s: %s", path, exc)
        return ""


def _build_prompt(
    *,
    project_name: str,
    project_description: str,
    system_prompt: str,
    planning_config: Dict[str, Any],
) -> str:
    goal_statement = planning_config.get(
        "goal", "Plan a Sprint Zero backlog before any implementation begins."
    )
    cadence = planning_config.get("cadence", "Scrum")
    output_guidance = planning_config.get(
        "output_guidance",
        textwrap.dedent(
            """
            Produce valid YAML with the following top-level keys:
            epics: list of objects with id, name, objective, success_metrics, features (ids)
            features: list with id, name, epic (id), narrative, business_value, stories (ids)
            stories: list with id, name, feature (id), description, acceptance_criteria (list), tasks (ids)
            tasks: list with id, story (id), description, estimate
            roadmap: high-level timeline for at least the first two sprints (list of objects with sprint, focus, expected_outcomes)
            dependencies: optional list capturing key technical or organisational dependencies.
            """
        ).strip(),
    )

    return textwrap.dedent(
        f"""
        You are the Product Owner and Business Analyst facilitating a Sprint Zero planning workshop for the project "{project_name}".
        Delivery framework: {cadence}.

        Project context:
        {project_description or 'No explicit project description supplied.'}

        Core mission statement:
        {system_prompt or 'No additional mission statement supplied.'}

        {goal_statement}

        {output_guidance}

        Keep IDs short and unique (e.g. EP-1, FE-1, US-1, TK-1). Ensure every feature references an epic, every story references a feature, and tasks reference stories. Use plain YAML with no surrounding commentary.
        """
    ).strip()


def _parse_backlog(raw_text: str) -> Optional[Dict[str, Any]]:
    cleaned = raw_text.strip()
    if not cleaned:
        return None

    try:
        data = yaml.safe_load(cleaned)
    except yaml.YAMLError as exc:
        logger.debug("Failed to parse planning YAML: %s", exc)
        return None

    if not isinstance(data, dict):
        return {"plan": data}

    return data
