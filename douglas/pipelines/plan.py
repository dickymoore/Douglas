"""Planning pipeline that seeds an initial backlog via the configured LLM."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
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
    charter_paths: Dict[str, Path] = field(default_factory=dict)

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
    backlog_exists = backlog_path.exists()
    allow_overwrite = bool(context.planning_config.get("allow_overwrite"))
    existing_backlog: Dict[str, Any] = {}
    if backlog_exists:
        existing_backlog = _load_backlog(backlog_path)

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

    if existing_backlog:
        if allow_overwrite:
            status = "overwritten"
        else:
            backlog_data = _merge_backlog(existing_backlog, backlog_data)
            status = "merged"
    else:
        status = "created"

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

    charter_paths: Dict[str, Path] = {}
    charters_cfg = context.planning_config.get("charters", {})
    if llm is not None and charters_cfg.get("enabled", True):
        charter_paths = _generate_charters(
            project_root=context.project_root,
            llm=llm,
            backlog_data=backlog_data,
            config=charters_cfg,
        )

    return PlanResult(
        True,
        backlog_path,
        backlog_data,
        raw_response,
        reason=status,
        charter_paths=charter_paths,
    )


def _load_backlog(path: Path) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        logger.warning("Unable to read existing backlog %s: %s", path, exc)
        return {}

    if not text.strip():
        return {}

    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        logger.warning("Existing backlog at %s is not valid YAML: %s", path, exc)
        return {}

    return data if isinstance(data, dict) else {}


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


def _merge_backlog(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(existing)

    for key in ("epics", "features", "stories", "tasks", "roadmap", "dependencies"):
        merged = _merge_list(existing.get(key), incoming.get(key))
        if merged is not None:
            result[key] = merged

    # Allow other scalar keys to overwrite
    for key, value in incoming.items():
        if key not in result:
            result[key] = value

    return result


def _merge_list(existing: Any, incoming: Any) -> Optional[List[Any]]:
    if not incoming:
        return existing if isinstance(existing, list) else None

    if not isinstance(incoming, list):
        return existing if isinstance(existing, list) else None

    existing_items = list(existing) if isinstance(existing, list) else []
    index: Dict[Any, Any] = {}
    for item in existing_items:
        if isinstance(item, dict) and "id" in item:
            index[item["id"]] = item
    for item in incoming:
        if isinstance(item, dict) and "id" in item:
            if item["id"] not in index:
                existing_items.append(item)
        else:
            if item not in existing_items:
                existing_items.append(item)
    return existing_items


def _generate_charters(
    *,
    project_root: Path,
    llm: Any,
    backlog_data: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Path]:
    directory = config.get("directory", "ai-inbox/charters")
    allow_overwrite = bool(config.get("allow_overwrite", False))
    charters_dir = project_root / directory

    backlog_yaml = yaml.safe_dump(backlog_data, sort_keys=False)
    prompt = _build_charter_prompt(backlog_yaml)
    raw_charters = llm.generate_code(prompt)
    charter_docs = _parse_charter_documents(raw_charters)

    if not charter_docs:
        charter_docs = _default_charter_documents(backlog_yaml)

    charters_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    for key, text in charter_docs.items():
        filename = _charter_filename(key)
        if not filename:
            continue
        path = charters_dir / filename
        if path.exists() and not allow_overwrite:
            written[key] = path
            continue
        path.write_text(text.strip() + "\n", encoding="utf-8")
        logger.info("Wrote charter document %s", path)
        written[key] = path

    return written


def _build_charter_prompt(backlog_yaml: str) -> str:
    return textwrap.dedent(
        f"""
        You are an agile coach facilitating Sprint Zero for an autonomous software team.
        Based on the backlog below, produce YAML with the following keys:

        agents_md: |  # Markdown describing each agile agent and their responsibilities
        agent_charter_md: |  # Markdown capturing team mission, success criteria, and cadences
        coding_guidelines_md: |  # Markdown summarising code standards, testing expectations, CI rules
        working_agreements_md: |  # Markdown listing agreements for communication, reviews, and delivery flow

        Backlog context:
        ```yaml
        {backlog_yaml}
        ```
        """
    ).strip()


def _parse_charter_documents(raw_text: str) -> Dict[str, str]:
    if not raw_text.strip():
        return {}
    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        logger.warning("Unable to parse charter YAML: %s", exc)
        return {}
    if not isinstance(data, dict):
        return {}
    docs: Dict[str, str] = {}
    for key in ("agents_md", "agent_charter_md", "coding_guidelines_md", "working_agreements_md"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            docs[key] = value
    return docs


def _default_charter_documents(backlog_yaml: str) -> Dict[str, str]:
    skeleton = textwrap.dedent(
        """# Team Charter

        This charter will be elaborated during Sprint Zero.
        """
    ).strip()
    return {
        "agents_md": "# Agile Agents\n\n- Product Owner\n- Developer\n- Tester\n- DevOps\n",
        "agent_charter_md": skeleton,
        "coding_guidelines_md": "# Coding Guidelines\n\n- Follow PEP 8\n- Maintain automated tests\n",
        "working_agreements_md": "# Working Agreements\n\n- Daily async standup\n- Pull request reviews within 24 hours\n",
    }


def _charter_filename(key: str) -> Optional[str]:
    mapping = {
        "agents_md": "AGENTS.md",
        "agent_charter_md": "AGENT_CHARTER.md",
        "coding_guidelines_md": "CODING_GUIDELINES.md",
        "working_agreements_md": "WORKING_AGREEMENTS.md",
    }
    return mapping.get(key)
