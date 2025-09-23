from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["write_demo_pack"]

_TEMPLATE_NAME = "demo.md.j2"
_HISTORY_EVENT = "demo_pack_generated"


class _SafeDict(dict):
    """Dictionary that returns an empty string for missing keys."""

    def __missing__(self, key: str) -> str:  # pragma: no cover - defensive
        return ""


@dataclass
class DemoMetadata:
    """Return payload for generated demo content."""

    output_path: Path
    sprint_folder: str
    head_commit: Optional[str]
    previous_commit: Optional[str]
    commits: List[str]
    generated_at: str
    format: str

    def as_event_payload(self) -> Dict[str, Any]:
        return {
            "sprint": self.sprint_folder,
            "path": str(self.output_path),
            "format": self.format,
            "head_commit": self.head_commit,
            "previous_commit": self.previous_commit,
            "commits": self.commits,
            "generated_at": self.generated_at,
        }


def write_demo_pack(context: Dict[str, Any]) -> DemoMetadata:
    """Render the sprint demo presentation and return metadata for logging."""

    project_root = Path(context.get("project_root", ".")).resolve()
    config: Dict[str, Any] = context.get("config", {}) or {}
    sprint_manager = context.get("sprint_manager")
    history_path = _resolve_history_path(context, project_root)
    loop_outcomes: Dict[str, Optional[bool]] = context.get("loop_outcomes", {}) or {}

    sprint_index = _resolve_sprint_index(context, sprint_manager)
    paths_config = config.get("paths", {}) or {}
    demos_root = project_root / str(paths_config.get("demos_dir", "demos"))
    sprint_prefix = str(paths_config.get("sprint_prefix", "sprint-"))
    sprint_folder = f"{sprint_prefix}{sprint_index}"
    target_dir = demos_root / sprint_folder
    target_dir.mkdir(parents=True, exist_ok=True)

    demo_config = config.get("demo", {}) or {}
    output_format = str(demo_config.get("format", "md") or "md").lower()
    if output_format != "md":
        raise ValueError(f"Unsupported demo format '{output_format}'.")

    template_path = _template_root() / _TEMPLATE_NAME
    template_text = template_path.read_text(encoding="utf-8")

    now = datetime.now()
    head_commit = _get_head_commit(project_root)
    previous_commit = _find_previous_demo_commit(history_path)
    commits = _collect_commits(project_root, previous_commit)
    commit_range = _format_commit_range(head_commit, previous_commit, commits)

    role_summaries = _collect_role_summaries(project_root, sprint_folder)
    implemented_features = _build_features_section(role_summaries, commits)

    commands = _infer_commands(config)
    how_to_run = _format_command_lines(commands)

    test_summary = _build_test_summary(project_root, loop_outcomes)

    limitations_text = _load_optional_text(project_root, sprint_folder, "limitations.md")
    if not limitations_text.strip():
        limitations_text = "_No limitations documented for this sprint._"

    next_steps_text = _load_optional_text(project_root, sprint_folder, "next_steps.md")
    if not next_steps_text.strip():
        next_steps_text = "_Pending planning for upcoming work._"

    include_tokens = {
        str(token).strip().lower() for token in (demo_config.get("include") or [])
    }

    data = {
        "sprint_number": sprint_index,
        "date": now.strftime("%Y-%m-%d"),
        "head_commit": head_commit or "N/A",
        "commit_range": commit_range or "N/A",
        "implemented_features_section": _render_section(
            "What was implemented", implemented_features
        )
        if "implemented_features" in include_tokens or not include_tokens
        else "",
        "how_to_run_section": _render_section("How to run / test", how_to_run)
        if "how_to_run" in include_tokens or not include_tokens
        else "",
        "test_results_section": _render_section("Test summary", test_summary)
        if "test_results" in include_tokens or not include_tokens
        else "",
        "limitations_section": _render_section(
            "Known limitations / TODO", limitations_text
        )
        if "limitations" in include_tokens or not include_tokens
        else "",
        "next_steps_section": _render_section("Next steps", next_steps_text)
        if "next_steps" in include_tokens or not include_tokens
        else "",
    }

    output_path = target_dir / f"demo.{output_format}"
    rendered = template_text.format_map(_SafeDict(data))
    output_path.write_text(rendered, encoding="utf-8")

    return DemoMetadata(
        output_path=output_path,
        sprint_folder=sprint_folder,
        head_commit=head_commit,
        previous_commit=previous_commit,
        commits=commits,
        generated_at=now.isoformat(),
        format=output_format,
    )


def _template_root() -> Path:
    return Path(__file__).resolve().parent.parent / "templates"


def _resolve_history_path(context: Dict[str, Any], project_root: Path) -> Path:
    raw_path = context.get("history_path")
    if raw_path:
        return Path(raw_path)
    return project_root / "ai-inbox" / "history.jsonl"


def _resolve_sprint_index(context: Dict[str, Any], sprint_manager: Any) -> int:
    if sprint_manager is not None:
        index = getattr(sprint_manager, "sprint_index", None)
        if index is not None:
            try:
                return int(index)
            except (TypeError, ValueError):
                pass
    index = context.get("sprint_index", 1)
    try:
        return int(index)
    except (TypeError, ValueError):
        return 1


def _get_head_commit(project_root: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _find_previous_demo_commit(history_path: Path) -> Optional[str]:
    if not history_path or not history_path.exists():
        return None
    try:
        lines = history_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") == _HISTORY_EVENT:
            commit = record.get("head_commit") or record.get("commit")
            if commit:
                return str(commit)
    return None


def _collect_commits(project_root: Path, since_commit: Optional[str]) -> List[str]:
    try:
        if since_commit:
            raw = subprocess.check_output(
                ["git", "log", f"{since_commit}..HEAD", "--pretty=format:%h %s"],
                cwd=project_root,
                text=True,
            )
        else:
            raw = subprocess.check_output(
                ["git", "log", "-5", "--pretty=format:%h %s"],
                cwd=project_root,
                text=True,
            )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

    commits = [line.strip() for line in raw.splitlines() if line.strip()]
    return commits


def _format_commit_range(
    head_commit: Optional[str],
    previous_commit: Optional[str],
    commits: Sequence[str],
) -> str:
    if head_commit and previous_commit and head_commit != previous_commit and commits:
        return f"{previous_commit[:7]}..{head_commit[:7]} ({len(commits)} commits)"
    if head_commit:
        return head_commit[:7]
    return ""


def _collect_role_summaries(project_root: Path, sprint_folder: str) -> List[Tuple[str, str]]:
    base = project_root / "ai-inbox" / "sprints" / sprint_folder / "roles"
    if not base.is_dir():
        return []

    summaries: List[Tuple[str, str]] = []
    for summary_path in sorted(base.glob("*/summary.md")):
        try:
            text = summary_path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not text:
            continue
        role = summary_path.parent.name.replace("_", " ")
        summaries.append((role, text))
    return summaries


def _build_features_section(
    role_summaries: Sequence[Tuple[str, str]],
    commits: Sequence[str],
) -> str:
    lines: List[str] = []
    for role, text in role_summaries:
        condensed = " ".join(text.split())
        display_role = role.strip().title()
        lines.append(f"- **{display_role}:** {condensed}")
    if commits:
        lines.append("- **Commits merged:**")
        lines.extend(f"  - {entry}" for entry in commits)
    return "\n".join(lines).strip()


def _infer_commands(config: Dict[str, Any]) -> List[str]:
    loop_steps = config.get("loop", {}).get("steps", []) or []
    step_names: List[str] = []
    for entry in loop_steps:
        if isinstance(entry, str):
            step_names.append(entry.lower())
        elif isinstance(entry, dict):
            name = entry.get("name") or entry.get("step")
            if name:
                step_names.append(str(name).lower())
    commands: List[str] = []
    if "lint" in step_names:
        commands.extend(["ruff check .", "black --check .", "isort --check ."])
    if "typecheck" in step_names:
        commands.append("mypy .")
    if "test" in step_names:
        tests_dir = config.get("paths", {}).get("tests")
        if tests_dir:
            commands.append(f"pytest {tests_dir}")
        else:
            commands.append("pytest -q")
    app_src = config.get("paths", {}).get("app_src")
    if app_src:
        module_name = Path(str(app_src)).name
        commands.append(f"python -m {module_name}.main")
    return commands


def _format_command_lines(commands: Iterable[str]) -> str:
    seen: List[str] = []
    for command in commands:
        normalized = command.strip()
        if not normalized:
            continue
        if normalized not in seen:
            seen.append(normalized)
    if not seen:
        return ""
    return "\n".join(f"- `{cmd}`" for cmd in seen)


def _build_test_summary(
    project_root: Path, loop_outcomes: Dict[str, Optional[bool]]
) -> str:
    status = loop_outcomes.get("test")
    if status is True:
        prefix = "✅ Tests passed in the last run."
    elif status is False:
        prefix = "❌ Tests failed in the last run."
    else:
        prefix = "ℹ️ Tests were not executed in the last loop iteration."

    log_path = _find_latest_test_log(project_root)
    if log_path:
        prefix += f" Logs: `{log_path}`."
    else:
        prefix += " No saved test logs were found."
    return prefix


def _find_latest_test_log(project_root: Path) -> Optional[str]:
    log_dir = project_root / "ai-inbox" / "tests"
    if not log_dir.is_dir():
        return None
    latest: Optional[Tuple[Path, float]] = None
    for candidate in log_dir.glob("*.log"):
        try:
            stat = candidate.stat()
        except OSError:
            continue
        if latest is None or stat.st_mtime > latest[1]:
            latest = (candidate, stat.st_mtime)
    if latest:
        try:
            return str(latest[0].relative_to(project_root))
        except ValueError:
            return str(latest[0])
    return None


def _load_optional_text(project_root: Path, sprint_folder: str, filename: str) -> str:
    candidate = project_root / "ai-inbox" / "sprints" / sprint_folder / filename
    try:
        return candidate.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _render_section(title: str, body: str) -> str:
    if not body.strip():
        return ""
    return f"## {title}\n{body.strip()}\n\n"


