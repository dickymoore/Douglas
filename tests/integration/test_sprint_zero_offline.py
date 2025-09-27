import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from douglas.providers.mock_provider import DeterministicMockProvider
from douglas.providers.replay_provider import (
    CassetteRecordingProvider,
    CassetteStore,
    ReplayProvider,
    compute_project_fingerprint,
)
from douglas.steps.sprint_zero import SprintZeroContext, run_sprint_zero

from tests.integration.test_offline_sprint_zero import _prepare_project


def _load_project_metadata(project_dir: Path) -> tuple[str, str, int, dict]:
    config_path = project_dir / "douglas.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    project = config.get("project", {}) if isinstance(config, dict) else {}
    ai = config.get("ai", {}) if isinstance(config, dict) else {}
    name = str(project.get("name", project_dir.name))
    description = str(project.get("description", ""))
    seed = int(ai.get("seed", 0))
    return name, description, seed, ai


def _baseline_context(project_dir: Path) -> SprintZeroContext:
    name, description, seed, _ = _load_project_metadata(project_dir)
    provider = DeterministicMockProvider(project_root=project_dir, seed=seed)
    return SprintZeroContext(
        project_root=project_dir,
        project_name=name,
        project_description=description,
        agent_label="ProductOwner",
        seed=seed,
        llm=provider,
    )


def test_sprint_zero_deterministic_backlog(tmp_path):
    project_dir = _prepare_project(tmp_path)
    context = _baseline_context(project_dir)

    first_result = run_sprint_zero(context)

    backlog_json_path = project_dir / ".douglas" / "state" / "backlog.json"
    backlog_md_path = project_dir / "ai-inbox" / "backlog.md"
    ci_path = project_dir / ".github" / "workflows" / "app.yml"

    assert backlog_json_path.exists()
    assert backlog_md_path.exists()
    assert ci_path.exists()

    backlog_json = backlog_json_path.read_text(encoding="utf-8")
    backlog_md = backlog_md_path.read_text(encoding="utf-8")
    ci_yaml = ci_path.read_text(encoding="utf-8")

    assert first_result.epics
    assert first_result.features
    assert first_result.stories

    second_context = _baseline_context(project_dir)
    second_result = run_sprint_zero(second_context)

    assert first_result == second_result
    assert backlog_json_path.read_text(encoding="utf-8") == backlog_json
    assert backlog_md_path.read_text(encoding="utf-8") == backlog_md
    assert ci_path.read_text(encoding="utf-8") == ci_yaml

    payload = json.loads(backlog_json)
    assert len(payload.get("epics", [])) == len(first_result.epics)


def test_sprint_zero_replay_round_trip(tmp_path):
    project_dir = _prepare_project(tmp_path)
    name, description, seed, ai_config = _load_project_metadata(project_dir)
    cassette_dir = project_dir / ".douglas" / "cassettes"
    store = CassetteStore(cassette_dir)
    fingerprint = compute_project_fingerprint(project_dir, ai_config)

    recording_provider = CassetteRecordingProvider(
        DeterministicMockProvider(project_root=project_dir, seed=seed),
        store=store,
        provider_name="mock",
        model_name=None,
        project_fingerprint=fingerprint,
        base_seed=seed,
    )
    record_context = SprintZeroContext(
        project_root=project_dir,
        project_name=name,
        project_description=description,
        agent_label="ProductOwner",
        seed=seed,
        llm=recording_provider,
    )

    recorded_result = run_sprint_zero(record_context)

    replay_provider = ReplayProvider(
        store=store,
        project_root=project_dir,
        project_fingerprint=fingerprint,
        base_seed=seed,
        model_name=None,
        provider_name="mock",
    )
    replay_context = replace(record_context, llm=replay_provider)
    replay_result = run_sprint_zero(replay_context)

    assert replay_result == recorded_result

    backlog_path = project_dir / ".douglas" / "state" / "backlog.json"
    replay_payload = json.loads(backlog_path.read_text(encoding="utf-8"))
    assert replay_payload.get("epics")
