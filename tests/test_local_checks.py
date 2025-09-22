import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from douglas.core import Douglas


class StaticProvider:
    def generate_code(self, prompt: str) -> str:
        return 'chore: update'


def _init_repo(tmp_path: Path, *, steps: List[dict], exit_conditions: Optional[List[str]] = None) -> Path:
    subprocess.run(['git', 'init'], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(
        ['git', 'config', 'user.email', 'test@example.com'],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ['git', 'config', 'user.name', 'Test User'],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    config_data: dict = {
        'project': {'name': 'LocalChecks', 'language': 'python'},
        'ai': {'provider': 'openai'},
        'loop': {'steps': steps},
        'push_policy': 'per_feature',
    }
    if exit_conditions is not None:
        config_data['loop']['exit_conditions'] = exit_conditions

    config_path = tmp_path / 'douglas.yaml'
    config_path.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding='utf-8')

    readme = tmp_path / 'README.md'
    readme.write_text('Initial content\n', encoding='utf-8')

    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(['git', 'commit', '-m', 'chore: initial'], cwd=tmp_path, check=True, capture_output=True, text=True)

    return config_path


def test_push_step_creates_bug_on_local_check_failure(monkeypatch, tmp_path):
    config_path = _init_repo(tmp_path, steps=[{'name': 'push'}], exit_conditions=[])

    monkeypatch.setattr(Douglas, 'create_llm_provider', lambda self: StaticProvider())

    douglas = Douglas(config_path)
    douglas.sprint_manager.mark_feature_completed('demo-feature')
    douglas.sprint_manager.commits_since_last_push = 1

    local_logs = 'simulated local check failure'

    def fake_local_checks(self):
        return False, local_logs

    def unexpected_push(self):
        raise AssertionError('push should not run when local checks fail')

    monkeypatch.setattr(Douglas, '_run_local_checks', fake_local_checks)
    monkeypatch.setattr(Douglas, '_run_git_push', unexpected_push)
    monkeypatch.setattr(Douglas, '_monitor_ci', lambda self: None)

    douglas.run_loop()

    bug_file = tmp_path / 'ai-inbox' / 'bugs.md'
    assert bug_file.exists()
    bug_contents = bug_file.read_text(encoding='utf-8')
    assert 'simulated local check failure' in bug_contents

    history_path = tmp_path / 'ai-inbox' / 'history.jsonl'
    assert history_path.exists()
    events = [json.loads(line)['event'] for line in history_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert 'bug_reported' in events
    assert 'step_failure' in events


def test_run_local_checks_success_records_history(monkeypatch, tmp_path):
    config_path = _init_repo(tmp_path, steps=[{'name': 'push'}], exit_conditions=[])

    monkeypatch.setattr(Douglas, 'create_llm_provider', lambda self: StaticProvider())
    douglas = Douglas(config_path)

    monkeypatch.setattr(Douglas, '_discover_local_check_commands', lambda self: [['echo', 'ok']])

    original_run = subprocess.run

    def fake_run(command, *args, **kwargs):
        if command == ['echo', 'ok']:
            return subprocess.CompletedProcess(command, 0, stdout='all good\n', stderr='')
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, 'run', fake_run)

    success, logs = douglas._run_local_checks()
    assert success is True
    assert 'all good' in logs

    history_path = tmp_path / 'ai-inbox' / 'history.jsonl'
    assert history_path.exists()
    entries = [json.loads(line) for line in history_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert any(entry['event'] == 'local_checks_pass' for entry in entries)
