import subprocess
from pathlib import Path
from typing import List, Optional

import yaml

from douglas.core import Douglas


class SequencedProvider:
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts: list[str] = []

    def generate_code(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self._responses:
            return self._responses.pop(0)
        return 'chore: automated commit'


def _init_repo(
    tmp_path: Path,
    *,
    push_policy: str,
    sprint_length: Optional[int],
    steps: List[dict],
) -> Path:
    subprocess.run(['git', 'init'], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=tmp_path, check=True, capture_output=True, text=True)

    config_data: dict = {
        'project': {'name': 'PushPolicy', 'language': 'python'},
        'ai': {'provider': 'openai'},
        'loop': {'steps': steps},
        'push_policy': push_policy,
    }
    if sprint_length is not None:
        config_data['sprint'] = {'length_days': sprint_length}

    config_path = tmp_path / 'douglas.yaml'
    config_path.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding='utf-8')

    readme = tmp_path / 'README.md'
    readme.write_text('Initial content\n', encoding='utf-8')

    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(['git', 'commit', '-m', 'chore: initial'], cwd=tmp_path, check=True, capture_output=True, text=True)

    return config_path


def test_per_feature_policy_pushes_and_creates_pr(monkeypatch, tmp_path):
    steps = [
        {'name': 'commit'},
        {'name': 'push'},
        {'name': 'pr'},
    ]
    config_path = _init_repo(tmp_path, push_policy='per_feature', sprint_length=None, steps=steps)

    provider = SequencedProvider(['feat: update docs'])
    monkeypatch.setattr(Douglas, 'create_llm_provider', lambda self: provider)

    push_calls: list[str] = []
    pr_calls: list[str] = []

    def fake_push(self):
        push_calls.append('push')
        return True, 'pushed'

    def fake_pr(self):
        pr_calls.append('pr')
        return True, 'https://example.test/pr'

    monkeypatch.setattr(Douglas, '_run_git_push', fake_push)
    monkeypatch.setattr(Douglas, '_open_pull_request', fake_pr)
    monkeypatch.setattr(Douglas, '_monitor_ci', lambda self: None)

    douglas = Douglas(config_path)
    readme = tmp_path / 'README.md'
    readme.write_text('Initial content\nMore details\n', encoding='utf-8')

    douglas.run_loop()

    assert push_calls == ['push']
    assert pr_calls == ['pr']


def test_per_sprint_policy_defers_push_until_final_day(monkeypatch, tmp_path):
    steps = [
        {'name': 'commit'},
        {'name': 'push'},
    ]
    config_path = _init_repo(tmp_path, push_policy='per_sprint', sprint_length=2, steps=steps)

    provider = SequencedProvider(['feat: add first change', 'feat: add second change'])
    monkeypatch.setattr(Douglas, 'create_llm_provider', lambda self: provider)

    push_calls: list[str] = []

    def fake_push(self):
        push_calls.append('push')
        return True, 'pushed'

    monkeypatch.setattr(Douglas, '_run_git_push', fake_push)
    monkeypatch.setattr(Douglas, '_monitor_ci', lambda self: None)

    douglas = Douglas(config_path)
    readme = tmp_path / 'README.md'

    readme.write_text('Initial content\nDay 1\n', encoding='utf-8')
    douglas.run_loop()
    assert push_calls == []

    readme.write_text('Initial content\nDay 1\nDay 2\n', encoding='utf-8')
    douglas.run_loop()
    assert push_calls == ['push']
