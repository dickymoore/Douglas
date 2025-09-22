import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from douglas.core import Douglas


class RecordingProvider:
    def __init__(self, response):
        self.response = response
        self.prompts: list[str] = []
        self.called = False

    def generate_code(self, prompt: str) -> str:
        self.prompts.append(prompt)
        self.called = True
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FailingProvider(RecordingProvider):
    def __init__(self):
        super().__init__(RuntimeError("llm failure"))


def _init_git_repo(tmp_path: Path) -> Path:
    subprocess.run(['git', 'init'], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=tmp_path, check=True, capture_output=True, text=True)

    config_content = textwrap.dedent(
        """
        project:
          name: 'CommitTest'
          language: 'python'
        ai:
          provider: 'openai'
        loop:
          steps: []
        """
    )
    (tmp_path / 'douglas.yaml').write_text(config_content, encoding='utf-8')
    (tmp_path / 'README.md').write_text("Initial content\n", encoding='utf-8')

    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(['git', 'commit', '-m', 'chore: initial'], cwd=tmp_path, check=True, capture_output=True, text=True)

    return tmp_path / 'douglas.yaml'


def _git_output(path: Path, *args: str) -> str:
    result = subprocess.run(['git', *args], cwd=path, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def test_commit_step_creates_commit_with_llm_message(monkeypatch, tmp_path):
    config_path = _init_git_repo(tmp_path)
    provider = RecordingProvider("feat: update readme.\n\n- add more details")
    monkeypatch.setattr(Douglas, 'create_llm_provider', lambda self: provider)

    douglas = Douglas(config_path)
    initial_head = _git_output(tmp_path, 'rev-parse', 'HEAD')

    readme = tmp_path / 'README.md'
    readme.write_text("Initial content\nMore docs\n", encoding='utf-8')

    douglas.run_loop()

    new_head = _git_output(tmp_path, 'rev-parse', 'HEAD')
    assert new_head != initial_head

    commit_message = _git_output(tmp_path, 'log', '-1', '--pretty=%B')
    assert commit_message == 'feat: update readme'
    assert provider.called
    assert provider.prompts
    assert 'STAGED DIFF' in provider.prompts[0]

    status = _git_output(tmp_path, 'status', '--porcelain')
    assert status == ''


def test_commit_step_falls_back_when_llm_fails(monkeypatch, tmp_path):
    config_path = _init_git_repo(tmp_path)
    provider = FailingProvider()
    monkeypatch.setattr(Douglas, 'create_llm_provider', lambda self: provider)

    douglas = Douglas(config_path)
    initial_head = _git_output(tmp_path, 'rev-parse', 'HEAD')

    target = tmp_path / 'README.md'
    target.write_text("Initial content\nAnother line\n", encoding='utf-8')

    douglas.run_loop()

    new_head = _git_output(tmp_path, 'rev-parse', 'HEAD')
    assert new_head != initial_head

    commit_message = _git_output(tmp_path, 'log', '-1', '--pretty=%B')
    assert commit_message == Douglas.DEFAULT_COMMIT_MESSAGE
    assert provider.called


def test_commit_step_skips_when_no_changes(monkeypatch, tmp_path):
    config_path = _init_git_repo(tmp_path)
    provider = RecordingProvider('feat: unused message')
    monkeypatch.setattr(Douglas, 'create_llm_provider', lambda self: provider)

    douglas = Douglas(config_path)
    initial_head = _git_output(tmp_path, 'rev-parse', 'HEAD')

    douglas.run_loop()

    final_head = _git_output(tmp_path, 'rev-parse', 'HEAD')
    assert final_head == initial_head
    assert not provider.called
