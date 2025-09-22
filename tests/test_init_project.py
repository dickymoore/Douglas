import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from douglas.core import Douglas


def test_init_project_creates_scaffold(monkeypatch, tmp_path):
    base_config = tmp_path / 'douglas.yaml'
    base_config.write_text('project:\n  name: Base\n', encoding='utf-8')

    monkeypatch.setattr(Douglas, 'create_llm_provider', lambda self: object())

    douglas = Douglas(base_config)
    target_dir = tmp_path / 'scaffold'

    douglas.init_project(str(target_dir))

    expected_paths = [
        target_dir / 'douglas.yaml',
        target_dir / 'README.md',
        target_dir / 'system_prompt.md',
        target_dir / '.gitignore',
        target_dir / 'src' / '__init__.py',
        target_dir / 'src' / 'main.py',
        target_dir / 'tests' / 'test_main.py',
        target_dir / '.github' / 'workflows' / 'ci.yml',
    ]

    for path in expected_paths:
        assert path.exists(), f'Missing expected scaffold file: {path}'

    ci_content = (target_dir / '.github' / 'workflows' / 'ci.yml').read_text(encoding='utf-8')
    assert 'name: CI' in ci_content
    assert 'pytest' in ci_content

    readme_text = (target_dir / 'README.md').read_text(encoding='utf-8')
    assert target_dir.name in readme_text
    assert 'Douglas' in readme_text
