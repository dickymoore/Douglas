import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from douglas.core import Douglas


def test_init_project_creates_scaffold(monkeypatch, tmp_path):
    base_config = tmp_path / 'douglas.yaml'
    base_config.write_text('project:\n  name: Base\n  language: python\n', encoding='utf-8')

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

    scaffold_config = yaml.safe_load((target_dir / 'douglas.yaml').read_text(encoding='utf-8'))
    assert scaffold_config['project']['name'] == target_dir.name
    assert scaffold_config['project']['language'] == 'python'
    assert scaffold_config['push_policy'] == 'per_feature'
    assert scaffold_config['loop']['exit_conditions'] == ['ci_pass']

    readme_text = (target_dir / 'README.md').read_text(encoding='utf-8')
    assert f"This Python project" in readme_text
    assert target_dir.name in readme_text

    system_prompt = (target_dir / 'system_prompt.md').read_text(encoding='utf-8')
    assert target_dir.name in system_prompt
    assert 'python project' in system_prompt.lower()

    main_py = (target_dir / 'src' / 'main.py').read_text(encoding='utf-8')
    assert target_dir.name in main_py
    assert 'scaffolded python application' in main_py.lower()

    test_main = (target_dir / 'tests' / 'test_main.py').read_text(encoding='utf-8')
    assert f'"{target_dir.name}"' in test_main

    gitignore_text = (target_dir / '.gitignore').read_text(encoding='utf-8')
    assert target_dir.name in gitignore_text.splitlines()[0]

    ci_content = (target_dir / '.github' / 'workflows' / 'ci.yml').read_text(encoding='utf-8')
    assert 'name: CI' in ci_content
    assert 'pytest' in ci_content
    assert 'the python project' in ci_content


def test_init_project_respects_language_override(monkeypatch, tmp_path):
    base_config = tmp_path / 'douglas.yaml'
    base_config.write_text('project:\n  name: Base\n  language: go\n', encoding='utf-8')

    monkeypatch.setattr(Douglas, 'create_llm_provider', lambda self: object())

    douglas = Douglas(base_config)
    target_dir = tmp_path / 'scaffold'

    douglas.init_project(str(target_dir))

    readme_text = (target_dir / 'README.md').read_text(encoding='utf-8')
    assert 'Go project' in readme_text

    system_prompt = (target_dir / 'system_prompt.md').read_text(encoding='utf-8')
    assert 'go project' in system_prompt.lower()

    main_py = (target_dir / 'src' / 'main.py').read_text(encoding='utf-8')
    assert 'scaffolded go application' in main_py.lower()

    ci_content = (target_dir / '.github' / 'workflows' / 'ci.yml').read_text(encoding='utf-8')
    assert 'the go project' in ci_content

    scaffold_config = yaml.safe_load((target_dir / 'douglas.yaml').read_text(encoding='utf-8'))
    assert scaffold_config['project']['language'] == 'go'
    assert scaffold_config['push_policy'] == 'per_feature'
