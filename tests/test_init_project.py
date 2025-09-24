import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from douglas.core import Douglas


def test_init_project_creates_python_scaffold(monkeypatch, tmp_path):
    base_config = tmp_path / "douglas.yaml"
    base_config.write_text(
        "project:\n  name: Base\n  language: python\n", encoding="utf-8"
    )

    monkeypatch.setattr(Douglas, "create_llm_provider", lambda self: object())

    douglas = Douglas(base_config)
    target_dir = tmp_path / "scaffold"

    douglas.init_project(
        target_dir,
        push_policy="per_feature_complete",
        sprint_length=6,
    )

    expected_paths = {
        target_dir / "douglas.yaml",
        target_dir / "README.md",
        target_dir / ".env.example",
        target_dir / ".gitignore",
        target_dir / "system_prompt.md",
        target_dir / "pyproject.toml",
        target_dir / "requirements-dev.txt",
        target_dir / "Makefile",
        target_dir / "src" / "app" / "__init__.py",
        target_dir / "tests" / "test_app.py",
        target_dir / ".github" / "workflows" / "ci.yml",
    }

    missing = [str(path) for path in expected_paths if not path.exists()]
    assert not missing, f"Missing expected scaffold files: {missing}"

    scaffold_config = yaml.safe_load(
        (target_dir / "douglas.yaml").read_text(encoding="utf-8")
    )
    assert scaffold_config["project"]["name"] == target_dir.name
    assert scaffold_config["project"]["language"] == "python"
    assert scaffold_config["push_policy"] == "per_feature_complete"
    assert scaffold_config["loop"]["exit_conditions"] == ["ci_pass"]
    assert [step["name"] for step in scaffold_config["loop"]["steps"]] == [
        "generate",
        "lint",
        "typecheck",
        "test",
        "commit",
        "push",
        "pr",
    ]
    assert scaffold_config["sprint"]["length_days"] == 6
    assert scaffold_config["history"]["max_log_excerpt_length"] == 4000

    readme_text = (target_dir / "README.md").read_text(encoding="utf-8")
    assert "Douglas" in readme_text
    assert target_dir.name in readme_text

    system_prompt = (target_dir / "system_prompt.md").read_text(encoding="utf-8")
    assert target_dir.name in system_prompt
    assert "src/app" in system_prompt

    package_code = (target_dir / "src" / "app" / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert "get_welcome_message" in package_code
    assert target_dir.name in package_code

    test_code = (target_dir / "tests" / "test_app.py").read_text(encoding="utf-8")
    assert "get_welcome_message" in test_code
    assert target_dir.name in test_code

    gitignore_text = (target_dir / ".gitignore").read_text(encoding="utf-8")
    assert ".venv" in gitignore_text
    assert ".pytest_cache" in gitignore_text

    env_example = (target_dir / ".env.example").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" in env_example

    pyproject_text = (target_dir / "pyproject.toml").read_text(encoding="utf-8")
MERGE_CONFLICT< codex/implement-bootstrapping-command-and-readme-update
    assert f"name = \"{target_dir.name.lower().replace('-', '_')}" in pyproject_text
MERGE_CONFLICT=
    assert f"name = \"{target_dir.name.lower().replace('-', '_')}\"" in pyproject_text
MERGE_CONFLICT> main

    requirements_text = (target_dir / "requirements-dev.txt").read_text(
        encoding="utf-8"
    )
    assert "pytest" in requirements_text

    makefile = (target_dir / "Makefile").read_text(encoding="utf-8")
    assert "venv" in makefile and "pytest -q" in makefile

    ci_content = (target_dir / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    assert "actions/setup-python" in ci_content
    assert "pytest -q" in ci_content


def test_init_project_supports_blank_template(monkeypatch, tmp_path):
    base_config = tmp_path / "douglas.yaml"
    base_config.write_text(
        "project:\n  name: Base\n  language: rust\n", encoding="utf-8"
    )

    monkeypatch.setattr(Douglas, "create_llm_provider", lambda self: object())

    douglas = Douglas(base_config)
    target_dir = tmp_path / "blank"

    douglas.init_project(
        target_dir,
        template="blank",
        ci="none",
        license_type="none",
    )

    assert (target_dir / "douglas.yaml").exists()
    assert (target_dir / "README.md").exists()
    assert (target_dir / ".env.example").exists()
    assert (target_dir / ".gitignore").exists()
    assert not (target_dir / "system_prompt.md").exists()
    assert not (target_dir / "src").exists()

    scaffold_config = yaml.safe_load(
        (target_dir / "douglas.yaml").read_text(encoding="utf-8")
    )
    assert scaffold_config["project"]["language"] == "rust"
    assert scaffold_config["push_policy"] == "per_feature"
    assert "prompt" not in scaffold_config["ai"]
