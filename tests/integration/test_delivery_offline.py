from pathlib import Path

import yaml

from douglas.providers.mock_provider import DeterministicMockProvider
from douglas.steps.delivery import DeliveryContext, run_delivery


def _write_backlog(path: Path) -> None:
    backlog = {
        "stories": [
            {
                "id": "US-1",
                "name": "Display onboarding checklist",
                "status": "todo",
            },
            {
                "id": "US-2",
                "name": "Track progress",
                "status": "planned",
            },
            {
                "id": "US-3",
                "name": "Already done",
                "status": "done",
            },
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(backlog, sort_keys=False), encoding="utf-8")


def test_delivery_step_generates_helpers_and_notes(tmp_path: Path) -> None:
    backlog_path = tmp_path / "ai-inbox" / "backlog" / "pre-features.yaml"
    _write_backlog(backlog_path)

    readme_path = tmp_path / "README.md"
    readme_path.write_text("# Demo project\n", encoding="utf-8")

    provider = DeterministicMockProvider(project_root=tmp_path, seed=11)
    context = DeliveryContext(
        project_root=tmp_path,
        backlog_path=backlog_path,
        readme_path=readme_path,
        llm=provider.with_context("Developer", "delivery"),
    )

    result = run_delivery(context)

    assert result.commits == [
        "feat(delivery-us-1): scaffold delivery helper for us-1",
        "feat(delivery-us-2): scaffold delivery helper for us-2",
    ]
    assert result.artifacts == [
        "README.md",
        "delivery_helpers/__init__.py",
        "delivery_helpers/us_1.py",
        "delivery_helpers/us_2.py",
        "tests/delivery/test_us_1.py",
        "tests/delivery/test_us_2.py",
    ]

    helper_one = (tmp_path / "delivery_helpers" / "us_1.py").read_text(encoding="utf-8")
    assert "def deliver_us_1" in helper_one
    assert "US-1:us_1" in helper_one

    test_one = (tmp_path / "tests" / "delivery" / "test_us_1.py").read_text(encoding="utf-8")
    assert "from delivery_helpers.us_1 import deliver_us_1" in test_one
    assert "deliver_us_1(\"ok\") == \"US-1:us_1:ok\"" in test_one

    readme_text = readme_path.read_text(encoding="utf-8")
    assert "<!-- delivery-notes:start -->" in readme_text
    assert "- [ ] US-1: Display onboarding checklist" in readme_text
    assert "- [ ] US-2: Track progress" in readme_text
    assert "US-3" not in readme_text

    repeat = run_delivery(context)
    assert repeat.commits == []
    assert repeat.artifacts == []
