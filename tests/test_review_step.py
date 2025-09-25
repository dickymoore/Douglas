from pathlib import Path


from douglas.core import Douglas


class DummyProvider:
    def __init__(self, response):
        self.response = response
        self.prompts: list[str] = []
        self.called = False

    def generate_code(self, prompt: str) -> str:
        self.prompts.append(prompt)
        self.called = True
        return self.response


def _write_basic_config(tmp_path: Path) -> Path:
    config = (
        "project:\n"
        "  name: 'ReviewTest'\n"
        "  language: 'python'\n"
        "ai:\n"
        "  provider: 'openai'\n"
        "loop:\n"
        "  steps: []\n"
    )
    config_path = tmp_path / "douglas.yaml"
    config_path.write_text(config, encoding="utf-8")
    return config_path


def test_review_invokes_llm_and_saves_feedback(monkeypatch, tmp_path, capsys):
    config_path = _write_basic_config(tmp_path)
    provider = DummyProvider("Looks good to me.")
    monkeypatch.setattr(Douglas, "create_llm_provider", lambda self: provider)

    douglas = Douglas(config_path)
    monkeypatch.setattr(
        Douglas,
        "_get_pending_diff",
        lambda self: 'diff --git a/foo b/foo\n+print("hi")',
    )

    douglas.review()

    captured = capsys.readouterr()
    assert "Language model review feedback:" in captured.out
    assert "Looks good to me." in captured.out

    review_file = tmp_path / "douglas_review.md"
    assert review_file.exists()
    content = review_file.read_text(encoding="utf-8")
    assert "Looks good to me." in content
    assert provider.prompts
    assert "CHANGES TO REVIEW" in provider.prompts[0]


def test_review_skips_when_no_diff(monkeypatch, tmp_path, capsys):
    config_path = _write_basic_config(tmp_path)
    provider = DummyProvider("Should not be used")
    monkeypatch.setattr(Douglas, "create_llm_provider", lambda self: provider)

    douglas = Douglas(config_path)
    monkeypatch.setattr(Douglas, "_get_pending_diff", lambda self: "")

    douglas.review()

    captured = capsys.readouterr()
    assert "No code changes detected for review; skipping." in captured.out
    assert not provider.called

    review_file = tmp_path / "douglas_review.md"
    assert not review_file.exists()
