import re

from douglas.providers.mock_provider import DeterministicMockProvider


def test_mock_provider_deterministic(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# Example\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "src").mkdir()

    provider = DeterministicMockProvider(project_root=tmp_path, seed=42)
    developer = provider.with_context("Developer", "generate")

    prompt = "Generate helpful scaffolding"
    first = developer.generate_code(prompt)
    second = developer.generate_code(prompt)

    assert first == second

    repeat_provider = DeterministicMockProvider(project_root=tmp_path, seed=42)
    repeat_output = repeat_provider.with_context("Developer", "generate").generate_code(prompt)
    assert repeat_output == first

    tester_output = provider.with_context("Tester", "generate").generate_code(prompt)

    dev_slug = re.search(r"tests/test_smoke_(?P<slug>[a-f0-9]+)\.py", first)
    tester_slug = re.search(r"tests/test_smoke_(?P<slug>[a-f0-9]+)\.py", tester_output)
    assert dev_slug is not None and tester_slug is not None
    assert dev_slug.group("slug") != tester_slug.group("slug")
