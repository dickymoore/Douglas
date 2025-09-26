import re
from concurrent.futures import ThreadPoolExecutor

from douglas.providers.mock_provider import DeterministicMockProvider


def test_parallel_mock_agents(tmp_path):
    provider = DeterministicMockProvider(project_root=tmp_path, seed=77)
    (tmp_path / "README.md").write_text("# Parallel run\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "src").mkdir()
    contexts = [
        provider.with_context("DeveloperA", "generate"),
        provider.with_context("DeveloperB", "generate"),
    ]
    prompt = "Parallel agent smoke test"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outputs = list(executor.map(lambda ctx: ctx.generate_code(prompt), contexts))

    slugs = []
    for output in outputs:
        match = re.search(r"tests/test_smoke_(?P<slug>[a-f0-9]+)\.py", output)
        assert match is not None
        slugs.append(match.group("slug"))
    assert len(set(slugs)) == len(slugs)
