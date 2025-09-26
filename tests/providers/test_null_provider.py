import json

from douglas.providers.null_provider import NullProvider


def test_null_provider_writes_state(tmp_path):
    provider = NullProvider(project_root=tmp_path)
    context = provider.with_context("Developer", "lint")
    output = context.generate_code("Skip this step")

    assert output.startswith("```") and output.endswith("```")
    path = output.strip("`").split("\n", 1)[0]
    content = output.split("\n", 1)[1].rsplit("\n", 1)[0]
    assert ".douglas/state/null_provider" in path
    data = json.loads(content)
    assert data["reason"] == "skipped by null provider"
    assert data["status"] == "skipped"
