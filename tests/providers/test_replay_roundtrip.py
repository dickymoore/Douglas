import pytest

from douglas.providers.llm_provider import LLMProvider
from douglas.providers.replay_provider import (
    CassetteRecordingProvider,
    CassetteStore,
    ReplayProvider,
    compute_project_fingerprint,
)


class _StaticProvider(LLMProvider):
    def __init__(self, response: str) -> None:
        self._response = response

    def generate_code(self, prompt: str) -> str:  # pragma: no cover - trivial
        return self._response


def test_replay_roundtrip(tmp_path):
    project_root = tmp_path
    (project_root / "douglas.yaml").write_text("ai: {mode: real}\n", encoding="utf-8")
    store_dir = project_root / ".douglas" / "cassettes"
    store = CassetteStore(store_dir)
    ai_config = {"mode": "real", "seed": 123}
    fingerprint = compute_project_fingerprint(project_root, ai_config)

    base = _StaticProvider("mocked output text")
    recorder = CassetteRecordingProvider(
        base,
        store=store,
        provider_name="dummy",
        model_name="offline-test",
        project_fingerprint=fingerprint,
        base_seed=123,
    )

    prompt = "Capture this prompt"
    recorded = recorder.with_context("Developer", "plan")
    response = recorded.generate_code(prompt)
    assert "mocked output text" in response

    replay = ReplayProvider(
        store=store,
        project_root=project_root,
        project_fingerprint=fingerprint,
        base_seed=123,
        model_name="offline-test",
        provider_name="dummy",
    )
    replayed = replay.with_context("Developer", "plan").generate_code(prompt)
    assert replayed == response

    with pytest.raises(KeyError):
        replay.with_context("Tester", "plan").generate_code("Different prompt")

    files = list(store_dir.glob("*.jsonl"))
    assert files, "Expected cassette file to be written"
    contents = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert any("mocked output text" in line for line in contents)
