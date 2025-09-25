from types import SimpleNamespace


from douglas.providers.openai_provider import OpenAIProvider


def _make_provider() -> OpenAIProvider:
    # Bypass __init__ to avoid environment-dependent setup.
    return OpenAIProvider.__new__(OpenAIProvider)  # type: ignore[call-arg]


def test_extract_responses_text_handles_list_content_from_choices() -> None:
    provider = _make_provider()
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=[
                        {"text": {"value": "First"}},
                        {"text": "Second"},
                        SimpleNamespace(text=SimpleNamespace(value="Third")),
                        "Fourth",
                        {"value": "Fifth"},
                    ]
                )
            )
        ]
    )

    text = provider._extract_responses_text(response)

    assert text == "First\nSecond\nThird\nFourth\nFifth"


def test_extract_responses_text_handles_model_dump_payload() -> None:
    provider = _make_provider()

    class ResponseWithModelDump:
        def __init__(self) -> None:
            self._payload = {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"text": {"value": "Alpha"}},
                                {"text": {"value": "Beta"}},
                            ]
                        }
                    }
                ]
            }

        def model_dump(self) -> dict:
            return self._payload

    response = ResponseWithModelDump()

    text = provider._extract_responses_text(response)

    assert text == "Alpha\nBeta"


def test_extract_responses_text_handles_output_segments() -> None:
    provider = _make_provider()

    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                content=[
                    SimpleNamespace(text=SimpleNamespace(value="Gamma")),
                    {"type": "output_text", "text": {"value": "Delta"}},
                ]
            ),
            {"content": [{"text": "Epsilon"}, {"value": "Zeta"}]},
        ]
    )

    text = provider._extract_responses_text(response)

    assert text == "Gamma\nDelta\nEpsilon\nZeta"
