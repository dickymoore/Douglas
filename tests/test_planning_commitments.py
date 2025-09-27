from __future__ import annotations

from douglas.steps import planning


def test_filter_commitments_logs_invalid_entries(monkeypatch) -> None:
    entries = [
        {"id": "COM-1", "title": "Valid item", "status": "todo"},
        {"id": "COM-2", "status": "todo"},  # Missing title triggers validation failure
        "not-a-mapping",  # Invalid type should log warning
        {"id": "COM-3", "title": "Finished", "status": "done"},
    ]

    logged_messages: list[str] = []

    def capture_warning(message: str, *args, **kwargs) -> None:
        formatted = message % args if args else message
        logged_messages.append(formatted)

    monkeypatch.setattr(planning.logger, "warning", capture_warning)

    commitments = planning.filter_commitments(entries)

    assert [commitment.identifier for commitment in commitments] == ["COM-1"]

    assert any("non-mapping commitment" in message for message in logged_messages)
    assert any("invalid commitment" in message for message in logged_messages)
