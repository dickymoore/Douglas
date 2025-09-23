from pathlib import Path

from douglas.journal import questions


def _build_context(tmp_path: Path) -> dict:
    config = {
        "paths": {
            "questions_dir": "user-portal/questions",
            "questions_archive_dir": "user-portal/questions-archive",
            "sprint_prefix": "sprint-",
        },
        "qna": {"filename_pattern": "sprint-{sprint}-{role}-{id}.md"},
    }
    return {
        "project_root": tmp_path,
        "config": config,
    }


def test_raise_and_archive_question(tmp_path):
    context = _build_context(tmp_path)
    context.update(
        {
            "context": "Need clarification about deployment process.",
            "question": "Which environment should we target first?",
        }
    )

    question_id = questions.raise_question(
        role="Developer",
        sprint=2,
        topic="Deployment clarification",
        context_data=context,
        blocking=True,
    )

    questions_dir = tmp_path / "user-portal" / "questions"
    files = list(questions_dir.glob("*.md"))
    assert len(files) == 1
    question_path = files[0]
    content = question_path.read_text(encoding="utf-8")

    assert f"id: {question_id}" in content
    assert "status: OPEN" in content
    assert "## User Answer" in content

    summary_path = (
        tmp_path
        / "ai-inbox"
        / "sprints"
        / "sprint-2"
        / "roles"
        / "developer"
        / "summary.md"
    )
    assert summary_path.exists()
    summary_text = summary_path.read_text(encoding="utf-8")
    assert question_id in summary_text

    # Simulate the user providing an answer
    updated = content.replace(
        "## User Answer\n\n", "## User Answer\nWe deployed to staging.\n"
    )
    question_path.write_text(updated, encoding="utf-8")

    scan_context = _build_context(tmp_path)
    open_questions = questions.scan_for_answers(scan_context)
    assert len(open_questions) == 1
    captured = open_questions[0]
    assert captured.user_answer.strip() == "We deployed to staging."
    assert captured.blocking is True

    captured.agent_follow_up = "Thanks! We'll proceed with staging verification."
    destination = questions.archive_question(captured)

    assert destination.exists()
    archived_text = destination.read_text(encoding="utf-8")
    assert "status: ANSWERED" in archived_text
    assert "closed_at:" in archived_text
    assert "Thanks! We'll proceed with staging verification." in archived_text

    # Summary should note the answered state
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "[ANSWERED]" in summary_text

    # Original file should be moved to the archive directory
    assert not question_path.exists()
    archive_dir = tmp_path / "user-portal" / "questions-archive"
    assert destination.parent == archive_dir
