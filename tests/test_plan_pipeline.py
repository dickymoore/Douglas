from pathlib import Path

import textwrap
import yaml

from douglas.pipelines import plan as planpipe


class DummyLLM:
    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[str] = []

    def generate_code(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.text


def _build_context(tmp_path: Path, llm) -> planpipe.PlanContext:
    project_root = tmp_path
    backlog_path = project_root / "ai-inbox" / "backlog" / "pre-features.yaml"
    system_prompt_path = project_root / "system_prompt.md"
    system_prompt_path.write_text("Build a marketplace", encoding="utf-8")
    planning_config = {
        "enabled": True,
        "sprint_zero_only": True,
        "backlog_file": "ai-inbox/backlog/pre-features.yaml",
    }

    return planpipe.PlanContext(
        project_name="Sample Project",
        project_description="Marketplace for artisans",
        project_root=project_root,
        backlog_path=backlog_path,
        system_prompt_path=system_prompt_path,
        sprint_index=1,
        sprint_day=1,
        planning_config=planning_config,
        llm=llm,
    )


def test_run_plan_creates_backlog(tmp_path):
    response = textwrap.dedent(
        """
    epics:
      - id: EP-1
        name: Seller onboarding
        objective: Allow artisans to list products
        success_metrics:
          - 50 sellers onboarded in sprint 1
        features:
          - FE-1
    features:
      - id: FE-1
        name: Listing wizard
        epic: EP-1
        narrative: As a seller I can add listings quickly
        business_value: Increase supply side growth
        stories:
          - US-1
    stories:
      - id: US-1
        name: Create a draft listing
        feature: FE-1
        description: Sellers can save draft listings
        acceptance_criteria:
          - Draft persists between sessions
        tasks:
          - TK-1
    tasks:
      - id: TK-1
        story: US-1
        description: Implement draft persistence
        estimate: 3
    """
    )
    llm = DummyLLM(response)
    context = _build_context(tmp_path, llm)

    result = planpipe.run_plan(context)

    assert result.created_backlog is True
    assert result.backlog_path.exists()
    data = yaml.safe_load(result.backlog_path.read_text(encoding="utf-8"))
    assert data["epics"][0]["id"] == "EP-1"
    assert llm.prompts  # ensure LLM was invoked


def test_run_plan_merges_with_existing_backlog(tmp_path):
    existing_backlog = {
        "epics": [{"id": "EP-0", "name": "Baseline"}],
        "stories": [{"id": "US-0", "name": "Initial story"}],
    }
    response = textwrap.dedent(
        """
    epics:
      - id: EP-1
        name: New Epic
        features: []
    stories:
      - id: US-1
        name: Follow-up story
    """
    )
    llm = DummyLLM(response)
    context = _build_context(tmp_path, llm)
    context.backlog_path.parent.mkdir(parents=True, exist_ok=True)
    context.backlog_path.write_text(
        yaml.safe_dump(existing_backlog, sort_keys=False),
        encoding="utf-8",
    )

    result = planpipe.run_plan(context)

    assert result.created_backlog is True
    merged = yaml.safe_load(context.backlog_path.read_text(encoding="utf-8"))
    epic_ids = {epic["id"] for epic in merged.get("epics", [])}
    assert epic_ids == {"EP-0", "EP-1"}
    story_ids = {story["id"] for story in merged.get("stories", [])}
    assert story_ids == {"US-0", "US-1"}


def test_run_plan_skips_without_llm(tmp_path):
    context = _build_context(tmp_path, llm=None)

    result = planpipe.run_plan(context)

    assert result.created_backlog is False
    assert result.reason == "no_llm"
