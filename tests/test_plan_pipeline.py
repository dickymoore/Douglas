from pathlib import Path

import textwrap
import yaml

from douglas.pipelines import plan as planpipe


class DummyLLM:
    def __init__(self, responses) -> None:
        if isinstance(responses, str):
            responses = [responses]
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate_code(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            return ""
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)


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
        agent_roles=[
            "Product Owner",
            "Developer",
            "Tester",
            "DevOps",
            "BA",
            "Account Manager",
        ],
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
    charter_yaml = textwrap.dedent(
        """
        agents_md: |
          # Agents
          - Product Owner
        agent_charter_md: |
          # Charter
          Mission statement.
        coding_guidelines_md: |
          # Coding Guidelines
          - Write tests
        working_agreements_md: |
          # Working Agreements
          - Daily standup
        """
    )
    llm = DummyLLM([response, charter_yaml])
    context = _build_context(tmp_path, llm)

    result = planpipe.run_plan(context)

    assert result.created_backlog is True
    assert result.backlog_path.exists()
    data = yaml.safe_load(result.backlog_path.read_text(encoding="utf-8"))
    assert data["epics"][0]["id"] == "EP-1"
    assert llm.prompts  # ensure LLM was invoked
    assert "agents_md" in result.charter_paths
    agents_path = result.charter_paths["agents_md"]
    assert agents_path.exists()
    agent_text = agents_path.read_text(encoding="utf-8")
    assert "Product Owner" in agent_text
    assert "Business Analyst" in agent_text
    assert "Account Manager" in agent_text


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
    charter_yaml = textwrap.dedent(
        """
        agents_md: |
          # Agents
          - Product Owner
        agent_charter_md: |
          # Charter
          Mission statement.
        coding_guidelines_md: |
          # Coding Guidelines
          - Write tests
        working_agreements_md: |
          # Working Agreements
          - Daily standup
        """
    )
    llm = DummyLLM([response, charter_yaml])
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
    assert result.charter_paths


def test_run_plan_skips_without_llm(tmp_path):
    context = _build_context(tmp_path, llm=None)

    result = planpipe.run_plan(context)

    assert result.created_backlog is False
    assert result.reason == "no_llm"
    assert result.charter_paths == {}


def test_run_plan_handles_cli_error_without_overwrite(tmp_path):
    existing_backlog = {"epics": [{"id": "EP-1", "name": "Keep me"}]}
    context = _build_context(tmp_path, llm=DummyLLM(["ERROR: 401 Unauthorized"]))
    context.backlog_path.parent.mkdir(parents=True, exist_ok=True)
    context.backlog_path.write_text(
        yaml.safe_dump(existing_backlog, sort_keys=False),
        encoding="utf-8",
    )

    result = planpipe.run_plan(context)

    assert result.created_backlog is False
    assert result.reason == "llm_error"
    persisted = yaml.safe_load(context.backlog_path.read_text(encoding="utf-8"))
    assert persisted == existing_backlog


def test_run_plan_generates_fallback_when_backlog_empty(tmp_path):
    charter_yaml = textwrap.dedent(
        """
        agents_md: |
          # Agents
          - Product Owner
        agent_charter_md: |
          # Charter
          Mission statement.
        coding_guidelines_md: |
          # Coding Guidelines
          - Write tests
        working_agreements_md: |
          # Working Agreements
          - Daily standup
        """
    )
    llm = DummyLLM(["plan: null", charter_yaml])
    context = _build_context(tmp_path, llm)

    result = planpipe.run_plan(context)

    assert result.created_backlog is True
    assert result.reason == "fallback"
    data = yaml.safe_load(result.backlog_path.read_text(encoding="utf-8"))
    assert data["epics"][0]["id"] == "EP-1"
    assert data.get("raw", {}).get("plan") is None
    assert result.epic_count() >= 1
