"""Deterministic mock LLM provider used for offline development loops."""

from __future__ import annotations

import hashlib
import json
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from douglas.domain.backlog import (
    Epic,
    Feature,
    Story,
    render_backlog_markdown,
    serialize_backlog,
)
from douglas.domain.step_result import StepResult
from douglas.providers.llm_provider import LLMProvider


def _normalize_whitespace(value: str) -> str:
    collapsed = " ".join(value.strip().split())
    return collapsed


def _hash_prompt(prompt: str) -> str:
    normalized = _normalize_whitespace(prompt)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _derive_seed(
    base_seed: int,
    agent_label: str,
    step_name: str,
    prompt_hash: str,
) -> int:
    seed_material = f"{base_seed}:{agent_label}:{step_name}:{prompt_hash}".encode(
        "utf-8"
    )
    digest = hashlib.sha256(seed_material).hexdigest()
    return int(digest[:16], 16)


def _slugify_text(text: str, length: int = 8) -> str:
    cleaned = [ch for ch in text.lower() if ch.isalnum()]
    slug = "".join(cleaned)[:length]
    if slug:
        return slug
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:length]


@dataclass
class _MockContext:
    project_root: Path
    agent_label: str
    step_name: str
    base_seed: int

    def workspace_dir(self) -> Path:
        slug = _slugify_text(f"{self.agent_label}-{self.step_name}")
        workspace = self.project_root / ".douglas" / "workspaces" / slug
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace


class _ContextualDeterministicMockProvider(LLMProvider):
    def __init__(self, context: _MockContext) -> None:
        self._context = context

    # Step-specific helpers -------------------------------------------------
    def _plan_entries(self, rng: random.Random) -> list[dict]:
        adjectives = [
            "resilient",
            "maintainable",
            "ergonomic",
            "accessible",
            "discoverable",
            "inclusive",
        ]
        artifacts = [
            "dashboard",
            "onboarding",
            "cli",
            "testing",
            "observability",
            "documentation",
        ]
        count = 1 + rng.randrange(2)
        entries = []
        for idx in range(count):
            adjective = adjectives[(idx + rng.randrange(len(adjectives))) % len(adjectives)]
            artifact = artifacts[(idx + rng.randrange(len(artifacts))) % len(artifacts)]
            title = f"{adjective.title()} {artifact.title()} polish"
            entry_id = f"mock-{_slugify_text(title, 10)}"
            entries.append(
                {
                    "id": entry_id,
                    "title": title,
                    "status": "todo",
                    "owner": self._context.agent_label,
                }
            )
        return entries

    def _render_markdown_backlog(
        self, existing: str, entries: Iterable[dict], marker: str
    ) -> str:
        lines: List[str] = []
        trimmed = existing.rstrip()
        if trimmed:
            lines.append(trimmed)
        lines.append(marker)
        lines.append("## Mock backlog additions")
        for entry in entries:
            lines.append(f"- [ ] {entry['title']} ({entry['id']})")
        rendered = "\n\n".join(lines[:2]) + "\n" + "\n".join(lines[2:])
        return rendered.strip() + "\n"

    def _plan_output(self, prompt: str) -> str:
        blocks = self._plan_blocks(prompt)
        return "\n\n".join(blocks)

    def _plan_blocks(self, prompt: str) -> List[str]:
        prompt_hash = _hash_prompt(prompt)
        seed = _derive_seed(
            self._context.base_seed,
            self._context.agent_label,
            self._context.step_name,
            prompt_hash,
        )
        rng = random.Random(seed)
        entries = self._plan_entries(rng)
        marker = f"<!-- mock-plan:{prompt_hash[:12]} -->"

        backlog_path = self._context.project_root / "ai-inbox" / "backlog.md"
        if backlog_path.exists():
            existing = backlog_path.read_text(encoding="utf-8")
        else:
            backlog_path.parent.mkdir(parents=True, exist_ok=True)
            existing = ""

        if marker in existing:
            rendered_md = existing
        else:
            rendered_md = self._render_markdown_backlog(existing, entries, marker)

        workspace_root = self._context.project_root / ".douglas"
        state_root = workspace_root / "state"
        backlog_state_path = state_root / "backlog.json"
        backlog_state_path.parent.mkdir(parents=True, exist_ok=True)
        if backlog_state_path.exists():
            try:
                data = json.loads(backlog_state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}
        items: List[dict] = list(data.get("items", []))
        existing_ids = {item.get("id") for item in items}
        updated = False
        for entry in entries:
            if entry["id"] not in existing_ids:
                items.append(entry)
                updated = True
        if not updated and marker in existing:
            json_payload = backlog_state_path.read_text(encoding="utf-8")
        else:
            new_payload = {
                "provider": "deterministic_mock",
                "items": items,
                "seed": self._context.base_seed,
            }
            json_payload = json.dumps(new_payload, indent=2, sort_keys=True) + "\n"

        md_block = self._format_block("ai-inbox/backlog.md", rendered_md)
        json_block = self._format_block(
            ".douglas/state/backlog.json", json_payload
        )

        work_items_dir = workspace_root / "work-items"
        work_items_dir.mkdir(parents=True, exist_ok=True)
        status_cycle = ["not_started", "in_progress", "in_test", "finished"]
        work_blocks: List[str] = []
        for idx, entry in enumerate(entries):
            status = status_cycle[idx % len(status_cycle)]
            work_payload = {
                "id": entry["id"],
                "title": entry["title"],
                "status": status,
                "owner": entry["owner"],
                "seed": self._context.base_seed,
            }
            work_path = work_items_dir / f"{entry['id']}.json"
            work_blocks.append(
                self._format_block(
                    self._relative_path(work_path),
                    json.dumps(work_payload, indent=2, sort_keys=True) + "\n",
                )
            )

        return [md_block, json_block, *work_blocks]

    def _render_readme_section(self, existing: str, marker: str, slug: str) -> str:
        if marker in existing:
            return existing
        section = textwrap.dedent(
            f'''
            {existing.rstrip()}\n
            {marker}
            ## Feature notes (mock)

            This deterministic mock iteration captured the `{slug}` scenario to
            validate the offline orchestration pipeline. Re-run with the same
            seed to reproduce this paragraph.
            '''
        ).strip("\n")
        return section + "\n"

    def _render_mock_module(self, module_path: Path, slug: str) -> str:
        content = textwrap.dedent(
            f'''
            """Deterministic mock feature helpers."""


            def describe_mock_feature(multiplier: int = 1) -> str:
                """Return a reproducible identifier for smoke testing."""

                base = f"mock-{slug}"
                return f"{{base}}-{{multiplier}}"
            '''
        ).strip("\n")
        if module_path.exists():
            return module_path.read_text(encoding="utf-8")
        return content + "\n"

    def _render_test_file(self, module_name: str, module_path: Path, slug: str) -> str:
        relative = module_path.relative_to(self._context.project_root)
        content = textwrap.dedent(
            f'''
            """Smoke test for deterministic mock provider output."""

            import importlib.util
            from pathlib import Path


            def _load_module():
                module_path = Path(__file__).resolve().parents[1] / "{relative.as_posix()}"
                spec = importlib.util.spec_from_file_location("{module_name}", module_path)
                module = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(module)
                return module


            def test_describe_mock_feature_stable():
                module = _load_module()
                assert module.describe_mock_feature() == "mock-{slug}-1"
                assert module.describe_mock_feature(2) == "mock-{slug}-2"
            '''
        ).strip("\n")
        return content + "\n"

    def _generate_output(self, prompt: str) -> str:
        prompt_hash = _hash_prompt(prompt)
        seed = _derive_seed(
            self._context.base_seed,
            self._context.agent_label,
            self._context.step_name,
            prompt_hash,
        )
        slug = f"{seed:016x}"[:6]
        marker = f"<!-- mock-generate:{slug} -->"

        readme_path = self._context.project_root / "README.md"
        blocks: List[str] = []
        if readme_path.exists():
            existing = readme_path.read_text(encoding="utf-8")
            updated = self._render_readme_section(existing, marker, slug)
            blocks.append(self._format_block("README.md", updated))

        src_dir = self._context.project_root / "src"
        candidate_module: Optional[Path] = None
        if src_dir.is_dir():
            candidate_module = src_dir / f"mock_feature_{slug}.py"
        else:
            app_py = self._context.project_root / "app.py"
            if app_py.exists():
                candidate_module = app_py
            else:
                python_files = sorted(self._context.project_root.glob("*.py"))
                for candidate in python_files:
                    if candidate.name.startswith("test_"):
                        continue
                    candidate_module = candidate
                    break

        module_name = f"mock_feature_{slug}"
        module_output_path: Optional[Path] = None
        module_content: Optional[str] = None

        if candidate_module is not None and candidate_module.is_file():
            if candidate_module.suffix == ".py" and candidate_module.parent.name == "src":
                module_output_path = candidate_module
                module_content = self._render_mock_module(candidate_module, slug)
            else:
                # Append helper function to existing file.
                existing_source = candidate_module.read_text(encoding="utf-8")
                if marker not in existing_source:
                    appended = textwrap.dedent(
                        f'''
                        {existing_source.rstrip()}\n
                        {marker}


                        def describe_mock_feature(multiplier: int = 1) -> str:
                            """Return a reproducible identifier."""

                            base = f"mock-{slug}"
                            return f"{{base}}-{{multiplier}}"
                        '''
                    ).strip("\n")
                    module_content = appended + "\n"
                else:
                    module_content = existing_source
                module_output_path = candidate_module
        elif candidate_module is not None:
            module_output_path = candidate_module
            module_content = self._render_mock_module(candidate_module, slug)

        if module_output_path is not None and module_content is not None:
            try:
                relative_module = module_output_path.relative_to(
                    self._context.project_root
                )
            except ValueError:
                relative_module = module_output_path
            blocks.append(
                self._format_block(relative_module.as_posix(), module_content)
            )

            tests_dir = self._context.project_root / "tests"
            tests_dir.mkdir(parents=True, exist_ok=True)
            test_path = tests_dir / f"test_smoke_{slug}.py"
            test_content = self._render_test_file(
                module_name, module_output_path, slug
            )
            blocks.append(
                self._format_block(
                    test_path.relative_to(self._context.project_root).as_posix(),
                    test_content,
                )
            )

        blocks.extend(self._plan_blocks(prompt))

        if not blocks:
            return "Deterministic mock provider produced no repository edits."

        return "\n\n".join(blocks)

    def _review_output(self, prompt: str) -> str:
        prompt_hash = _hash_prompt(prompt)
        seed = _derive_seed(
            self._context.base_seed,
            self._context.agent_label,
            self._context.step_name,
            prompt_hash,
        )
        slug = f"{seed:016x}"[:10]
        review_dir = self._context.project_root / "ai-inbox" / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        review_path = review_dir / f"mock-review-{slug}.md"
        content = textwrap.dedent(
            f'''
            # Mock review findings

            - Seed: {self._context.base_seed}
            - Agent: {self._context.agent_label}
            - Step: {self._context.step_name}
            - Prompt hash: {prompt_hash}

            ## Summary

            The deterministic mock provider identified no blocking issues. Ensure
            generated helpers remain covered by smoke tests.
            '''
        ).strip("\n") + "\n"
        return self._format_block(
            review_path.relative_to(self._context.project_root).as_posix(), content
        )

    def _ci_output(self, prompt: str, category: str) -> str:
        prompt_hash = _hash_prompt(prompt)
        seed = _derive_seed(
            self._context.base_seed,
            self._context.agent_label,
            self._context.step_name,
            prompt_hash,
        )
        slug = f"{seed:016x}"[:8]
        state_dir = self._context.project_root / ".douglas" / "state"
        inbox_dir = self._context.project_root / "ai-inbox" / "ci"
        state_dir.mkdir(parents=True, exist_ok=True)
        inbox_dir.mkdir(parents=True, exist_ok=True)

        rng = random.Random(seed)
        check_count = 1 + rng.randrange(3)
        summary = {
            "status": "passed",
            "checks": check_count,
            "seed": self._context.base_seed,
            "category": category,
            "prompt_hash": prompt_hash,
        }
        json_payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
        text_report = textwrap.dedent(
            f'''
            {category.upper()} REPORT (mock)
            Seed: {self._context.base_seed}
            Prompt: {prompt_hash}
            Checks executed: {check_count}
            Result: PASS
            '''
        ).strip("\n") + "\n"

        json_block = self._format_block(
            self._relative_path(
                state_dir / f"{category}_report_{slug}.json"
            ),
            json_payload,
        )
        text_block = self._format_block(
            self._relative_path(
                inbox_dir / f"{category}_report_{slug}.txt"
            ),
            text_report,
        )
        return f"{json_block}\n\n{text_block}"

    # Generic helpers -------------------------------------------------------
    def _format_block(self, path: str, content: str) -> str:
        return f"```{path}\n{content.rstrip()}\n```"

    def _relative_path(self, path: Path) -> str:
        try:
            relative = path.relative_to(self._context.project_root)
        except ValueError:
            relative = path
        return relative.as_posix()

    def generate_step_result(
        self,
        prompt: str,
        *,
        step_name: str | None = None,
        agent: str | None = None,
        role: str | None = None,
        seed: int | None = None,
        prompt_hash: str | None = None,
        timestamps: dict | None = None,
    ) -> StepResult:
        resolved_step = step_name or self._context.step_name
        resolved_agent = agent or self._context.agent_label
        resolved_role = role or self._context.agent_label
        prompt_digest = prompt_hash or _hash_prompt(prompt)
        derived_seed = seed
        if derived_seed is None:
            derived_seed = _derive_seed(
                self._context.base_seed,
                resolved_agent,
                resolved_step,
                prompt_digest,
            )
        return super().generate_step_result(
            prompt,
            step_name=resolved_step,
            agent=resolved_agent,
            role=resolved_role,
            seed=derived_seed,
            prompt_hash=prompt_digest,
            timestamps=timestamps,
        )

    def generate_code(self, prompt: str) -> str:
        step = self._context.step_name.lower()
        if step == "sprint_zero":
            return self._sprint_zero_output(prompt)
        if step == "plan":
            return self._plan_output(prompt)
        if step == "generate":
            return self._generate_output(prompt)
        if step == "review":
            return self._review_output(prompt)
        if step in {"lint", "typecheck", "security", "test"}:
            return self._ci_output(prompt, step)
        return (
            "Offline mock provider executed without generating edits."
        )

    def _sprint_zero_output(self, prompt: str) -> str:
        prompt_hash = _hash_prompt(prompt)
        seed = _derive_seed(
            self._context.base_seed,
            self._context.agent_label,
            self._context.step_name,
            prompt_hash,
        )
        rng = random.Random(seed)

        epic_models: List[Epic] = []
        feature_models: List[Feature] = []
        story_models: List[Story] = []

        themes = [
            "platform enablement",
            "developer experience",
            "quality insights",
            "release confidence",
            "user onboarding",
        ]
        descriptors = [
            "resilient",
            "delightful",
            "accessible",
            "collaborative",
            "automated",
            "auditable",
        ]
        capabilities = [
            "analytics",
            "workflow",
            "observability",
            "documentation",
            "security",
            "feedback",
        ]
        outcomes = [
            "dashboard",
            "pipeline",
            "playbook",
            "portal",
            "insights",
            "reporting",
        ]
        personas = [
            "developer",
            "tester",
            "operator",
            "stakeholder",
            "customer",
        ]
        actions = [
            "track",
            "validate",
            "share",
            "configure",
            "monitor",
            "experiment with",
        ]

        epic_count = 3
        features_per_epic = 2
        stories_per_feature = 2

        for epic_index in range(epic_count):
            theme = themes[(epic_index + rng.randrange(len(themes))) % len(themes)]
            descriptor = descriptors[(epic_index + rng.randrange(len(descriptors))) % len(descriptors)]
            epic_title = f"{descriptor.title()} {theme.title()} Initiative"
            epic_id = f"EP-{_slugify_text(epic_title, 10)}"
            epic_description = (
                f"Deliver a {descriptor} {theme} experience for early project alignment."
            )
            epic_models.append(
                Epic(
                    identifier=epic_id,
                    title=epic_title,
                    description=epic_description,
                )
            )

            for feature_offset in range(features_per_epic):
                capability = capabilities[
                    (feature_offset + rng.randrange(len(capabilities)))
                    % len(capabilities)
                ]
                outcome = outcomes[
                    (feature_offset + rng.randrange(len(outcomes))) % len(outcomes)
                ]
                feature_title = f"{capability.title()} {outcome.title()}"
                feature_id = f"FT-{_slugify_text(feature_title + epic_id, 10)}"
                feature_description = (
                    f"Provide {capability} capabilities through a {outcome} focused on {theme}."
                )
                feature_models.append(
                    Feature(
                        identifier=feature_id,
                        title=feature_title,
                        epic_id=epic_id,
                        description=feature_description,
                        acceptance_criteria=(
                            f"Supports {theme} objectives",
                            f"Improves {descriptor} metrics",
                        ),
                    )
                )

                for story_offset in range(stories_per_feature):
                    persona = personas[
                        (story_offset + rng.randrange(len(personas))) % len(personas)
                    ]
                    action = actions[
                        (story_offset + rng.randrange(len(actions))) % len(actions)
                    ]
                    story_title = (
                        f"As a {persona}, I want to {action} the {outcome} outcomes."
                    )
                    story_id = f"ST-{_slugify_text(story_title + feature_id, 12)}"
                    story_description = (
                        f"Enable the {persona} persona to {action} the {outcome} produced by {feature_title}."
                    )
                    story_models.append(
                        Story(
                            identifier=story_id,
                            title=story_title,
                            feature_id=feature_id,
                            description=story_description,
                            acceptance_criteria=(
                                f"{persona.title()} can {action} without assistance",
                                f"{outcome.title()} data is captured for analytics",
                            ),
                        )
                    )

        backlog_dict = serialize_backlog(epic_models, feature_models, story_models)
        backlog_json = json.dumps(backlog_dict, indent=2, sort_keys=True) + "\n"
        backlog_markdown = render_backlog_markdown(
            self._context.project_root.name,
            epic_models,
            feature_models,
            story_models,
        )
        ci_stub = textwrap.dedent(
            """name: Sprint Zero CI
on:
  push:
    branches: [ main ]
  pull_request:
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Sprint Zero placeholder
        run: echo \"Sprint Zero generated initial workflow\"
"""
        )

        payload = {
            "epics": backlog_dict["epics"],
            "features": backlog_dict["features"],
            "stories": backlog_dict["stories"],
            "artifacts": {
                ".douglas/state/backlog.json": backlog_json,
                "ai-inbox/backlog.md": backlog_markdown,
                ".github/workflows/app.yml": ci_stub,
            },
            "metadata": {
                "seed": seed,
                "prompt_hash": prompt_hash,
            },
        }
        return json.dumps(payload, indent=2, sort_keys=True)


class DeterministicMockProvider(LLMProvider):
    provider_id = "mock"

    def __init__(self, project_root: Path, seed: int = 0) -> None:
        self.project_root = Path(project_root)
        self.seed = int(seed)
        self._contexts: Dict[
            Tuple[str, str], _ContextualDeterministicMockProvider
        ] = {}

    def with_context(self, agent_label: str, step_name: str) -> LLMProvider:
        key = (agent_label or "agent", step_name or "step")
        if key not in self._contexts:
            context = _MockContext(
                project_root=self.project_root,
                agent_label=agent_label,
                step_name=step_name,
                base_seed=self.seed,
            )
            self._contexts[key] = _ContextualDeterministicMockProvider(context)
        return self._contexts[key]

    def generate_code(self, prompt: str) -> str:
        raise RuntimeError(
            "DeterministicMockProvider requires contextualisation via with_context()."
        )
