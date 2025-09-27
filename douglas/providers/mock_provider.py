"""Deterministic mock LLM provider used for offline development loops."""

from __future__ import annotations

import hashlib
import json
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from douglas.providers.llm_provider import LLMProvider
from douglas.steps.ci import CIStep
from douglas.steps.testing import OfflineTestingConfig, OfflineTestingStep


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


def _slugify(text: str, length: int = 8) -> str:
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
        slug = _slugify(f"{self.agent_label}-{self.step_name}")
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
            entry_id = f"mock-{_slugify(title, 10)}"
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
        rng = random.Random(seed)

        if category == "test":
            low = 0.6 + rng.random() * 0.15
            high = min(0.98, low + 0.05 + rng.random() * 0.1)
            config = OfflineTestingConfig(
                seed=seed,
                suite="unit",
                test_count=6 + rng.randrange(5),
                failure_rate=min(0.4, rng.random() * 0.25),
                coverage_range=(low * 100, high * 100),
            )
            simulator = OfflineTestingStep(self._context.project_root, config=config)
            result = simulator.run()
        else:
            failure_rate = min(0.5, 0.05 + rng.random() * 0.2)
            simulator = CIStep(
                self._context.project_root,
                name=category,
                pipelines=(category,),
                seed=seed,
                failure_rate=failure_rate,
            )
            result = simulator.run()

        blocks = []
        for artifact in result.artifacts:
            content = artifact.read_text(encoding="utf-8")
            blocks.append(self._format_block(self._relative_path(artifact), content))

        if not blocks:
            summary = {
                "status": result.status,
                "category": category,
                "prompt_hash": prompt_hash,
            }
            blocks.append(
                self._format_block(
                    f".douglas/state/{category}_summary.json",
                    json.dumps(summary, indent=2) + "\n",
                )
            )

        return "\n\n".join(blocks)

    # Generic helpers -------------------------------------------------------
    def _format_block(self, path: str, content: str) -> str:
        return f"```{path}\n{content.rstrip()}\n```"

    def _relative_path(self, path: Path) -> str:
        try:
            relative = path.relative_to(self._context.project_root)
        except ValueError:
            relative = path
        return relative.as_posix()

    def generate_code(self, prompt: str) -> str:
        step = self._context.step_name.lower()
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

