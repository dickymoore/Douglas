import importlib.resources as resources
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

import yaml

from douglas.cadence_manager import CadenceManager
from douglas.controls import run_state as run_state_control
from douglas.integrations.repository import resolve_repository_integration
from douglas.journal import agent_io
from douglas.journal import questions as question_journal
from douglas.pipelines import demo as demopipe
from douglas.pipelines import lint
from douglas.pipelines import plan as planpipe
from douglas.pipelines import standup as standuppipe
from douglas.pipelines import retro as retropipe
from douglas.pipelines import security as securitypipe
from douglas.pipelines import test as testpipe
from douglas.pipelines import typecheck
from douglas.steps import planning as planning_step
from douglas.providers.claude_code_provider import ClaudeCodeProvider
from douglas.providers.codex_provider import CodexProvider
from douglas.providers.copilot_provider import CopilotProvider
from douglas.providers.gemini_provider import GeminiProvider
from douglas.providers.llm_provider import LLMProvider
from douglas.providers.openai_provider import OpenAIProvider
from douglas.providers.provider_registry import (
    LLMProviderRegistry,
    StaticLLMProviderRegistry,
)
from douglas.sprint_manager import CadenceDecision, SprintManager
from douglas.logging_utils import configure_logging, get_logger

TEMPLATE_ROOT = Path(__file__).resolve().parent.parent / "templates"


logger = get_logger(__name__)


_DEFAULT_PROVIDER_MODELS = {
    "codex": CodexProvider.DEFAULT_MODEL,
    "openai": OpenAIProvider.DEFAULT_MODEL,
    "claude_code": ClaudeCodeProvider.DEFAULT_MODEL,
    "claude": ClaudeCodeProvider.DEFAULT_MODEL,
    "gemini": GeminiProvider.DEFAULT_MODEL,
    "copilot": CopilotProvider.DEFAULT_MODEL,
}


@dataclass
class StepExecutionResult:
    executed: bool
    success: bool
    override_event: Optional[str] = None
    already_recorded: bool = False
    failure_reported: bool = False
    failure_details: Optional[str] = None


class DouglasSystemExit(SystemExit):
    """SystemExit variant that carries Douglas-specific failure metadata."""

    def __init__(
        self,
        code: Optional[int] = None,
        *,
        message: Optional[str] = None,
        logs: Optional[str] = None,
        handled: bool = False,
    ) -> None:
        super().__init__(code)
        self.douglas_failure_message: Optional[str] = message
        self.douglas_failure_logs: Optional[str] = logs
        self.douglas_failure_handled: bool = handled


TLS_ERROR_PATTERN = re.compile(
    r"\btls\b[^\n]*(error|failed|failure|handshake|protocol)", re.IGNORECASE
)


class Douglas:
    DEFAULT_COMMIT_MESSAGE = "chore: automated commit"
    DEFAULT_SPRINT_LENGTH_DAYS = SprintManager.DEFAULT_SPRINT_LENGTH_DAYS
    SUPPORTED_PUSH_POLICIES = {
        "per_feature",
        "per_feature_complete",
        "per_bug",
        "per_epic",
        "per_sprint",
    }

    DEFAULT_ITERATION_LIMIT = 1000

    MAX_LOG_EXCERPT_LENGTH = 4000  # Default number of characters retained from the end of CI logs and bug report excerpts.

    DEFAULT_INIT_TEMPLATE: Dict[str, Any] = {
        "project": {
            "name": "PROJECT_NAME",
            "description": "Project description",
            "license": "MIT",
            "language": "python",
        },
        "ai": {
            "default_provider": "codex",
            "mode": "real",
            "seed": 0,
            "replay_dir": ".douglas/cassettes",
            "record_cassettes": False,
            "providers": {
                "codex": {
                    "provider": "codex",
                    "model": CodexProvider.DEFAULT_MODEL,
                }
            },
            "prompt": "system_prompt.md",
        },
        "planning": {
            "enabled": True,
            "sprint_zero_only": False,
            "first_day_only": True,
            "backlog_file": "ai-inbox/backlog/pre-features.yaml",
            "allow_overwrite": False,
            "goal": "Facilitate Sprint Zero by brainstorming epics, features, user stories, and tasks before coding begins.",
            "charters": {
                "enabled": True,
                "directory": "ai-inbox/charters",
                "allow_overwrite": False,
            },
        },
        "cadence": {
            "Developer": {
                "development": "daily",
                "quality_checks": "daily",
                "code_review": "per_feature",
            },
            "Tester": {"test_cases": "daily"},
            "ProductOwner": {
                "backlog_refinement": "per_sprint",
                "sprint_review": "per_sprint",
            },
            "ScrumMaster": {
                "daily_standup": "daily",
                "retrospective": "per_sprint",
            },
            "DevOps": {
                "release": "per_feature",
                "security_checks": "per_feature",
            },
            "Designer": {
                "design_review": "per_sprint",
                "ux_review": "per_feature",
            },
            "BA": {"requirements_analysis": "per_sprint"},
            "Stakeholder": {
                "check_in": "per_sprint",
                "status_update": "on_demand",
            },
        },
        "loop": {
            "steps": [
                {
                    "name": "standup",
                    "role": "ScrumMaster",
                    "activity": "daily_standup",
                    "cadence": "daily",
                },
                {
                    "name": "plan",
                    "role": "ProductOwner",
                    "activity": "backlog_refinement",
                    "cadence": "daily",
                },
                {
                    "name": "generate",
                    "role": "Developer",
                    "activity": "development",
                },
                {
                    "name": "lint",
                    "role": "Developer",
                    "activity": "quality_checks",
                },
                {
                    "name": "typecheck",
                    "role": "Developer",
                    "activity": "quality_checks",
                },
                {
                    "name": "security",
                    "role": "DevOps",
                    "activity": "security_checks",
                },
                {
                    "name": "test",
                    "role": "Tester",
                    "activity": "test_cases",
                },
                {
                    "name": "review",
                    "role": "Developer",
                    "activity": "code_review",
                },
                {
                    "name": "retro",
                    "role": "ScrumMaster",
                    "activity": "retrospective",
                    "cadence": "per_sprint",
                },
                {
                    "name": "demo",
                    "role": "ProductOwner",
                    "activity": "sprint_review",
                    "cadence": "per_sprint",
                },
                {
                    "name": "commit",
                    "role": "Developer",
                    "activity": "development",
                },
                {
                    "name": "push",
                    "role": "DevOps",
                    "activity": "release",
                },
                {
                    "name": "pr",
                    "role": "Developer",
                    "activity": "code_review",
                },
            ],
            "exit_condition_mode": "all",
            "exit_conditions": ["feature_delivery_complete", "sprint_demo_complete"],
            "exhaustive": False,
        },
        "demo": {
            "format": "md",
            "include": [
                "implemented_features",
                "how_to_run",
                "test_results",
                "limitations",
                "next_steps",
            ],
        },
        "retro": {
            "outputs": ["role_instructions", "pre_feature_backlog"],
            "backlog_file": "ai-inbox/backlog/pre-features.yaml",
        },
        "history": {"max_log_excerpt_length": MAX_LOG_EXCERPT_LENGTH},
        "paths": {
            "inbox_dir": "ai-inbox",
            "app_src": "src",
            "tests": "tests",
            "demos_dir": "demos",
            "sprint_prefix": "sprint-",
            "questions_dir": "user-portal/questions",
            "questions_archive_dir": "user-portal/questions-archive",
            "user_portal_dir": "user-portal",
            "run_state_file": "user-portal/run-state.txt",
        },
        "agents": {
            "roles": [
                "Developer",
                "Tester",
                "Product Owner",
                "Scrum Master",
                "Designer",
                "Business Analyst",
                "DevOps",
                "Account Manager",
            ]
        },
        "accountability": {
            "enabled": True,
            "stall_iterations": 3,
            "soft_stop": True,
        },
        "run_state": {"allowed": ["CONTINUE", "SOFT_STOP", "HARD_STOP"]},
        "qna": {"filename_pattern": "sprint-{sprint}-{role}-{id}.md"},
    }

    def __init__(
        self,
        config_path: Union[str, Path, None] = "douglas.yaml",
        *,
        config: Optional[Dict[str, Any]] = None,
        config_data: Optional[Dict[str, Any]] = None,
    ):
        if config is not None and config_data is not None:
            raise ValueError(
                "Only one of 'config' or 'config_data' may be provided when instantiating Douglas."
            )

        if config_path is None:
            resolved_path = Path("douglas.yaml")
        else:
            resolved_path = Path(config_path)

        self.config_path = resolved_path

        source_config: Optional[Dict[str, Any]]
        if config is not None:
            source_config = config
        elif config_data is not None:
            source_config = config_data
        else:
            source_config = None

        if source_config is None:
            self.config = self.load_config(self.config_path)
        else:
            self.config = deepcopy(source_config)
        self.project_root = self.config_path.resolve().parent
        self.project_name = self.config.get("project", {}).get("name", "")

        log_file_override = os.getenv("DOUGLAS_LOG_FILE")
        if log_file_override:
            configure_logging(log_file=log_file_override)
        else:
            default_log_path = self.project_root / "ai-inbox" / "logs" / "douglas.log"
            configure_logging(log_file=str(default_log_path))

        vcs_config: Mapping[str, Any] = {}
        raw_vcs = self.config.get("vcs")
        if isinstance(raw_vcs, Mapping):
            vcs_config = raw_vcs
        vcs_provider_name = (
            vcs_config.get("provider") if isinstance(vcs_config, Mapping) else None
        )
        self._repository_integration = resolve_repository_integration(vcs_provider_name)
        self._repository_provider_name = (
            getattr(self._repository_integration, "name", "github").strip().lower()
        )

        self._llm_registry = None
        self.lm_provider = self.create_llm_provider()
        if getattr(self, "_llm_registry", None) is None:
            self._llm_registry = StaticLLMProviderRegistry(self.lm_provider)
        self.sprint_manager = SprintManager(
            sprint_length_days=self._resolve_sprint_length()
        )
        self.cadence_manager = CadenceManager(
            self.config.get("cadence"), self.sprint_manager
        )
        self.push_policy = self._resolve_push_policy()
        self._max_log_excerpt_length = self._resolve_log_excerpt_length()
        self.history_path = self.project_root / "ai-inbox" / "history.jsonl"
        self._loop_outcomes: Dict[str, Optional[bool]] = {}
        self._ci_status: Optional[str] = None
        self._ci_monitoring_triggered: bool = False
        self._ci_monitoring_deferred: bool = False
        self._configured_steps: set[str] = set()
        self._executed_step_names: set[str] = set()
        self._blocking_questions_by_role: Dict[
            str, List[question_journal.Question]
        ] = {}
        self._run_state_path = self._resolve_run_state_path()
        self._soft_stop_pending = False
        # Flag indicating whether a sprint plan refresh is pending.
        # Set to True in methods such as `update_cadence`, `trigger_sprint_plan_refresh`, or when an external
        # event requires the sprint plan to be refreshed before the next iteration.
        self._pending_sprint_plan_refresh = False
        accountability_cfg = self.config.get("accountability", {}) or {}
        self._accountability_enabled = bool(accountability_cfg.get("enabled", True))
        stall_value = accountability_cfg.get("stall_iterations", 3)
        try:
            self._accountability_stall_threshold = max(1, int(stall_value))
        except (TypeError, ValueError):
            self._accountability_stall_threshold = 3
        self._accountability_soft_stop = bool(accountability_cfg.get("soft_stop", True))
        self._iterations_without_progress = 0

    @staticmethod
    def load_scaffold_config() -> Dict[str, Any]:
        template_candidates: List[Path] = []
        try:
            package_template = (
                resources.files("douglas") / "templates" / "douglas.yaml.tpl"
            )
        except ModuleNotFoundError:
            package_template = None
        else:
            template_candidates.append(package_template)

        template_candidates.append(TEMPLATE_ROOT / "douglas.yaml.tpl")

        template_text = None
        last_missing: Optional[FileNotFoundError] = None
        for candidate in template_candidates:
            try:
                template_text = candidate.read_text(encoding="utf-8")
            except FileNotFoundError as exc:
                last_missing = exc
                continue
            except OSError as exc:
                print(
                    f"Unable to read template file '{candidate}': {exc}. Using fallback defaults."
                )
                return deepcopy(Douglas.DEFAULT_INIT_TEMPLATE)

            if template_text is not None:
                break

        if template_text is None:
            if last_missing is not None:
                missing_location = last_missing.filename or "douglas.yaml.tpl"
            else:
                missing_location = "douglas.yaml.tpl"
            print(
                f"Template file '{missing_location}' not found; using fallback defaults."
            )
            return deepcopy(Douglas.DEFAULT_INIT_TEMPLATE)

        rendered = Template(template_text).safe_substitute(
            PROJECT_NAME="DouglasProject"
        )
        data = yaml.safe_load(rendered) or {}
        if not isinstance(data, dict):
            return deepcopy(Douglas.DEFAULT_INIT_TEMPLATE)
        return data

    def load_config(self, path):
        try:
            with open(path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config {path}: {e}")
            sys.exit(1)

    def create_llm_provider(self):
        ai_config = self.config.get("ai", {}) or {}
        if not isinstance(ai_config, Mapping):
            ai_config = {}

        try:
            registry = LLMProviderRegistry(ai_config, project_root=self.project_root)
        except ValueError as exc:
            print(f"LLM provider configuration invalid: {exc}")
            sys.exit(1)

        self._llm_registry = registry
        default_provider = registry.default_provider
        if default_provider is None:  # pragma: no cover - defensive guard
            print("No default LLM provider configured; exiting.")
            sys.exit(1)
        return default_provider

    def _resolve_llm_provider(self, agent_label: str, step_name: str) -> LLMProvider:
        registry = getattr(self, "_llm_registry", None)
        if registry is None:
            return self.lm_provider
        try:
            provider = registry.resolve(agent_label, step_name)
        except Exception:  # pragma: no cover - defensive fallback
            provider = None
        return provider or self.lm_provider

    def _build_planning_context(
        self,
        planning_config: Mapping[str, Any],
        *,
        backlog_fallback: Optional[Mapping[str, object]] = None,
    ) -> planning_step.PlanningContext:
        ai_config = self.config.get("ai") if hasattr(self.config, "get") else {}
        base_seed = 0
        if isinstance(ai_config, Mapping):
            seed_value = ai_config.get("seed", 0)
            try:
                base_seed = int(seed_value)
            except (TypeError, ValueError):
                base_seed = 0
        raw_items_per_sprint = planning_config.get("items_per_sprint", 3)
        try:
            items_per_sprint = int(raw_items_per_sprint)
        except (TypeError, ValueError):
            items_per_sprint = 3
        if items_per_sprint < 0:
            items_per_sprint = 0
        goal_text = planning_config.get("goal")
        summary_intro = (
            goal_text if isinstance(goal_text, str) and goal_text.strip() else None
        )
        planning_provider = self._resolve_llm_provider("ProductOwner", "planning")
        return planning_step.PlanningContext(
            project_root=self.project_root,
            backlog_state_path=(
                self.project_root / ".douglas" / "state" / "backlog.json"
            ),
            sprint_index=self.sprint_manager.sprint_index,
            items_per_sprint=items_per_sprint,
            seed=base_seed,
            provider=planning_provider,
            summary_intro=summary_intro,
            backlog_fallback=backlog_fallback,
        )

    def _refresh_sprint_plan(
        self, planning_config: Mapping[str, Any]
    ) -> Optional[planning_step.PlanningStepResult]:
        try:
            planning_context = self._build_planning_context(planning_config)
            result = planning_step.run_planning(planning_context)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Sprint planning refresh failed", exc_info=exc)
            return None

        summary = result.summary(self.project_root)
        plan_outcome = self._loop_outcomes.get("plan")
        if isinstance(plan_outcome, dict):
            plan_outcome["sprint_plan"] = summary

        if result.plan is not None and result.success:
            event_payload = {
                "sprint": result.plan.sprint_index,
                "goals": result.plan.goals,
                "commitments": result.plan.commitment_ids,
            }
            json_ref = summary.get("json_path")
            if json_ref:
                event_payload["json_path"] = json_ref
            markdown_ref = summary.get("markdown_path")
            if markdown_ref:
                event_payload["markdown_path"] = markdown_ref
            self._write_history_event("sprint_plan", event_payload)
            if result.plan.commitments:
                print(
                    "Updated sprint plan after backlog initialization "
                    f"with {len(result.plan.commitments)} commitments."
                )

        return result

    def _infer_ai_provider_from_config(
        self, ai_config: Mapping[str, Any]
    ) -> Optional[str]:
        if not isinstance(ai_config, Mapping):
            return None
        candidate = ai_config.get("default_provider")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip().lower()
        candidate = ai_config.get("provider")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip().lower()
        providers_section = ai_config.get("providers")
        if isinstance(providers_section, Mapping) and providers_section:
            first_key = next(iter(providers_section))
            if isinstance(first_key, str) and first_key.strip():
                return first_key.strip().lower()
        return None

    def _infer_ai_model_from_config(
        self, ai_config: Mapping[str, Any], provider: Optional[str]
    ) -> Optional[str]:
        if not isinstance(ai_config, Mapping):
            return None
        normalized = (provider or "").strip().lower()
        providers_section = ai_config.get("providers")
        if isinstance(providers_section, Mapping):
            for key, spec in providers_section.items():
                if not isinstance(spec, Mapping):
                    continue
                key_normalized = str(key).strip().lower()
                provider_name = str(spec.get("provider", "")).strip().lower()
                if key_normalized == normalized or provider_name == normalized:
                    model = spec.get("model") or spec.get("model_name")
                    if isinstance(model, str) and model.strip():
                        return model.strip()
        model = ai_config.get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()
        return None

    def _default_model_for_provider(self, provider: Optional[str]) -> Optional[str]:
        if not provider:
            return None
        return _DEFAULT_PROVIDER_MODELS.get(provider.strip().lower())

    def _resolve_sprint_length(self) -> Optional[int]:
        sprint_config = self.config.get("sprint", {}) or {}
        if "length" in sprint_config:
            logger.warning(
                "'length' is deprecated. Please use 'length_days' in the sprint configuration."
            )
        raw_length = sprint_config.get("length_days")
        if raw_length is None:
            return None
        try:
            length = int(raw_length)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid sprint length '%s'; falling back to default cadence.",
                raw_length,
            )
            return None
        if length <= 0:
            logger.warning(
                "Sprint length must be positive; defaulting to %d days.",
                self.DEFAULT_SPRINT_LENGTH_DAYS,
            )
            return None
        return length

    def _resolve_push_policy(self) -> str:
        candidate = self.config.get("push_policy")
        if not candidate:
            candidate = self.config.get("vcs", {}).get("push_policy")
        if not candidate:
            return "per_feature"

        normalized = str(candidate).strip().lower()
        if normalized not in self.SUPPORTED_PUSH_POLICIES:
            logger.warning(
                "Unsupported push_policy '%s'; defaulting to per_feature.", candidate
            )
            return "per_feature"
        return normalized

    def _resolve_log_excerpt_length(self) -> int:
        history_cfg = self.config.get("history", {}) or {}
        candidate = history_cfg.get("max_log_excerpt_length")

        if candidate is None:
            return self.MAX_LOG_EXCERPT_LENGTH

        try:
            value = int(candidate)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid history.max_log_excerpt_length value '%s'; defaulting to %d.",
                candidate,
                self.MAX_LOG_EXCERPT_LENGTH,
            )
            return self.MAX_LOG_EXCERPT_LENGTH

        if value <= 0:
            logger.warning(
                "history.max_log_excerpt_length must be positive; defaulting to %d.",
                self.MAX_LOG_EXCERPT_LENGTH,
            )
            return self.MAX_LOG_EXCERPT_LENGTH

        return value

    def _resolve_iteration_limit(self) -> int:
        loop_config = self.config.get("loop", {}) or {}

        candidate = loop_config.get("max_iterations")
        if candidate is None:
            candidate = loop_config.get("iterations")

        if candidate is None:
            return self.DEFAULT_ITERATION_LIMIT

        try:
            value = int(candidate)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid loop iteration limit '%s'; defaulting to %d iterations.",
                candidate,
                self.DEFAULT_ITERATION_LIMIT,
            )
            return self.DEFAULT_ITERATION_LIMIT

        if value <= 0:
            logger.warning(
                "loop iteration limit must be positive; defaulting to %d iterations.",
                self.DEFAULT_ITERATION_LIMIT,
            )
            return self.DEFAULT_ITERATION_LIMIT

        return value

    def _resolve_run_state_path(self) -> Path:
        paths_cfg = self.config.get("paths", {}) or {}

        candidate = paths_cfg.get("run_state_file")
        if candidate:
            run_state_path = Path(candidate)
            if not run_state_path.is_absolute():
                run_state_path = self.project_root / run_state_path
            return run_state_path

        portal_dir = paths_cfg.get("user_portal_dir")
        if portal_dir:
            portal_path = Path(portal_dir)
            if not portal_path.is_absolute():
                portal_path = self.project_root / portal_path
            return portal_path / "run-state.txt"

        return self.project_root / "user-portal" / "run-state.txt"

    def run_loop(self):
        print("Starting Douglas AI development loop...")
        print(
            f"Sprint status: {self.sprint_manager.describe_day()} (iteration {self.sprint_manager.current_iteration})"
        )
        print(f"Push/PR policy: {self.push_policy}")

        self._soft_stop_pending = False
        self._check_run_state(phase="loop_start", allow_soft_stop_exit=False)

        error: Optional[BaseException] = None
        try:
            steps = self._normalize_step_configs(
                self.config.get("loop", {}).get("steps", [])
            )
            self._configured_steps = {str(step["name"]).lower() for step in steps}
            commit_step_present = "commit" in self._configured_steps
            self._loop_outcomes = {}
            self._ci_status = None
            self._ci_monitoring_triggered = False
            self._ci_monitoring_deferred = False

            iteration_limit = self._resolve_iteration_limit()
            iteration_index = 0
            last_executed_steps: List[str] = []

            while iteration_index < iteration_limit:
                if self._exit_conditions_met(last_executed_steps):
                    print("Exit condition satisfied; ending loop early.")
                    break

                baseline_completed_features = len(
                    self.sprint_manager.completed_features
                )
                iteration_index += 1
                self._refresh_question_state()

                executed_steps: List[str] = []
                self._executed_step_names = set()
                self._loop_outcomes = {}
                self._ci_status = None
                self._ci_monitoring_triggered = False
                self._ci_monitoring_deferred = False

                for step_config in steps:
                    step_name = step_config["name"]
                    decision = self.cadence_manager.evaluate_step(
                        step_name, step_config
                    )
                    if not decision.should_run:
                        print(f"Skipping {step_name} step: {decision.reason}")
                        self._record_step_outcome(
                            step_name, executed=False, success=False
                        )
                        continue

                    role_for_step = self._resolve_step_role(step_name)
                    if self._should_defer_for_questions(role_for_step, step_name):
                        self._record_step_outcome(
                            step_name, executed=False, success=False
                        )
                        continue

                    try:
                        result = self._execute_step(step_name, step_config, decision)
                    except SystemExit as exc:
                        self._record_step_outcome(
                            step_name, executed=True, success=False
                        )
                        failure_message = getattr(
                            exc,
                            "douglas_failure_message",
                            f"{step_name} step exited with status {exc.code}.",
                        )
                        failure_logs = getattr(exc, "douglas_failure_logs", None)
                        failure_already_handled = getattr(
                            exc, "douglas_failure_handled", False
                        )
                        if not failure_already_handled:
                            self._handle_step_failure(
                                step_name, failure_message, failure_logs
                            )
                        raise
                    except Exception as exc:
                        self._record_step_outcome(
                            step_name, executed=True, success=False
                        )
                        self._handle_step_failure(
                            step_name,
                            f"{step_name} step raised an exception: {exc}",
                            None,
                        )
                        raise

                    if result.executed:
                        event_type = (
                            result.override_event
                            if result.override_event is not None
                            else decision.event_type
                        )
                        if not result.already_recorded:
                            self.sprint_manager.record_step_execution(
                                step_name, event_type
                            )
                        self._executed_step_names.add(step_name.lower())
                        executed_steps.append(step_name)
                        self._record_step_outcome(
                            step_name, executed=True, success=result.success
                        )
                        if not result.success and not result.failure_reported:
                            self._handle_step_failure(
                                step_name,
                                result.failure_details
                                or f"{step_name} step reported failure.",
                                None,
                            )
                    else:
                        self._record_step_outcome(
                            step_name, executed=False, success=False
                        )

                if not commit_step_present:
                    committed, _ = self._commit_if_needed()
                    if committed:
                        self.sprint_manager.record_step_execution("commit", None)

                self.sprint_manager.finish_iteration()
                print("Douglas loop iteration completed.")

                progress = self._iteration_had_progress(
                    executed_steps, baseline_completed_features
                )
                self._accountability_monitor(progress)

                last_executed_steps = executed_steps

                if self._exit_conditions_met(executed_steps):
                    print("Exit condition satisfied; ending loop early.")
                    break

            print("Douglas loop completed.")
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._check_run_state(
                phase="loop_end",
                allow_soft_stop_exit=True,
                enforce_exit=error is None,
            )

    def _normalize_step_configs(self, raw_steps: List[Any]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for entry in raw_steps or []:
            if isinstance(entry, str):
                normalized.append({"name": entry, "cadence": None})
                continue

            if isinstance(entry, dict):
                step_entry: Dict[str, Any] = dict(entry)
                name = step_entry.get("name") or step_entry.get("step")
                if not name and len(step_entry) == 1:
                    sole_key = next(iter(step_entry))
                    value = step_entry[sole_key]
                    if isinstance(value, dict):
                        name = sole_key
                        merged = dict(value)
                        merged["name"] = name
                        step_entry = merged
                if not name:
                    logger.warning("Skipping loop step without a name.")
                    continue
                step_entry["name"] = str(name)
                normalized.append(step_entry)
                continue

            logger.warning("Unsupported loop step configuration '%s'; skipping.", entry)

        return normalized

    def _check_run_state(
        self,
        *,
        phase: str,
        agent_label: Optional[str] = None,
        allow_soft_stop_exit: bool,
        enforce_exit: bool = True,
    ) -> run_state_control.RunState:
        descriptor = phase
        if agent_label:
            descriptor = f"{phase}:{agent_label}"

        state = run_state_control.read_run_state(self._run_state_path)
        print(f"Run state check ({descriptor}): {state.value}")

        previously_pending = self._soft_stop_pending
        if state is run_state_control.RunState.SOFT_STOP and not previously_pending:
            print("Soft stop requested; will finish current sprint before exiting.")
        if state is run_state_control.RunState.SOFT_STOP:
            self._soft_stop_pending = True

        should_exit = run_state_control.should_exit_now(
            state,
            {
                "phase": phase,
                "allow_soft_stop_exit": allow_soft_stop_exit,
                "soft_stop_pending": self._soft_stop_pending,
            },
        )

        if enforce_exit and should_exit:
            if state is run_state_control.RunState.HARD_STOP:
                print("Hard stop requested; aborting immediately.")
                raise SystemExit(1)
            print("Soft stop in effect; exiting after completing sprint.")
            raise SystemExit(0)

        return state

    def _run_agent_with_state(
        self,
        agent_label: str,
        step_name: str,
        action: Callable[[], Any],
    ) -> Any:
        descriptor = f"{agent_label}:{step_name}"
        self._check_run_state(
            phase="agent_start",
            agent_label=descriptor,
            allow_soft_stop_exit=False,
        )

        error: Optional[BaseException] = None
        try:
            return action()
        except BaseException as exc:  # pragma: no cover - propagate to caller
            error = exc
            raise
        finally:
            self._check_run_state(
                phase="agent_end",
                agent_label=descriptor,
                allow_soft_stop_exit=False,
                enforce_exit=error is None,
            )

    def _question_context(self, **overrides: Any) -> Dict[str, Any]:
        context: Dict[str, Any] = {
            "project_root": self.project_root,
            "config": self.config,
            "sprint": self.sprint_manager.sprint_index,
        }
        context.update(overrides)
        return context

    def _refresh_question_state(self) -> None:
        try:
            open_questions = question_journal.scan_for_answers(self._question_context())
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Unable to scan for user questions: %s", exc)
            self._blocking_questions_by_role = {}
            return

        blocking: Dict[str, List[question_journal.Question]] = {}
        for item in open_questions:
            answer = (item.user_answer or "").strip()
            if answer:
                logger.info("Processing user answer for %s: %s", item.id, item.topic)
                item.agent_follow_up = self._format_question_follow_up(item)
                try:
                    question_journal.archive_question(item)
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.warning(
                        "Failed to archive answered question %s: %s",
                        item.id,
                        exc,
                    )
                continue

            if item.blocking:
                key = item.normalized_role()
                blocking.setdefault(key, []).append(item)

        self._blocking_questions_by_role = blocking

    def _format_question_follow_up(self, question: question_journal.Question) -> str:
        timestamp = datetime.now(timezone.utc).isoformat()
        answer = (question.user_answer or "").strip()
        if answer:
            trimmed = answer if len(answer) <= 280 else f"{answer[:277]}..."
            return f"User response recorded on {timestamp}.\n\n{trimmed}"
        return f"User response recorded on {timestamp}."

    def _should_defer_for_questions(self, role: Optional[str], step_name: str) -> bool:
        role_key = self._normalize_role_key(role)
        blocking = list(self._blocking_questions_by_role.get(role_key, []))
        blocking.extend(self._blocking_questions_by_role.get("", []))
        if not blocking:
            return False

        identifiers = ", ".join(question.id for question in blocking)
        print(
            f"Skipping {step_name} step: awaiting user answer for question(s) {identifiers}."
        )
        return True

    def _resolve_step_role(self, step_name: str) -> Optional[str]:
        context = self.cadence_manager.last_context
        if context and context.step_name == step_name:
            return context.role
        return None

    @staticmethod
    def _normalize_role_key(role: Optional[str]) -> str:
        if role is None:
            return ""
        return str(role).strip().lower().replace(" ", "_")

    def _record_agent_summary(
        self,
        role: Optional[str],
        step_name: str,
        summary_text: str,
        details: Optional[Any] = None,
        *,
        handoff_ids: Optional[Iterable[str]] = None,
        title: Optional[str] = None,
    ) -> None:
        if not role:
            return

        sprint_index = getattr(self.sprint_manager, "sprint_index", 1)
        meta: Dict[str, Any] = {
            "project_root": self.project_root,
            "config": self.config,
            "step": step_name,
            "details": details or {},
            "handoff_ids": handoff_ids or [],
        }
        if title:
            meta["title"] = title

        try:
            agent_io.append_summary(role, sprint_index, summary_text, meta)
        except Exception as exc:  # pragma: no cover - logging best effort
            logger.warning(
                "Unable to record summary for role '%s' during %s: %s",
                role,
                step_name,
                exc,
            )

    def _raise_agent_handoff(
        self,
        from_role: Optional[str],
        to_role: Optional[str],
        topic: str,
        context: Any,
        *,
        blocking: bool = True,
    ) -> Optional[str]:
        if not from_role or not to_role:
            return None

        sprint_index = getattr(self.sprint_manager, "sprint_index", 1)
        meta: Dict[str, Any] = {
            "project_root": self.project_root,
            "config": self.config,
        }

        try:
            return agent_io.append_handoff(
                from_role,
                to_role,
                sprint_index,
                topic,
                context,
                blocking,
                meta,
            )
        except Exception as exc:  # pragma: no cover - logging best effort
            logger.warning(
                "Unable to record handoff from '%s' to '%s': %s",
                from_role,
                to_role,
                exc,
            )
            return None

    def _record_step_outcome(
        self, step_name: str, executed: bool, success: bool
    ) -> None:
        if executed:
            self._loop_outcomes[step_name] = bool(success)
        else:
            self._loop_outcomes[step_name] = None

    def _handle_step_failure(
        self,
        step_name: str,
        message: str,
        logs: Optional[str],
    ) -> str:
        summary = f"{step_name} step failed"
        details: list[str] = []
        if message:
            details.append(message.strip())
        if logs:
            details.append(logs.strip())
        combined_details = "\n\n".join(details)
        bug_id = self._create_bug_ticket(
            category=step_name,
            summary=summary,
            details=combined_details or summary,
            log_excerpt=logs,
        )
        self._write_history_event(
            "step_failure",
            {
                "step": step_name,
                "message": message,
                "bug_id": bug_id,
            },
        )

        return bug_id

    def _exit_conditions_met(self, executed_steps: List[str]) -> bool:
        loop_config = self.config.get("loop", {}) or {}
        exit_conditions = list(loop_config.get("exit_conditions") or [])
        mode = str(loop_config.get("exit_condition_mode", "any")).lower()

        if loop_config.get("exhaustive"):
            if "all_work_complete" not in exit_conditions:
                exit_conditions.append("all_work_complete")
            mode = "all"

        if not exit_conditions:
            return False

        executed_lookup = {str(step).strip().lower() for step in executed_steps}

        results: List[bool] = []
        for condition in exit_conditions:
            satisfied = self._evaluate_exit_condition(condition, executed_lookup)
            results.append(satisfied)

        if mode == "all":
            return all(results)

        return any(results)

    def _evaluate_exit_condition(
        self, condition: str, executed_lookup: Set[str]
    ) -> bool:
        normalized = condition.strip().lower()

        if normalized == "sprint_demo_complete":
            return self.sprint_manager.has_step_run("demo")
        if normalized == "tests_pass":
            if "test" not in executed_lookup:
                return False
            return self._loop_outcomes.get("test") is True
        if normalized == "lint_pass":
            if "lint" not in executed_lookup:
                return False
            return self._loop_outcomes.get("lint") is True
        if normalized == "typecheck_pass":
            if "typecheck" not in executed_lookup:
                return False
            return self._loop_outcomes.get("typecheck") is True
        if normalized == "local_checks_pass":
            if "local_checks" not in executed_lookup:
                return False
            return self._loop_outcomes.get("local_checks") is True
        if normalized == "push_complete":
            if "push" not in executed_lookup:
                return False
            return self._loop_outcomes.get("push") is True
        if normalized == "pr_created":
            if "pr" not in executed_lookup:
                return False
            return self._loop_outcomes.get("pr") is True
        if normalized == "ci_pass":
            return self._ci_status == "success"
        if normalized == "feature_delivery_complete":
            return self._feature_delivery_complete()
        if normalized == "all_work_complete":
            return self._all_work_complete()
        if normalized == "all_features_delivered":
            return self._all_features_delivered()
        if normalized == "feature_delivery_goal":
            return self._feature_delivery_goal_met()

        logger.debug("Unknown exit condition '%s' treated as unsatisfied.", condition)
        return False

    def _feature_delivery_complete(self) -> bool:
        if not self.sprint_manager.completed_features:
            return False
        outstanding = self.sprint_manager.outstanding_features()
        if outstanding > 0:
            return False
        return self._loop_outcomes.get("push") is True

    def _all_work_complete(self) -> bool:
        outstanding = self.sprint_manager.outstanding_work_summary()
        if any(value > 0 for value in outstanding.values()):
            return False
        if not self.sprint_manager.has_step_run("demo"):
            return False
        if self._loop_outcomes.get("push") is not True:
            return False
        # Ensure tests and lint/typecheck succeeded on final pass
        return (
            self._loop_outcomes.get("test") is True
            and self._loop_outcomes.get("lint") is True
            and self._loop_outcomes.get("typecheck") is True
        )

    def _all_features_delivered(self) -> bool:
        outstanding = self.sprint_manager.outstanding_features()
        if outstanding > 0:
            return False
        if not self.sprint_manager.completed_features:
            return False
        return self._loop_outcomes.get("push") is True

    def _feature_delivery_goal_met(self) -> bool:
        goal = self._resolve_feature_goal()
        if goal is None:
            return False
        if len(self.sprint_manager.completed_features) < goal:
            return False
        if self.sprint_manager.outstanding_features() > 0:
            return False
        return (
            self._loop_outcomes.get("push") is True
            and self._loop_outcomes.get("test") is True
        )

    def _resolve_feature_goal(self) -> Optional[int]:
        loop_config = self.config.get("loop", {}) or {}
        candidate = loop_config.get("feature_goal")
        if candidate is None:
            return None
        try:
            value = int(candidate)
        except (TypeError, ValueError):
            logger.warning("Invalid feature_goal '%s'; ignoring.", candidate)
            return None
        return value if value > 0 else None

    def _iteration_had_progress(
        self, executed_steps: List[str], baseline_completed_features: int
    ) -> bool:
        executed_lookup = {str(step).strip().lower() for step in executed_steps}
        progress_steps = ("generate", "commit", "push", "pr")
        for name in progress_steps:
            if name in executed_lookup and self._loop_outcomes.get(name) is True:
                return True

        supporting_steps = ("test", "lint", "typecheck", "local_checks")
        for name in supporting_steps:
            if name in executed_lookup and self._loop_outcomes.get(name) is True:
                return True

        if len(self.sprint_manager.completed_features) > baseline_completed_features:
            return True
        return False

    def _accountability_monitor(self, progress_made: bool) -> None:
        if not self._accountability_enabled:
            return
        if progress_made:
            self._iterations_without_progress = 0
            return
        self._iterations_without_progress += 1
        if self._iterations_without_progress >= self._accountability_stall_threshold:
            self._iterations_without_progress = 0
            self._raise_accountability_alert()

    def _raise_accountability_alert(self) -> None:
        sprint_index = self.sprint_manager.sprint_index
        summary = "Detected repeated iterations without meaningful progress. Raising soft stop."
        details = {
            "status": "alert",
            "stall_iterations": self._accountability_stall_threshold,
            "completed_features": len(self.sprint_manager.completed_features),
        }
        self._write_history_event(
            "accountability_alert",
            {
                "sprint": sprint_index,
                "stall_iterations": self._accountability_stall_threshold,
            },
        )
        self._record_agent_summary(
            "AccountManager",
            "monitor",
            summary,
            details,
        )
        if self._accountability_soft_stop:
            print(
                "Account Manager detected sustained lack of progress; requesting soft stop."
            )
            self._soft_stop_pending = True

    def _execute_step(
        self,
        step_name: str,
        step_config: Dict[str, Any],
        cadence_decision: CadenceDecision,
    ) -> StepExecutionResult:
        override_event: Optional[str] = None
        already_recorded = False

        if step_name == "generate":
            print("Running generate step...")
            changes_applied = bool(self.generate())
            if self._pending_sprint_plan_refresh:
                planning_config = self.config.get("planning") or {}
                if planning_config.get("enabled", True):
                    refresh_succeeded = self._refresh_sprint_plan(planning_config)
                    if refresh_succeeded:
                        self._pending_sprint_plan_refresh = False
            return StepExecutionResult(
                True, changes_applied, override_event, already_recorded
            )

        if step_name == "standup":
            print("Running standup step...")
            planning_config = self.config.get("planning") or {}
            backlog_relative = planning_config.get(
                "backlog_file",
                self.config.get("retro", {}).get(
                    "backlog_file", "ai-inbox/backlog/pre-features.yaml"
                ),
            )
            backlog_path = self.project_root / backlog_relative
            questions_dir = self.project_root / self.config.get("paths", {}).get(
                "questions_dir", "user-portal/questions"
            )
            standup_output_dir = (
                self.project_root
                / "ai-inbox"
                / "sprints"
                / f"sprint-{self.sprint_manager.sprint_index}"
                / "standups"
            )

            standup_context = standuppipe.StandupContext(
                project_root=self.project_root,
                sprint_index=self.sprint_manager.sprint_index,
                sprint_day=self.sprint_manager.current_day,
                backlog_path=backlog_path,
                questions_dir=questions_dir,
                output_dir=standup_output_dir,
                planning_config=planning_config,
            )

            standup_result = standuppipe.run_standup(standup_context)
            summary_details = {
                "status": "recorded" if standup_result.wrote_report else "skipped",
                "output": str(
                    standup_result.output_path.relative_to(self.project_root)
                ),
                "stories": standup_result.story_count,
                "blockers": standup_result.blocker_count,
            }
            self._record_agent_summary(
                "ScrumMaster",
                "standup",
                "Documented daily standup snapshot.",
                summary_details,
            )
            self._write_history_event(
                "standup_snapshot",
                {
                    "sprint": self.sprint_manager.sprint_index,
                    "day": self.sprint_manager.current_day,
                    "output": summary_details["output"],
                    "stories": standup_result.story_count,
                    "blockers": standup_result.blocker_count,
                },
            )
            self._loop_outcomes["standup"] = summary_details
            return StepExecutionResult(True, True, None, already_recorded)

        if step_name == "plan":
            print("Running plan step...")
            planning_config = self.config.get("planning") or {}
            if not planning_config.get("enabled", True):
                print("Planning disabled; skipping plan step.")
                self._record_agent_summary(
                    "ProductOwner",
                    "plan",
                    "Planning disabled in configuration.",
                    {"status": "skipped"},
                )
                self._loop_outcomes["plan"] = {"status": "skipped"}
                return StepExecutionResult(False, False, None, already_recorded)

            if (
                planning_config.get("first_day_only", False)
                and self.sprint_manager.current_day > 1
            ):
                print(
                    "Skipping plan step: planning restricted to the first day of each sprint."
                )
                self._record_agent_summary(
                    "ProductOwner",
                    "plan",
                    "Planning limited to sprint day 1; skipping.",
                    {"status": "skipped"},
                )
                self._loop_outcomes["plan"] = {"status": "skipped"}
                return StepExecutionResult(False, False, None, already_recorded)

            if planning_config.get("sprint_zero_only", True) and (
                self.sprint_manager.sprint_index > 1
                or self.sprint_manager.current_day > 1
            ):
                print(
                    "Skipping plan step: planning configured for Sprint Zero only and sprint has progressed."
                )
                self._record_agent_summary(
                    "ProductOwner",
                    "plan",
                    "Sprint Zero planning already completed.",
                    {"status": "skipped"},
                )
                self._loop_outcomes["plan"] = {"status": "skipped"}
                return StepExecutionResult(False, False, None, already_recorded)

            backlog_relative = planning_config.get(
                "backlog_file",
                self.config.get("retro", {}).get(
                    "backlog_file", "ai-inbox/backlog/pre-features.yaml"
                ),
            )
            backlog_path = self.project_root / backlog_relative
            system_prompt_name = self.config.get("ai", {}).get(
                "prompt", "system_prompt.md"
            )
            system_prompt_path = self.project_root / system_prompt_name

            llm = self._resolve_llm_provider("ProductOwner", "plan")

            plan_context = planpipe.PlanContext(
                project_name=self.config.get("project", {}).get(
                    "name", "Unknown Project"
                ),
                project_description=self.config.get("project", {}).get(
                    "description", ""
                ),
                project_root=self.project_root,
                backlog_path=backlog_path,
                system_prompt_path=system_prompt_path,
                sprint_index=self.sprint_manager.sprint_index,
                sprint_day=self.sprint_manager.current_day,
                planning_config=planning_config,
                llm=llm,
                agent_roles=list(self.config.get("agents", {}).get("roles", []) or []),
            )

            plan_result = planpipe.run_plan(plan_context)

            planning_result: Optional[planning_step.PlanningStepResult] = None
            try:
                planning_context = self._build_planning_context(
                    planning_config,
                    backlog_fallback=plan_result.backlog_data
                    if getattr(plan_result, "backlog_data", None)
                    else None,
                )
                planning_result = planning_step.run_planning(planning_context)
            except Exception as exc:
                logger.exception("Sprint planning step failed", exc_info=exc)
                planning_result = planning_step.PlanningStepResult(
                    executed=False,
                    success=False,
                    reason=f"exception:{exc}",
                )

            if plan_result.created_backlog:
                try:
                    backlog_display = str(
                        plan_result.backlog_path.relative_to(self.project_root)
                    )
                except ValueError:
                    backlog_display = str(plan_result.backlog_path)
                message = (
                    "Seeded initial backlog with "
                    f"{plan_result.epic_count()} epics and {plan_result.feature_count()} features."
                )
                summary_details = {
                    "status": "seeded",
                    "backlog_path": backlog_display,
                    "epics": plan_result.epic_count(),
                    "features": plan_result.feature_count(),
                }
                self._write_history_event(
                    "backlog_planning",
                    {
                        "status": "seeded",
                        "backlog_path": backlog_display,
                        "epics": plan_result.epic_count(),
                        "features": plan_result.feature_count(),
                    },
                )
            else:
                reason = _friendly_plan_reason(plan_result.reason)
                message = f"Planning skipped: {reason}."
                summary_details = {"status": "skipped", "reason": reason}

            if plan_result.charter_paths:
                charter_map: Dict[str, str] = {}
                for name, path in plan_result.charter_paths.items():
                    try:
                        relative = path.relative_to(self.project_root)
                    except ValueError:
                        relative = path
                    charter_map[name] = str(relative)
                summary_details["charters"] = charter_map

            if planning_result is not None:
                sprint_plan_summary = planning_result.summary(self.project_root)
                summary_details["sprint_plan"] = sprint_plan_summary
                if planning_result.plan is not None and planning_result.success:
                    event_payload = {
                        "sprint": planning_result.plan.sprint_index,
                        "goals": planning_result.plan.goals,
                        "commitments": planning_result.plan.commitment_ids,
                    }
                    json_ref = sprint_plan_summary.get("json_path")
                    if json_ref:
                        event_payload["json_path"] = json_ref
                    markdown_ref = sprint_plan_summary.get("markdown_path")
                    if markdown_ref:
                        event_payload["markdown_path"] = markdown_ref
                    self._write_history_event("sprint_plan", event_payload)
                    if planning_result.plan.commitments:
                        print(
                            "Selected "
                            f"{len(planning_result.plan.commitments)} backlog items for sprint "
                            f"{planning_result.plan.sprint_index}."
                        )
            self._pending_sprint_plan_refresh = bool(
                planning_result and planning_result.used_fallback
            )

            self._record_agent_summary(
                "ProductOwner",
                "plan",
                message,
                summary_details,
            )
            print(message)
            self._loop_outcomes["plan"] = summary_details
            reason_key = (plan_result.reason or "").strip().lower()
            plan_config_raw = (
                self.config.get("plan") if hasattr(self.config, "get") else {}
            )
            plan_config = plan_config_raw or {}
            if not isinstance(plan_config, Mapping):
                plan_config = {}

            configured_prefixes = plan_config.get("skip_reason_prefixes")
            if isinstance(configured_prefixes, str):
                configured_prefixes = [configured_prefixes]
            if isinstance(configured_prefixes, (list, tuple, set)):
                skip_reason_prefixes = tuple(
                    str(prefix).strip().lower()
                    for prefix in configured_prefixes
                    if prefix is not None and str(prefix).strip()
                )
            else:
                skip_reason_prefixes = ()
            if not skip_reason_prefixes:
                skip_reason_prefixes = ("no_llm", "existing_backlog", "skipped")

            empty_reason_setting = plan_config.get("empty_reason_is_skip", True)
            empty_reason_is_skip = bool(empty_reason_setting)

            is_skip = False
            if not plan_result.created_backlog:
                if not reason_key and empty_reason_is_skip:
                    is_skip = True
                else:
                    is_skip = any(
                        reason_key.startswith(prefix) for prefix in skip_reason_prefixes
                    )
            executed = plan_result.created_backlog or not is_skip
            success = plan_result.created_backlog or is_skip
            failure_details = None if success else message
            if planning_result is not None:
                if planning_result.executed:
                    executed = True
                if not planning_result.success:
                    success = False
                    if planning_result.reason:
                        failure_details = planning_result.reason
            return StepExecutionResult(
                executed,
                success,
                None,
                already_recorded,
                failure_details=failure_details,
            )

        if step_name == "lint":
            print("Running lint step...")
            try:
                lint.run_lint()
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    print("Lint step failed; aborting remaining steps.")
                    exit_code = exc.code if exc.code is not None else 1
                    self._record_agent_summary(
                        "Developer",
                        "lint",
                        f"Lint checks failed with exit code {exit_code}.",
                        {"status": "failed", "exit_code": exit_code},
                    )
                raise
            self._record_agent_summary(
                "Developer",
                "lint",
                "Executed configured lint checks successfully.",
                {"status": "passed"},
            )
            return StepExecutionResult(True, True, override_event, already_recorded)

        if step_name == "typecheck":
            print("Running typecheck step...")
            try:
                typecheck.run_typecheck()
            except SystemExit as exc:
                exit_code = exc.code if exc.code is not None else 1
                self._record_agent_summary(
                    "Developer",
                    "typecheck",
                    f"Type checks failed with exit code {exit_code}.",
                    {"status": "failed", "exit_code": exit_code},
                )
                raise
            self._record_agent_summary(
                "Developer",
                "typecheck",
                "Static type checks completed successfully.",
                {"status": "passed"},
            )
            return StepExecutionResult(True, True, override_event, already_recorded)

        if step_name == "test":
            print("Running test step...")
            try:
                testpipe.run_tests()
            except SystemExit as exc:
                exit_code = exc.code if exc.code is not None else 1
                message = f"Test suite failed with exit code {exit_code}."
                context = (
                    f"pytest -q exited with status {exit_code}. "
                    "Review failing tests and coordinate with development."
                )
                handoff_id = self._raise_agent_handoff(
                    "Tester",
                    "Developer",
                    "Investigate failing tests",
                    context,
                    blocking=True,
                )
                self._record_agent_summary(
                    "Tester",
                    "test",
                    message,
                    {"status": "failed", "exit_code": exit_code},
                    handoff_ids=[handoff_id] if handoff_id else None,
                )
                raise
            self._record_agent_summary(
                "Tester",
                "test",
                "Executed automated tests using pytest.",
                {"status": "passed"},
            )
            return StepExecutionResult(True, True, override_event, already_recorded)

        if step_name == "security":
            print("Running security step...")
            tools_config = (
                step_config.get("tools") if isinstance(step_config, dict) else None
            )
            default_paths = None
            if isinstance(step_config, dict):
                default_paths = (
                    step_config.get("paths")
                    or step_config.get("targets")
                    or step_config.get("directories")
                )

            try:
                report = securitypipe.run_security(
                    tools=tools_config,
                    default_paths=default_paths,
                )
            except securitypipe.SecurityCheckError as exc:
                exit_code = exc.exit_code or 1
                command_display = " ".join(exc.command)
                logs = self._format_command_output(
                    command_display,
                    exc.stdout,
                    exc.stderr,
                    exc.exit_code,
                )
                message = (
                    f"Security check '{exc.tool}' failed with exit code {exit_code}."
                )
                bug_id = self._handle_step_failure("security", message, logs)
                summary_details: Dict[str, Any] = {
                    "status": "failed",
                    "tool": exc.tool,
                    "exit_code": exit_code,
                    "command": exc.command,
                }
                if bug_id:
                    summary_details["bug_id"] = bug_id
                self._record_agent_summary(
                    "Security",
                    "security",
                    message,
                    summary_details,
                )
                sys_exit = DouglasSystemExit(
                    exit_code,
                    message=message,
                    logs=logs,
                    handled=True,
                )
                raise sys_exit from exc

            tool_names = report.tool_names()
            history_tools: List[Dict[str, Any]] = []
            for result in report.results:
                entry: Dict[str, Any] = {
                    "name": result.name,
                    "command": result.command,
                    "exit_code": result.exit_code,
                }
                stdout_excerpt = self._tail_log_excerpt(result.stdout)
                stderr_excerpt = self._tail_log_excerpt(result.stderr)
                if stdout_excerpt:
                    entry["stdout"] = stdout_excerpt
                if stderr_excerpt:
                    entry["stderr"] = stderr_excerpt
                history_tools.append(entry)

            self._write_history_event(
                "security_checks_passed",
                {"tools": history_tools},
            )

            summary_details = {
                "status": "passed",
                "tools": tool_names,
            }
            if history_tools:
                summary_details["commands"] = [
                    entry.get("command") for entry in history_tools
                ]

            if tool_names:
                summary_text = "Completed security checks with {}.".format(
                    ", ".join(tool_names)
                )
            else:
                summary_text = "Completed configured security checks."

            self._record_agent_summary(
                "Security",
                "security",
                summary_text,
                summary_details,
            )
            return StepExecutionResult(True, True, override_event, already_recorded)

        if step_name == "review":
            print("Running review step...")
            self.review()
            return StepExecutionResult(True, True, override_event, already_recorded)

        if step_name == "retro":
            print("Running retro step...")
            retro_context = {
                "project_root": self.project_root,
                "config": self.config,
                "sprint_manager": self.sprint_manager,
                "llm": self._resolve_llm_provider("ScrumMaster", "retro"),
                "loop_outcomes": dict(self._loop_outcomes),
            }
            try:
                retro_result = retropipe.run_retro(retro_context)
            except Exception as exc:
                message = f"Retro step failed: {exc}"
                print(message)
                self._record_agent_summary(
                    "ScrumMaster",
                    "retro",
                    message,
                    {"status": "failed", "error": str(exc)},
                )
                return StepExecutionResult(
                    True,
                    False,
                    None,
                    already_recorded,
                    failure_details=message,
                )

            self._write_history_event(
                "retro_completed",
                {
                    "sprint": retro_result.sprint_folder,
                    "generated_at": retro_result.generated_at,
                    "instructions": {
                        role: str(path)
                        for role, path in retro_result.instructions.items()
                    },
                    "backlog_entries": [
                        entry.get("id") for entry in retro_result.backlog_entries
                    ],
                },
            )
            details = {
                "status": "completed",
                "instructions_generated": len(retro_result.instructions or {}),
                "backlog_entries": len(retro_result.backlog_entries or []),
            }
            self._record_agent_summary(
                "ScrumMaster",
                "retro",
                "Completed sprint retrospective and published follow-up actions.",
                details,
            )
            return StepExecutionResult(True, True, override_event, already_recorded)

        if step_name == "demo":
            print("Running demo step...")
            demo_context = {
                "project_root": self.project_root,
                "config": self.config,
                "sprint_manager": self.sprint_manager,
                "history_path": self.history_path,
                "loop_outcomes": dict(self._loop_outcomes),
            }
            try:
                metadata = demopipe.write_demo_pack(demo_context)
            except Exception as exc:
                message = f"Demo generation failed: {exc}"
                print(message)
                self._record_agent_summary(
                    "ProductOwner",
                    "demo",
                    message,
                    {"status": "failed", "error": str(exc)},
                )
                return StepExecutionResult(
                    True,
                    False,
                    None,
                    already_recorded,
                    failure_details=message,
                )

            self._write_history_event(
                "demo_pack_generated", metadata.as_event_payload()
            )
            try:
                relative_path = metadata.output_path.relative_to(self.project_root)
                output_display = str(relative_path)
            except ValueError:
                output_display = str(metadata.output_path)
            details = {
                "status": "completed",
                "output": output_display,
                "format": metadata.format,
                "sprint_folder": metadata.sprint_folder,
            }
            self._record_agent_summary(
                "ProductOwner",
                "demo",
                "Generated sprint demo pack for the current sprint.",
                details,
            )
            return StepExecutionResult(True, True, override_event, already_recorded)

        if step_name == "commit":
            print("Running commit step...")
            committed, commit_message = self._commit_if_needed()
            if committed:
                override_event = cadence_decision.event_type
                summary = (
                    f"Committed changes with message: {commit_message}"
                    if commit_message
                    else "Committed staged changes."
                )
                details = {"status": "committed"}
                if commit_message:
                    details["message"] = commit_message
                self._record_agent_summary("Developer", "commit", summary, details)
            else:
                self._record_agent_summary(
                    "Developer",
                    "commit",
                    "No staged changes available for committing.",
                    {"status": "skipped"},
                )
            return StepExecutionResult(
                True, committed, override_event, already_recorded
            )

        if step_name == "push":
            print("Running push step...")
            if self._release_blocked_by_pending_demo():
                print("Skipping push step: awaiting sprint demo before release.")
                self._loop_outcomes["local_checks"] = None
                return StepExecutionResult(False, False, None, already_recorded)
            decision = self.sprint_manager.should_run_push(self.push_policy)
            if not decision.should_run:
                print(f"Skipping push step: {decision.reason}")
                self._loop_outcomes["local_checks"] = None
                return StepExecutionResult(False, False, None, already_recorded)
            if not self._commits_ready_for_push(decision.event_type):
                print("No commits ready for push; skipping push step.")
                self._loop_outcomes["local_checks"] = None
                return StepExecutionResult(False, False, None, already_recorded)

            local_checks_ok, local_logs = self._run_local_checks()
            if not local_checks_ok:
                print("Local checks failed; aborting push.")
                self._handle_step_failure(
                    "local_checks", "Local checks failed before push.", local_logs
                )
                handoff_context = self._build_local_check_handoff_context(local_logs)
                handoff_id = self._raise_agent_handoff(
                    "DevOps",
                    "Developer",
                    "Resolve local guard check failures",
                    handoff_context,
                    blocking=True,
                )
                self._record_agent_summary(
                    "DevOps",
                    "push",
                    "Push blocked because required local checks failed.",
                    {"status": "failed", "reason": "local_checks"},
                    handoff_ids=[handoff_id] if handoff_id else None,
                )
                self._ci_monitoring_deferred = False
                return StepExecutionResult(
                    True,
                    False,
                    None,
                    already_recorded,
                    failure_reported=True,
                    failure_details="Local checks failed before push.",
                )

            success, push_logs = self._run_git_push()
            if success:
                print("Push completed according to policy.")
                self.sprint_manager.record_push(decision.event_type, self.push_policy)
                already_recorded = True
                self._write_history_event(
                    "push",
                    {
                        "policy": self.push_policy,
                        "event_type": decision.event_type,
                        "details": push_logs,
                    },
                )
                self._record_agent_summary(
                    "DevOps",
                    "push",
                    "Pushed committed changes to the remote repository.",
                    {
                        "status": "completed",
                        "policy": self.push_policy,
                        "event_type": decision.event_type,
                    },
                )
                if "pr" not in self._configured_steps:
                    self._monitor_ci(source_step="push")
                    self._ci_monitoring_deferred = False
                else:
                    print("Deferring CI monitoring until PR step completes.")
                    self._ci_monitoring_deferred = True
                return StepExecutionResult(
                    True, True, decision.event_type, already_recorded
                )

            print("Push step failed; leaving commits local.")
            self._handle_step_failure("push", "git push failed.", push_logs)
            self._record_agent_summary(
                "DevOps",
                "push",
                "Push to remote failed; commits remain local.",
                {
                    "status": "failed",
                    "reason": "push",
                    "policy": self.push_policy,
                },
            )
            self._ci_monitoring_deferred = False
            return StepExecutionResult(
                True,
                False,
                None,
                already_recorded,
                failure_reported=True,
                failure_details="Push failed; see logs for details.",
            )

        if step_name == "pr":
            print("Running pr step...")
            if self._release_blocked_by_pending_demo():
                print("Skipping pr step: awaiting sprint demo before opening PR.")
                return StepExecutionResult(False, False, None, already_recorded)
            decision = self.sprint_manager.should_open_pr(self.push_policy)
            if not decision.should_run:
                print(f"Skipping pr step: {decision.reason}")
                if self._ci_monitoring_deferred:
                    self._monitor_ci(source_step="push")
                    self._ci_monitoring_deferred = False
                return StepExecutionResult(False, False, None, already_recorded)
            if not self._commits_ready_for_pr(decision.event_type):
                print("No commits ready for PR; skipping pr step.")
                if self._ci_monitoring_deferred:
                    self._monitor_ci(source_step="push")
                    self._ci_monitoring_deferred = False
                return StepExecutionResult(False, False, None, already_recorded)

            pr_created, pr_metadata = self._open_pull_request()
            if pr_created:
                print("Pull request created according to policy.")
                self.sprint_manager.record_pr(decision.event_type, self.push_policy)
                already_recorded = True
                self._write_history_event(
                    "pr_created",
                    {
                        "event_type": decision.event_type,
                        "policy": self.push_policy,
                        "metadata": pr_metadata,
                    },
                )
                self._monitor_ci(source_step="pr")
                self._ci_monitoring_deferred = False
                self._record_agent_summary(
                    "Developer",
                    "pr",
                    "Opened a pull request following the configured policy.",
                    {
                        "status": "completed",
                        "policy": self.push_policy,
                        "event_type": decision.event_type,
                    },
                )
                return StepExecutionResult(
                    True, True, decision.event_type, already_recorded
                )

            print("Failed to create pull request; leaving for manual follow-up.")
            self._handle_step_failure(
                "pr", "Pull request creation failed.", pr_metadata
            )
            if self._ci_monitoring_deferred:
                self._monitor_ci(source_step="push")
                self._ci_monitoring_deferred = False
            self._record_agent_summary(
                "Developer",
                "pr",
                "Attempt to open a pull request failed and needs follow-up.",
                {
                    "status": "failed",
                    "policy": self.push_policy,
                    "event_type": decision.event_type,
                },
            )
            return StepExecutionResult(
                True,
                False,
                None,
                already_recorded,
                failure_reported=True,
                failure_details="Failed to create pull request.",
            )

        print(
            f"Step '{step_name}' is not automated; remember to handle it manually when prompted."
        )
        return StepExecutionResult(
            True, True, cadence_decision.event_type, already_recorded
        )

    def _release_blocked_by_pending_demo(self) -> bool:
        """Return True when release actions must wait for the demo step."""
        return (
            self.push_policy == "per_sprint"
            and "demo" in self._configured_steps
            and "demo" not in self._executed_step_names
        )

    def _commits_ready_for_push(self, event_type: Optional[str]) -> bool:
        if event_type in {"feature", "bug", "epic"}:
            return True
        return self.sprint_manager.commits_since_last_push > 0

    def _commits_ready_for_pr(self, event_type: Optional[str]) -> bool:
        if event_type in {"feature", "bug", "epic"}:
            return True
        return self.sprint_manager.commits_since_last_pr > 0

    def _discover_local_check_commands(self) -> List[List[str]]:
        commands: list[list[str]] = []
        ci_config = self.config.get("ci", {}) or {}

        def collect(value: Any) -> None:
            if value is None:
                return
            if isinstance(value, str):
                parsed = shlex.split(value)
                if parsed:
                    commands.append(parsed)
                return
            if isinstance(value, dict):
                for item in value.values():
                    collect(item)
                return
            if isinstance(value, Iterable) and not isinstance(
                value, (bytes, bytearray, str)
            ):
                for item in value:
                    collect(item)
                return
            command = [str(value)]
            commands.append(command)

        for key in ("additional_local_checks", "local_checks"):
            collect(ci_config.get(key))

        existing = {tuple(cmd) for cmd in commands}
        default_candidates = [
            ("black", ["black", "--check", "."]),
            ("bandit", ["bandit", "-r", str(self.project_root)]),
            ("semgrep", ["semgrep", "--config", "auto"]),
        ]

        for tool, command in default_candidates:
            if shutil.which(tool) and tuple(command) not in existing:
                existing.add(tuple(command))
                commands.append(command)

        return commands

    def _run_local_checks(self) -> Tuple[bool, str]:
        commands = self._discover_local_check_commands()
        if not commands:
            self._loop_outcomes["local_checks"] = True
            self._write_history_event("local_checks_pass", {"commands": []})
            return True, "No additional local checks configured."

        logs: list[str] = []
        for command in commands:
            display = " ".join(command)
            print(f"Running local check: {display}")
            try:
                result = subprocess.run(
                    command,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError as exc:
                message = f"Local check command '{command[0]}' not found: {exc}"
                logs.append(message)
                self._loop_outcomes["local_checks"] = False
                self._write_history_event(
                    "local_checks_fail",
                    {
                        "command": display,
                        "error": str(exc),
                    },
                )
                return False, "\n\n".join(logs)

            logs.append(
                self._format_command_output(
                    display, result.stdout, result.stderr, result.returncode
                )
            )

            if result.returncode != 0:
                if self._should_ignore_semgrep_failure(command, result):
                    skip_message = "Semgrep command failed due to environment issues; skipping default check."
                    print(skip_message)
                    logs.append(skip_message)
                    self._write_history_event(
                        "local_checks_skip",
                        {
                            "command": display,
                            "reason": "semgrep_environment_error",
                            "returncode": result.returncode,
                            "stdout_excerpt": self._tail_log_excerpt(result.stdout),
                            "stderr_excerpt": self._tail_log_excerpt(result.stderr),
                        },
                    )
                    continue
                self._loop_outcomes["local_checks"] = False
                self._write_history_event(
                    "local_checks_fail",
                    {
                        "command": display,
                        "returncode": result.returncode,
                        "stdout_excerpt": self._tail_log_excerpt(result.stdout),
                        "stderr_excerpt": self._tail_log_excerpt(result.stderr),
                    },
                )
                return False, "\n\n".join(logs)

        self._loop_outcomes["local_checks"] = True
        self._write_history_event(
            "local_checks_pass",
            {
                "commands": [" ".join(command) for command in commands],
            },
        )
        return True, "\n\n".join(logs)

    def _should_ignore_semgrep_failure(
        self,
        command: List[str],
        result: subprocess.CompletedProcess[str],
    ) -> bool:
        if not command or command[0] != "semgrep":
            return False
        if result.returncode in (0, 1):
            return False

        combined_output = "\n".join(
            part for part in (result.stdout or "", result.stderr or "") if part
        )
        if not combined_output:
            return False

        combined_output_lower = combined_output.lower()
        network_markers = [
            "connectionerror",
            "connection error",
            "cannot connect",
            "failed to establish a new connection",
            "network is unreachable",
            "temporary failure in name resolution",
            "proxyerror",
            "max retries exceeded",
            "check your internet connection",
            "offline",
            "ssl error",
            "certificate verify failed",
            "timeout",
            "timed out",
            "http error",
            "forbidden",
            "unauthorized",
        ]
        if any(marker in combined_output_lower for marker in network_markers):
            return True

        if TLS_ERROR_PATTERN.search(combined_output):
            return True

        credential_markers = [
            "semgrep login",
            "semgrep_app_token",
            "set an app token",
            "set the semgrep app token",
        ]
        return any(marker in combined_output_lower for marker in credential_markers)

    def _format_command_output(
        self,
        command_display: str,
        stdout: Optional[str],
        stderr: Optional[str],
        returncode: Optional[int],
    ) -> str:
        parts = [f"$ {command_display}"]
        if stdout:
            stripped = stdout.strip()
            if stripped:
                parts.append(stripped)
        if stderr:
            stripped_err = stderr.strip()
            if stripped_err:
                parts.append(stripped_err)
        parts.append(f"(exit code {returncode if returncode is not None else 0})")
        return "\n".join(parts)

    def _tail_log_excerpt(
        self, logs: Optional[str], limit: int = 1200
    ) -> Optional[str]:
        if not logs:
            return None
        snippet = logs.strip()
        if not snippet:
            return None
        max_len = min(limit, self._max_log_excerpt_length)
        if len(snippet) <= max_len:
            return snippet
        return snippet[-max_len:]

    def _build_local_check_handoff_context(self, logs: Optional[str]) -> str:
        lines = [
            "Local guard checks failed prior to push. Please resolve the reported issues before retrying the release.",
        ]
        excerpt = self._tail_log_excerpt(logs)
        if excerpt:
            lines.extend(["", "```", excerpt, "```"])
        return "\n".join(lines)

    def _run_git_push(self) -> Tuple[bool, str]:
        remote = self.config.get("vcs", {}).get("remote", "origin")
        branch = (
            self.config.get("vcs", {}).get("current_branch")
            or self._get_current_branch()
        )
        command = ["git", "push", "-u", remote, branch]
        logs: list[str] = []

        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            message = f"git not available to push changes: {exc}"
            logger.warning("%s", message)
            return False, message

        logs.append(
            self._format_command_output(
                " ".join(command), result.stdout, result.stderr, result.returncode
            )
        )

        if result.returncode == 0:
            return True, "\n\n".join(logs)

        error_msg = result.stderr.strip() or result.stdout.strip()
        if error_msg:
            logger.warning("git push failed: %s", error_msg)
        else:
            logger.warning("git push failed without diagnostics.")

        lowered_error = (error_msg or "").lower()
        if "rejected" in lowered_error or "non-fast-forward" in lowered_error:
            logger.info("Push rejected; attempting fast-forward pull.")
            pull_cmd = ["git", "pull", "--ff-only", remote, branch]
            try:
                pull_result = subprocess.run(
                    pull_cmd,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError as exc:
                message = f"git not available to pull changes: {exc}"
                logger.warning("%s", message)
                logs.append(message)
                return False, "\n\n".join(logs)

            logs.append(
                self._format_command_output(
                    " ".join(pull_cmd),
                    pull_result.stdout,
                    pull_result.stderr,
                    pull_result.returncode,
                )
            )

            if pull_result.returncode == 0:
                try:
                    retry_result = subprocess.run(
                        command,
                        cwd=self.project_root,
                        capture_output=True,
                        text=True,
                    )
                except FileNotFoundError as exc:
                    message = f"git not available to retry push: {exc}"
                    logger.warning("%s", message)
                    logs.append(message)
                    return False, "\n\n".join(logs)

                logs.append(
                    self._format_command_output(
                        " ".join(command),
                        retry_result.stdout,
                        retry_result.stderr,
                        retry_result.returncode,
                    )
                )

                if retry_result.returncode == 0:
                    return True, "\n\n".join(logs)

                retry_error = retry_result.stderr.strip() or retry_result.stdout.strip()
                if retry_error:
                    logger.warning("git push retry failed: %s", retry_error)

        return False, "\n\n".join(logs)

    def _open_pull_request(self) -> Tuple[bool, Optional[str]]:
        title, body = self._build_pr_content()
        base_branch = self.config.get("vcs", {}).get("default_branch", "main")
        head_branch = self._get_current_branch()
        integration = getattr(self, "_repository_integration", None)
        if integration is None:
            integration = resolve_repository_integration(None)
            self._repository_integration = integration
            self._repository_provider_name = (
                getattr(integration, "name", "github").strip().lower()
            )

        try:
            metadata = integration.create_pull_request(
                title=title,
                body=body,
                base=base_branch,
                head=head_branch,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Unable to create pull request: %s", exc)
            return False, str(exc)
        return True, metadata

    def _build_pr_content(self) -> Tuple[str, str]:
        subject, commit_body = self._get_latest_commit_summary()
        if not subject:
            subject = f"Sprint update for {self.project_name or 'project'}"

        body_lines = [
            f"Automated changes for {self.sprint_manager.describe_day()} (policy: {self.push_policy}).",
        ]
        if commit_body:
            body_lines.append("\n## Latest commit details\n" + commit_body.strip())

        return subject, "\n\n".join(body_lines)

    def _get_latest_commit_summary(self) -> Tuple[str, str]:
        try:
            subject = subprocess.check_output(
                ["git", "log", "-1", "--pretty=%s"],
                cwd=self.project_root,
                text=True,
            ).strip()
            body = subprocess.check_output(
                ["git", "log", "-1", "--pretty=%b"],
                cwd=self.project_root,
                text=True,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            logger.warning("Unable to collect latest commit summary: %s", exc)
            return "", ""
        return subject, body

    def _get_current_branch(self) -> str:
        configured = self.config.get("vcs", {}).get("current_branch")
        if configured:
            return str(configured)
        try:
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.project_root,
                text=True,
            ).strip()
            return branch or "HEAD"
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "HEAD"

    def _get_current_commit(self) -> Optional[str]:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                text=True,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def _monitor_ci(
        self,
        source_step: str = "pr",
        max_attempts: int = 10,
        poll_interval: int = 10,
    ) -> Optional[bool]:
        source_label = (source_step or "pr").strip().lower() or "pr"

        if self._ci_monitoring_triggered:
            print(
                "CI monitoring already handled for this iteration; skipping duplicate check."
            )
            if self._ci_status == "success":
                return True
            if self._ci_status == "failure":
                return False
            return None

        self._ci_monitoring_triggered = True

        if not getattr(self._repository_integration, "supports_ci_monitoring", False):
            print(
                "CI monitoring is only implemented for GitHub repositories at this time."
            )
            self._ci_status = None
            self._record_agent_summary(
                "DevOps",
                "ci",
                "CI monitoring skipped because the configured repository provider "
                "does not support automated checks.",
                {
                    "status": "skipped",
                    "reason": "unsupported_provider",
                    "provider": self._repository_provider_name,
                    "source_step": source_label,
                },
            )
            return None

        if shutil.which("gh") is None:
            print("GitHub CLI not available; skipping CI monitoring.")
            self._ci_status = None
            self._record_agent_summary(
                "DevOps",
                "ci",
                "CI monitoring skipped because the GitHub CLI is unavailable.",
                {
                    "status": "skipped",
                    "reason": "missing_cli",
                    "source_step": source_label,
                },
            )
            return None

        commit_sha = self._get_current_commit()
        if not commit_sha:
            print("Unable to determine latest commit SHA for CI monitoring.")
            self._ci_status = None
            self._record_agent_summary(
                "DevOps",
                "ci",
                "CI monitoring skipped because the current commit could not be determined.",
                {
                    "status": "skipped",
                    "reason": "unknown_commit",
                    "source_step": source_label,
                },
            )
            return None

        branch = self._get_current_branch()
        for attempt in range(max_attempts):
            try:
                result = subprocess.run(
                    [
                        "gh",
                        "run",
                        "list",
                        "--limit",
                        "20",
                        "--json",
                        "databaseId,headSha,status,conclusion,url",
                        "--branch",
                        branch,
                    ],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError as exc:
                logger.warning("GitHub CLI not available for CI monitoring: %s", exc)
                self._ci_status = None
                self._record_agent_summary(
                    "DevOps",
                    "ci",
                    "CI monitoring aborted because the GitHub CLI could not be executed.",
                    {
                        "status": "skipped",
                        "reason": "missing_cli",
                        "source_step": source_label,
                    },
                )
                return None

            if result.returncode != 0:
                message = (
                    result.stderr.strip() or result.stdout.strip() or "unknown error"
                )
                logger.warning("Unable to list GitHub runs: %s", message)
                time.sleep(poll_interval)
                continue

            try:
                runs = json.loads(result.stdout or "[]")
            except json.JSONDecodeError as exc:
                logger.warning("Unable to parse GitHub run list: %s", exc)
                time.sleep(poll_interval)
                continue

            target_run = next(
                (run for run in runs if run.get("headSha") == commit_sha), None
            )
            if not target_run:
                time.sleep(poll_interval)
                continue

            status = target_run.get("status")
            conclusion = target_run.get("conclusion")
            run_id = target_run.get("databaseId")
            run_url = target_run.get("url")

            if status != "completed":
                logger.info(
                    "Waiting for CI run %s to complete (status: %s).",
                    run_id,
                    status,
                )
                time.sleep(poll_interval)
                continue

            if conclusion == "success":
                logger.info("CI checks succeeded for the latest commit.")
                self._ci_status = "success"
                self._write_history_event(
                    "ci_pass",
                    {
                        "run_id": run_id,
                        "url": run_url,
                        "commit": commit_sha,
                    },
                )
                self._record_agent_summary(
                    "DevOps",
                    "ci",
                    "CI checks succeeded for the latest release.",
                    {
                        "status": "success",
                        "run_id": run_id,
                        "url": run_url,
                        "commit": commit_sha,
                        "source_step": source_label,
                    },
                )
                return True

            summary = f"CI run {run_id} failed with conclusion {conclusion}."
            log_path = self._download_ci_logs(run_id)
            excerpt = None
            log_display = None
            if log_path and log_path.exists():
                try:
                    content = log_path.read_text(encoding="utf-8")
                    excerpt = content[-self._max_log_excerpt_length :]
                except OSError as exc:
                    excerpt = f"Unable to read CI log file: {exc}"
                try:
                    log_display = str(log_path.relative_to(self.project_root))
                except ValueError:
                    log_display = str(log_path)

            self._ci_status = "failure"
            self._write_history_event(
                "ci_fail",
                {
                    "run_id": run_id,
                    "url": run_url,
                    "commit": commit_sha,
                    "conclusion": conclusion,
                },
            )
            bug_id = self._handle_step_failure("ci", summary, excerpt)
            handoff_context = {
                "run_id": run_id,
                "run_url": run_url,
                "conclusion": conclusion,
                "commit": commit_sha,
                "source_step": source_label,
                "bug_id": bug_id,
            }
            if log_display:
                handoff_context["log_path"] = log_display
            handoff_id = self._raise_agent_handoff(
                "DevOps",
                "Developer",
                "Investigate failing CI run",
                handoff_context,
                blocking=True,
            )
            summary_details = {
                "status": "failed",
                "run_id": run_id,
                "url": run_url,
                "conclusion": conclusion,
                "commit": commit_sha,
                "source_step": source_label,
            }
            if bug_id:
                summary_details["bug_id"] = bug_id
            if log_display:
                summary_details["log_path"] = log_display
            self._record_agent_summary(
                "DevOps",
                "ci",
                "CI checks failed for the latest release and need attention.",
                summary_details,
                handoff_ids=[handoff_id] if handoff_id else None,
            )
            return False

        logger.info(
            "CI run not found or did not complete within the monitoring window."
        )
        self._ci_status = None
        self._record_agent_summary(
            "DevOps",
            "ci",
            "CI monitoring timed out before any run completed.",
            {
                "status": "pending",
                "reason": "not_found",
                "source_step": source_label,
            },
        )
        return None

    def _download_ci_logs(self, run_id: Optional[int]) -> Optional[Path]:
        if not getattr(self._repository_integration, "supports_ci_monitoring", False):
            return None
        if not run_id:
            return None
        if shutil.which("gh") is None:
            logger.warning("GitHub CLI not available; cannot download CI logs.")
            return None

        log_dir = self.project_root / "ai-inbox" / "ci"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"ci-run-{run_id}.log"

        try:
            result = subprocess.run(
                ["gh", "run", "view", str(run_id), "--log"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            logger.warning("GitHub CLI not available to download CI logs: %s", exc)
            return None

        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip() or "unknown error"
            logger.warning("Unable to download CI logs: %s", message)
            return None

        try:
            log_path.write_text(result.stdout, encoding="utf-8")
        except OSError as exc:
            logger.warning("Unable to write CI log file: %s", exc)
            return None

        return log_path

    def _create_bug_ticket(
        self,
        *,
        category: str,
        summary: str,
        details: str,
        log_excerpt: Optional[str],
    ) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        bug_id = f"FEAT-BUG-{timestamp}"

        inbox_dir = self.project_root / "ai-inbox"
        inbox_dir.mkdir(parents=True, exist_ok=True)
        bug_file = inbox_dir / "bugs.md"

        entry_lines = []
        if not bug_file.exists():
            entry_lines.append("# Automated Bug Tickets\n")

        entry_lines.append(f"## {bug_id} - {summary}\n")
        entry_lines.append(f"*Category*: {category}\n")

        commit_sha = self._get_current_commit()
        branch = self._get_current_branch()
        if commit_sha:
            entry_lines.append(f"*Commit*: `{commit_sha}`\n")
        if branch:
            entry_lines.append(f"*Branch*: `{branch}`\n")

        cleaned_details = (details or summary).strip()
        if cleaned_details:
            entry_lines.append(cleaned_details + "\n")

        if log_excerpt:
            snippet = log_excerpt.strip()
            if len(snippet) > self._max_log_excerpt_length:
                snippet = snippet[-self._max_log_excerpt_length :]
            entry_lines.append("### Log Excerpt\n")
            entry_lines.append("```\n" + snippet + "\n```\n")

        entry_lines.append("\n")

        try:
            with bug_file.open("a", encoding="utf-8") as handle:
                handle.write("".join(entry_lines))
        except OSError as exc:
            logger.warning("Unable to write bug ticket: %s", exc)

        self._write_history_event(
            "bug_reported",
            {
                "bug_id": bug_id,
                "category": category,
                "summary": summary,
                "commit": commit_sha,
            },
        )
        return bug_id

    def write_history(self, record: Dict[str, Any]) -> None:
        if not isinstance(record, dict):
            raise TypeError(
                "History records must be mappings of field names to values."
            )

        payload: Dict[str, Any] = dict(record)
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        payload.setdefault("timestamp", timestamp)

        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            self._ensure_history_is_git_ignored()
            with self.history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=False))
                handle.write("\n")
        except OSError as exc:
            logger.warning("Unable to write history event: %s", exc)

    def _write_history_event(
        self, event_type: str, payload: Optional[Dict[str, Any]] = None
    ) -> None:
        record: Dict[str, Any] = {"event": event_type}
        if payload:
            record.update(payload)
        self.write_history(record)

    def _ensure_history_is_git_ignored(self) -> None:
        git_dir = self.project_root / ".git"
        if not git_dir.is_dir():
            return

        exclude_path = git_dir / "info" / "exclude"
        try:
            exclude_path.parent.mkdir(parents=True, exist_ok=True)
            existing = ""
            if exclude_path.exists():
                existing = exclude_path.read_text(encoding="utf-8")
            if "ai-inbox/" in existing:
                return
            with exclude_path.open("a", encoding="utf-8") as handle:
                if existing and not existing.endswith("\n"):
                    handle.write("\n")
                handle.write("ai-inbox/\n")
        except OSError:
            pass

    def generate(self):
        return self._run_agent_with_state("Developer", "generate", self._generate_impl)

    def _generate_impl(self):
        changes_applied = False
        prompt = self._build_generation_prompt()
        if not prompt:
            print("No prompt constructed for generation step; skipping.")
            self._record_agent_summary(
                "Developer",
                "generate",
                "Generation step skipped because no prompt was constructed.",
                {"status": "skipped"},
            )
            return False

        print("Invoking language model to propose code changes...")
        provider = self._resolve_llm_provider("Developer", "generate")
        try:
            llm_output = provider.generate_code(prompt)
        except Exception as exc:
            print(f"Error while invoking language model: {exc}")
            self._record_agent_summary(
                "Developer",
                "generate",
                "Generation step aborted due to language model error.",
                {"status": "error", "error": str(exc)},
            )
            self._write_history_event(
                "llm_no_output",
                {
                    "step": "generate",
                    "provider": provider.__class__.__name__,
                    "reason": "exception",
                    "error": str(exc),
                },
            )
            return False

        if not llm_output or not llm_output.strip():
            print("Language model returned an empty response; no changes applied.")
            self._record_agent_summary(
                "Developer",
                "generate",
                "Language model returned no actionable changes during generation.",
                {"status": "no_changes"},
            )
            self._write_history_event(
                "llm_no_output",
                {
                    "step": "generate",
                    "provider": provider.__class__.__name__,
                    "reason": "empty_response",
                },
            )
            return False

        applied_paths = self._apply_llm_output(llm_output)
        if applied_paths:
            self._stage_changes(applied_paths)
            details = {
                "status": "applied",
                "files_changed": sorted(applied_paths),
            }
            summary = f"Applied generated updates to {len(applied_paths)} file(s)."
            changes_applied = True
        else:
            print("Model output did not yield any actionable changes.")
            details = {"status": "no_changes"}
            summary = "Model output did not yield any actionable changes."
            if llm_output.strip().startswith("# Codex API unavailable"):
                self._write_history_event(
                    "llm_no_output",
                    {
                        "step": "generate",
                        "provider": provider.__class__.__name__,
                        "reason": "provider_fallback",
                    },
                )

        self._record_agent_summary("Developer", "generate", summary, details)
        return changes_applied

    def review(self):
        self._run_agent_with_state("Developer", "review", self._review_impl)

    def _review_impl(self):
        diff_text = self._get_pending_diff()
        if not diff_text:
            print("No code changes detected for review; skipping.")
            self._record_agent_summary(
                "Developer",
                "review",
                "Review step skipped because there were no pending changes.",
                {"status": "skipped"},
            )
            return

        prompt = self._build_review_prompt(diff_text)
        if not prompt:
            print("Unable to construct review prompt; skipping review step.")
            self._record_agent_summary(
                "Developer",
                "review",
                "Review prompt construction failed; no feedback recorded.",
                {"status": "skipped"},
            )
            return

        print("Requesting language model review of recent changes...")
        provider = self._resolve_llm_provider("Developer", "review")
        try:
            feedback = provider.generate_code(prompt)
        except Exception as exc:
            print(f"Error while invoking language model for review: {exc}")
            self._record_agent_summary(
                "Developer",
                "review",
                "Review step aborted due to language model error.",
                {"status": "error", "error": str(exc)},
            )
            return

        if not feedback or not feedback.strip():
            print("Language model returned empty review feedback.")
            self._record_agent_summary(
                "Developer",
                "review",
                "Language model returned empty feedback during review.",
                {"status": "no_feedback"},
            )
            return

        self._record_review_feedback(feedback)
        cleaned = feedback.strip()
        excerpt = cleaned if len(cleaned) <= 240 else f"{cleaned[:237]}..."
        details = {"status": "recorded"}
        if excerpt:
            details["feedback_excerpt"] = excerpt
        self._record_agent_summary(
            "Developer",
            "review",
            "Recorded language model review feedback for pending changes.",
            details,
        )

    def _build_generation_prompt(self):
        sections = []

        system_prompt = self._read_system_prompt()
        if system_prompt:
            sections.append(f"SYSTEM PROMPT:\n{system_prompt.strip()}")

        project_cfg = self.config.get("project", {})
        metadata_lines = []
        if self.project_name:
            metadata_lines.append(f"Name: {self.project_name}")
        description = project_cfg.get("description")
        if description:
            metadata_lines.append(f"Description: {description}")
        language = project_cfg.get("language")
        if language:
            metadata_lines.append(f"Primary language: {language}")
        license_name = project_cfg.get("license")
        if license_name:
            metadata_lines.append(f"License: {license_name}")
        if metadata_lines:
            sections.append("PROJECT METADATA:\n" + "\n".join(metadata_lines))

        commit_history = self._get_recent_commits()
        if commit_history:
            sections.append("RECENT COMMITS:\n" + commit_history)

        working_tree_status = self._get_git_status()
        if working_tree_status:
            sections.append("WORKING TREE STATUS:\n" + working_tree_status)

        open_tasks = self._collect_open_tasks()
        if open_tasks:
            sections.append("OPEN TASKS / TODOS:\n" + "\n".join(open_tasks))

        instructions = (
            "TASK:\nUsing the context above, determine the next meaningful code changes. Respond with either a "
            "unified diff (starting with 'diff --git') that can be applied with `git apply`, or with one or more "
            "fenced code blocks formatted as ```path/to/file.ext\\n<complete file contents>\\n```.\n"
            "Avoid extra commentary outside the provided diffs or code blocks."
        )
        sections.append(instructions)

        return "\n\n".join(section for section in sections if section).strip()

    def _build_review_prompt(self, diff_text):
        sections = []

        system_prompt = self._read_system_prompt()
        if system_prompt:
            sections.append(f"SYSTEM PROMPT:\n{system_prompt.strip()}")

        project_cfg = self.config.get("project", {})
        metadata_lines = []
        name = project_cfg.get("name")
        if name:
            metadata_lines.append(f"Project: {name}")
        language = project_cfg.get("language")
        if language:
            metadata_lines.append(f"Primary language: {language}")
        if metadata_lines:
            sections.append("PROJECT CONTEXT:\n" + "\n".join(metadata_lines))

        working_tree_status = self._get_git_status()
        if working_tree_status:
            sections.append("CURRENT STATUS:\n" + working_tree_status)

        instructions = (
            "TASK:\nYou are acting as a meticulous code reviewer."
            " Examine the pending changes below and identify potential bugs,"
            " risky assumptions, missing tests, or style issues. Provide"
            " actionable suggestions in bullet form."
        )
        sections.append(instructions)

        sections.append("CHANGES TO REVIEW:\n" + diff_text.strip())

        return "\n\n".join(section for section in sections if section).strip()

    def _read_system_prompt(self):
        prompt_path_config = self.config.get("ai", {}).get("prompt")
        if not prompt_path_config:
            return ""

        prompt_path = Path(prompt_path_config)
        if not prompt_path.is_absolute():
            prompt_path = (self.project_root / prompt_path).resolve()
        try:
            return prompt_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Unable to read system prompt '%s': %s", prompt_path, exc)
            return ""

    def _get_recent_commits(self, limit=5):
        try:
            result = subprocess.run(
                ["git", "log", f"-{limit}", "--pretty=format:%h %s"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            logger.warning("Unable to retrieve recent commits: %s", exc)
            return ""

    def _get_git_status(self):
        try:
            result = subprocess.run(
                ["git", "status", "--short"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            logger.warning("Unable to determine git status: %s", exc)
            return ""

    def _collect_open_tasks(self, limit=5):
        todos = []
        skip_dirs = {
            ".git",
            ".hg",
            ".svn",
            ".venv",
            "venv",
            "__pycache__",
            "node_modules",
            ".mypy_cache",
            ".pytest_cache",
            "dist",
            "build",
        }
        allowed_suffixes = {
            ".py",
            ".md",
            ".txt",
            ".rst",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".cfg",
        }
        try:
            for path in self.project_root.rglob("*"):
                if len(todos) >= limit:
                    break
                if path.is_dir():
                    continue
                if any(part in skip_dirs for part in path.parts):
                    continue
                suffix = path.suffix.lower()
                if suffix and suffix not in allowed_suffixes:
                    continue
                if not suffix and path.name not in {"Dockerfile", "Makefile"}:
                    continue
                try:
                    content = path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                for idx, line in enumerate(content.splitlines()):
                    if "TODO" in line or "todo" in line:
                        todos.append(
                            f"{path.relative_to(self.project_root)}:{idx + 1} {line.strip()}"
                        )
                        if len(todos) >= limit:
                            break
        except OSError:
            return []
        return todos

    def _apply_llm_output(self, output):
        if not output:
            return set()

        applied_paths = set()
        diff_applied = False

        for diff_text in self._extract_diff_candidates(output):
            paths = self._apply_diff(diff_text)
            if paths:
                applied_paths.update(paths)
                diff_applied = True

        if diff_applied:
            return applied_paths

        code_block_paths = self._apply_code_blocks(output)
        if code_block_paths:
            applied_paths.update(code_block_paths)

        return applied_paths

    def _extract_diff_candidates(self, output):
        candidates = []
        stripped = output.strip()
        if stripped and ("diff --git" in stripped or stripped.startswith("--- ")):
            candidates.append(stripped)

        pattern = re.compile(r"```(?P<header>[^\n]*)\n(?P<body>.*?)```", re.DOTALL)
        for match in pattern.finditer(output):
            header = match.group("header").strip().lower()
            body = match.group("body").strip()
            if not body:
                continue
            if (
                header in {"diff", "patch"}
                or "diff --git" in body
                or body.startswith("--- ")
            ):
                candidates.append(body)

        unique_candidates = []
        seen = set()
        for candidate in candidates:
            if candidate not in seen:
                unique_candidates.append(candidate)
                seen.add(candidate)
        return unique_candidates

    def _apply_diff(self, diff_text):
        if "diff --git" not in diff_text and not diff_text.lstrip().startswith("--- "):
            return set()

        if not diff_text.endswith("\n"):
            diff_text += "\n"

        try:
            result = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", "-"],
                cwd=self.project_root,
                input=diff_text,
                text=True,
                capture_output=True,
            )
        except FileNotFoundError as exc:
            logger.warning("git not available to apply diff: %s", exc)
            return set()

        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            if error_msg:
                logger.warning("git apply failed: %s", error_msg)
            else:
                logger.warning("git apply failed without diagnostics.")
            return set()

        print("Applied diff from model output.")
        return self._extract_paths_from_diff(diff_text)

    def _extract_paths_from_diff(self, diff_text):
        paths = set()
        for line in diff_text.splitlines():
            if line.startswith("diff --git"):
                try:
                    parts = shlex.split(line)
                except ValueError:
                    parts = line.split()
                if len(parts) >= 4:
                    for token in parts[2:4]:
                        token = token.strip('"')
                        if token.startswith("a/") or token.startswith("b/"):
                            paths.add(token[2:])
            elif line.startswith("+++ ") or line.startswith("--- "):
                token = line[4:].strip()
                if token == "/dev/null":
                    continue
                token = token.strip('"')
                if token.startswith("a/") or token.startswith("b/"):
                    token = token[2:]
                paths.add(token)
        cleaned = {p for p in paths if p and p != "/dev/null"}
        return cleaned

    def _apply_code_blocks(self, output):
        pattern = re.compile(r"```(?P<header>[^\n]*)\n(?P<body>.*?)```", re.DOTALL)
        updated_paths = set()

        for match in pattern.finditer(output):
            header = match.group("header").strip()
            if header.lower() in {"diff", "patch"}:
                continue
            body = match.group("body")
            path, content = self._extract_file_update_from_block(header, body)
            if not path:
                continue
            resolved_path = self._resolve_project_path(Path(path))
            if not resolved_path:
                print(f"Skipping invalid path in model output: {path}")
                continue
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            if content and not content.endswith("\n"):
                content += "\n"
            try:
                resolved_path.write_text(content, encoding="utf-8")
            except OSError as exc:
                logger.warning(
                    "Failed to write generated content to %s: %s",
                    resolved_path,
                    exc,
                )
                continue
            relative = resolved_path.relative_to(self.project_root)
            updated_paths.add(str(relative))
            print(f"Updated {relative} from model output.")

        return updated_paths

    def _extract_file_update_from_block(self, header, body):
        header = header.strip()
        if self._header_looks_like_path(header):
            return header, body

        first_line, remainder = self._split_first_line(body)
        possible_path = self._extract_path_marker(first_line)
        if possible_path:
            return possible_path, remainder

        return None, body

    def _header_looks_like_path(self, header):
        if not header:
            return False
        lowered = header.lower()
        language_tokens = {
            "python",
            "py",
            "javascript",
            "js",
            "typescript",
            "ts",
            "tsx",
            "jsx",
            "json",
            "yaml",
            "yml",
            "markdown",
            "md",
            "text",
            "txt",
            "bash",
            "sh",
            "shell",
            "go",
            "java",
            "c",
            "cpp",
            "c++",
            "rust",
            "rb",
            "ruby",
            "php",
            "html",
            "css",
            "sql",
            "diff",
            "patch",
            "toml",
            "ini",
            "cfg",
        }
        if lowered in language_tokens or lowered.startswith("lang="):
            return False
        if "/" in header or "\\" in header:
            return True
        if "." in header and " " not in header:
            return True
        return False

    def _split_first_line(self, body):
        if "\n" in body:
            first, remainder = body.split("\n", 1)
            return first, remainder
        return body, ""

    def _extract_path_marker(self, line):
        cleaned = line.strip()
        if not cleaned:
            return None

        prefixes = ("#", "//", "/*", "<!--")
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
                break

        for suffix in ("-->", "*/"):
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)].strip()

        lower_cleaned = cleaned.lower()
        if lower_cleaned.startswith("file:") or lower_cleaned.startswith("path:"):
            return cleaned.split(":", 1)[1].strip()
        return None

    def _resolve_project_path(self, relative_path):
        try:
            if Path(relative_path).is_absolute():
                candidate = Path(relative_path).resolve()
            else:
                candidate = (self.project_root / Path(relative_path)).resolve()
            project_root = self.project_root.resolve()
            candidate.relative_to(project_root)
            return candidate
        except (ValueError, OSError):
            return None

    def _stage_changes(self, paths):
        if not paths:
            print("No changes detected after applying model output.")
            return

        sorted_paths = sorted(paths)
        try:
            result = subprocess.run(
                ["git", "add", "--"] + sorted_paths,
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            logger.warning("git not available to stage changes: %s", exc)
            return

        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            if error_msg:
                logger.warning("git add failed: %s", error_msg)
            else:
                logger.warning("git add failed without diagnostics.")
            return

        print("Staged generated changes: " + ", ".join(sorted_paths))

    def _get_pending_diff(self):
        commands = [
            ["git", "diff", "--cached"],
            ["git", "diff"],
        ]
        collected_error = None

        for command in commands:
            try:
                result = subprocess.run(
                    command,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except FileNotFoundError as exc:
                collected_error = f"git not available for review diff: {exc}"
                break

            if result.returncode != 0:
                error_msg = result.stderr.strip() or result.stdout.strip()
                if error_msg:
                    collected_error = error_msg
                continue

            diff_text = result.stdout.strip()
            if diff_text:
                return diff_text

        if collected_error:
            logger.warning("Unable to collect diff for review: %s", collected_error)
        return ""

    def _record_review_feedback(self, feedback):
        cleaned = feedback.strip()
        if not cleaned:
            print("Language model returned empty review feedback.")
            return

        separator = "=" * 60
        print(separator)
        print("Language model review feedback:")
        print(cleaned)
        print(separator)

        review_path = self.project_root / "douglas_review.md"
        try:
            review_path.parent.mkdir(parents=True, exist_ok=True)
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            header = f"## Latest Feedback ({now})\n\n"
            if review_path.exists():
                with review_path.open("r", encoding="utf-8") as fh:
                    content = fh.read()
                # Remove previous "## Latest Feedback" section and its content
                # Keep everything before the section, and everything after the next "## " header (if any)
                pattern = r"(## Latest Feedback.*?)(?=^## |\Z)"  # non-greedy up to next section or end
                content_new = re.sub(
                    pattern, "", content, flags=re.DOTALL | re.MULTILINE
                )
                # Ensure file starts with the main header
                if not content_new.lstrip().startswith("# Douglas Review Feedback"):
                    content_new = "# Douglas Review Feedback\n\n" + content_new.lstrip()
            else:
                content_new = "# Douglas Review Feedback\n\n"
            # Write new content with latest feedback at the top
            with review_path.open("w", encoding="utf-8") as fh:
                fh.write(content_new)
                fh.write(header)
                fh.write(cleaned)
                fh.write("\n\n")
            print(
                f"Saved review feedback to {review_path.relative_to(self.project_root)}."
            )
        except OSError as exc:
            logger.warning("Unable to save review feedback to %s: %s", review_path, exc)

    def _commit_if_needed(self) -> Tuple[bool, Optional[str]]:
        if not self._has_uncommitted_changes():
            print("No pending changes detected; skipping commit step.")
            return False, None

        if not self._stage_all_changes():
            return False, None

        staged_paths = self._get_staged_paths()
        if not staged_paths:
            print("No staged changes detected after adding; skipping commit step.")
            return False, None

        commit_message = self._generate_commit_message()
        if not commit_message:
            commit_message = self.DEFAULT_COMMIT_MESSAGE

        if self._run_git_commit(commit_message):
            print(f"Created commit: {commit_message}")
            self.sprint_manager.record_commit(commit_message)
            commit_sha = self._get_current_commit()
            self._write_history_event(
                "commit",
                {
                    "message": commit_message,
                    "commit": commit_sha,
                },
            )
            return True, commit_message

        return False, None

    def _has_uncommitted_changes(self):
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            error_msg = exc.stderr or exc.stdout or ""
            error_msg = error_msg.strip()
            if error_msg:
                logger.warning(
                    "Unable to determine git status for commit: %s",
                    error_msg,
                )
            else:
                logger.warning("Unable to determine git status for commit.")
            return False
        except FileNotFoundError as exc:
            logger.warning("git not available to detect changes: %s", exc)
            return False

        return bool(result.stdout.strip())

    def _stage_all_changes(self):
        try:
            result = subprocess.run(
                ["git", "add", "-A"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            logger.warning("git not available to stage changes: %s", exc)
            return False

        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            if error_msg:
                logger.warning("git add -A failed: %s", error_msg)
            else:
                logger.warning("git add -A failed without diagnostics.")
            return False

        return True

    def _get_staged_paths(self):
        try:
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            logger.warning("Unable to list staged files: %s", exc)
            return []

        paths = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return paths

    def _get_staged_diff(self):
        try:
            result = subprocess.run(
                ["git", "diff", "--cached"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            error_msg = exc.stderr or exc.stdout or ""
            error_msg = error_msg.strip()
            if error_msg:
                logger.warning("Unable to collect staged diff: %s", error_msg)
            else:
                logger.warning("Unable to collect staged diff.")
            return ""
        except FileNotFoundError as exc:
            logger.warning("git not available to collect staged diff: %s", exc)
            return ""

        diff_text = result.stdout
        max_length = 20000
        if len(diff_text) > max_length:
            # Truncate at the last complete line before max_length
            last_newline = diff_text.rfind("\n", 0, max_length)
            if last_newline != -1:
                truncated = diff_text[:last_newline]
            else:
                truncated = diff_text[:max_length]
            truncated += "\n... (diff truncated)"
            return truncated
        return diff_text

    def _generate_commit_message(self):
        diff_text = self._get_staged_diff()
        status_text = self._get_git_status()
        prompt = self._build_commit_prompt(diff_text, status_text)
        if not prompt:
            return self.DEFAULT_COMMIT_MESSAGE

        provider = self._resolve_llm_provider("Developer", "commit")
        try:
            response = provider.generate_code(prompt)
        except Exception as exc:
            logger.warning(
                "Unable to generate commit message via language model: %s", exc
            )
            return self.DEFAULT_COMMIT_MESSAGE

        message = self._sanitize_commit_message(response)
        if not message:
            return self.DEFAULT_COMMIT_MESSAGE
        return message

    def _build_commit_prompt(self, diff_text, status_text):
        sections = []

        system_prompt = self._read_system_prompt()
        if system_prompt:
            sections.append(f"SYSTEM PROMPT:\n{system_prompt.strip()}")

        metadata = []
        if self.project_name:
            metadata.append(f"Project: {self.project_name}")
        language = self.config.get("project", {}).get("language")
        if language:
            metadata.append(f"Primary language: {language}")
        if metadata:
            sections.append("PROJECT CONTEXT:\n" + "\n".join(metadata))

        instructions = (
            "TASK:\nGenerate a concise Conventional Commits style subject line for the staged changes. "
            "Respond with a single line formatted as '<type>: <description>' without additional commentary or trailing punctuation."
        )
        sections.append(instructions)

        if status_text:
            sections.append("GIT STATUS:\n" + status_text.strip())

        if diff_text:
            sections.append("STAGED DIFF:\n" + diff_text.strip())

        return "\n\n".join(section for section in sections if section).strip()

    def _sanitize_commit_message(self, message):
        if not message:
            return ""

        first_line = message.strip().splitlines()[0].strip()
        if not first_line:
            return ""

        first_line = first_line.strip("`'\"")
        first_line = re.sub(r"\s+", " ", first_line)
        while first_line.endswith((".", "!", "?")):
            first_line = first_line[:-1].rstrip()

        max_length = 72
        if len(first_line) > max_length:
            first_line = first_line[:max_length].rstrip()

        return first_line

    def _run_git_commit(self, message):
        try:
            result = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            logger.warning("git not available to create commit: %s", exc)
            return False

        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            if error_msg:
                logger.warning("git commit failed: %s", error_msg)
            else:
                logger.warning("git commit failed without diagnostics.")
            return False

        return True

    def check(self):
        print("Checking Douglas configuration and environment...")
        print(f"Configuration loaded. Project name: {self.project_name}")

    def doctor(self):
        print("Diagnosing environment...")
        try:
            subprocess.run(["python", "--version"], check=True)
            subprocess.run(["git", "--version"], check=True)
        except subprocess.CalledProcessError:
            print("Error: Required tools are missing.")
        print("Douglas doctor complete.")

    def _render_init_template(
        self, filename: str, context: Dict[str, str], *, template: Optional[str] = None
    ) -> str:
        candidate_paths = []
        if template:
            candidate_paths.append(TEMPLATE_ROOT / "init" / template / filename)
        candidate_paths.append(TEMPLATE_ROOT / "init" / filename)

        last_error: Optional[FileNotFoundError] = None
        for template_path in candidate_paths:
            try:
                template_text = template_path.read_text(encoding="utf-8")
            except FileNotFoundError as exc:
                last_error = exc
                continue
            return Template(template_text).substitute(context)

        missing_path = candidate_paths[0] if candidate_paths else TEMPLATE_ROOT / "init"
        raise FileNotFoundError(
            f"Missing initialization template: {missing_path}"
        ) from last_error

    def init_project(
        self,
        target: Union[str, Path] = ".",
        *,
        name: Optional[str] = None,
        template: str = "python",
        ai_provider: Optional[str] = None,
        ai_model: Optional[str] = None,
        push_policy: Optional[str] = None,
        sprint_length: Optional[int] = None,
        ci: str = "github",
        git: bool = False,
        license_type: str = "none",
        non_interactive: bool = False,
    ) -> None:
        target_path = Path(target)
        print(f"Initializing new project scaffold in '{target_path}' with Douglas...")

        if target_path.exists() and target_path.is_file():
            raise ValueError(
                f"Cannot initialize project at file path '{target_path}'. Provide a directory."
            )

        target_path.mkdir(parents=True, exist_ok=True)

        if non_interactive:
            # Reserved for future interactive prompts; currently all scaffolding is non-interactive.
            pass

        if name:
            scaffold_name = name
        else:
            resolved_name = target_path.resolve().name
            if not resolved_name:
                resolved_name = Path.cwd().resolve().name
            scaffold_name = resolved_name or "DouglasProject"
        normalized_template = (template or "python").strip().lower()
        if normalized_template not in {"python", "blank"}:
            logger.warning("Unsupported template '%s'; defaulting to python.", template)
            normalized_template = "python"

        policy_candidate = (push_policy or "per_feature").strip().lower()
        if policy_candidate not in self.SUPPORTED_PUSH_POLICIES:
            logger.warning(
                "Unsupported push_policy '%s'; defaulting to per_feature.",
                push_policy,
            )
            policy_candidate = "per_feature"

        sprint_length_value = (
            self.DEFAULT_SPRINT_LENGTH_DAYS
            if sprint_length is None
            else int(sprint_length)
        )
        if sprint_length_value <= 0:
            logger.warning(
                "sprint length '%s' is invalid; defaulting to %d.",
                sprint_length_value,
                self.DEFAULT_SPRINT_LENGTH_DAYS,
            )
            sprint_length_value = self.DEFAULT_SPRINT_LENGTH_DAYS

        ci_choice = (ci or "github").strip().lower()
        if ci_choice not in {"github", "none"}:
            logger.warning("Unsupported CI provider '%s'; defaulting to github.", ci)
            ci_choice = "github"

        license_choice = (license_type or "none").strip().lower()
        if license_choice not in {"none", "mit"}:
            logger.warning(
                "Unsupported license '%s'; defaulting to none.", license_type
            )
            license_choice = "none"

        configured_language = (
            self.config.get("project", {}).get("language")
            if isinstance(self.config, dict)
            else None
        )
        default_language = str(configured_language or "python")
        language = "python" if normalized_template == "python" else default_language

        def _normalize_module_name(value: str) -> str:
            slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
            if not slug:
                slug = "app"
            return slug.replace("-", "_")

        package_name = _normalize_module_name(scaffold_name)
        module_name = "app"

        context = {
            "project_name": scaffold_name,
            "package_name": package_name,
            "module_name": module_name,
            "language": language,
            "language_title": language.title(),
            "douglas_readme_url": "https://github.com/dickymoore/Douglas",
            "current_year": str(datetime.now(timezone.utc).year),
            "license_holder": scaffold_name,
        }

        ai_config = self.config.get("ai", {}) if isinstance(self.config, dict) else {}
        if not isinstance(ai_config, Mapping):
            ai_config = {}

        provider_choice = (
            (ai_provider or self._infer_ai_provider_from_config(ai_config) or "codex")
            .strip()
            .lower()
        )
        model_choice = ai_model or self._infer_ai_model_from_config(
            ai_config, provider_choice
        )
        if not model_choice:
            model_choice = self._default_model_for_provider(provider_choice)

        config_template = deepcopy(self.load_scaffold_config())

        config_template.setdefault("project", {})
        config_template["project"].update(
            {
                "name": scaffold_name,
                "description": f"Project scaffolded by Douglas for {scaffold_name}.",
                "language": language,
            }
        )
        if license_choice == "mit":
            config_template["project"]["license"] = "MIT"
        elif license_choice == "none":
            config_template["project"]["license"] = "none"
        else:
            config_template["project"]["license"] = license_choice.upper()

        ai_section: Dict[str, Any] = config_template.setdefault("ai", {})
        providers_section = ai_section.setdefault("providers", {})
        providers_section[provider_choice] = {
            "provider": provider_choice,
        }
        if model_choice:
            providers_section[provider_choice]["model"] = model_choice
        ai_section["default_provider"] = provider_choice
        if normalized_template != "blank":
            ai_section["prompt"] = "system_prompt.md"

        loop_section = config_template.setdefault("loop", {})
        loop_section.setdefault("exit_condition_mode", "all")
        loop_section.setdefault(
            "exit_conditions",
            ["feature_delivery_complete", "sprint_demo_complete"],
        )
        loop_section.setdefault("exhaustive", False)
        if not loop_section.get("steps"):
            loop_section["steps"] = deepcopy(
                Douglas.DEFAULT_INIT_TEMPLATE["loop"]["steps"]
            )

        config_template["push_policy"] = policy_candidate
        config_template.setdefault("sprint", {})["length_days"] = sprint_length_value
        config_template.setdefault("history", {})["max_log_excerpt_length"] = (
            self._max_log_excerpt_length
        )

        douglas_config_path = target_path / "douglas.yaml"
        douglas_config_path.write_text(
            yaml.safe_dump(config_template, sort_keys=False), encoding="utf-8"
        )

        env_example = self._render_init_template(
            ".env.example.tpl", context, template=normalized_template
        )
        (target_path / ".env.example").write_text(env_example, encoding="utf-8")

        gitignore_content = self._render_init_template(
            ".gitignore.tpl", context, template=normalized_template
        )
        (target_path / ".gitignore").write_text(gitignore_content, encoding="utf-8")

        readme_content = self._render_init_template(
            "README.md.tpl", context, template=normalized_template
        )
        (target_path / "README.md").write_text(readme_content, encoding="utf-8")

        if normalized_template == "python":
            system_prompt = self._render_init_template(
                "system_prompt.md.tpl", context, template=normalized_template
            )
            (target_path / "system_prompt.md").write_text(
                system_prompt, encoding="utf-8"
            )

            pyproject = self._render_init_template(
                "pyproject.toml.tpl", context, template=normalized_template
            )
            (target_path / "pyproject.toml").write_text(pyproject, encoding="utf-8")

            requirements_dev = self._render_init_template(
                "requirements-dev.txt.tpl", context, template=normalized_template
            )
            (target_path / "requirements-dev.txt").write_text(
                requirements_dev, encoding="utf-8"
            )

            makefile_content = self._render_init_template(
                "Makefile.tpl", context, template=normalized_template
            )
            (target_path / "Makefile").write_text(makefile_content, encoding="utf-8")

            src_dir = target_path / "src" / module_name
            src_dir.mkdir(parents=True, exist_ok=True)
            init_py = self._render_init_template(
                "src_app_init.py.tpl", context, template=normalized_template
            )
            (src_dir / "__init__.py").write_text(init_py, encoding="utf-8")

            tests_dir = target_path / "tests"
            tests_dir.mkdir(parents=True, exist_ok=True)
            test_py = self._render_init_template(
                "tests_test_app.py.tpl", context, template=normalized_template
            )
            (tests_dir / "test_app.py").write_text(test_py, encoding="utf-8")

        if ci_choice == "github":
            workflow_dir = target_path / ".github" / "workflows"
            workflow_dir.mkdir(parents=True, exist_ok=True)
            workflow_content = self._render_init_template(
                "ci.yml.tpl", context, template=normalized_template
            )
            (workflow_dir / "ci.yml").write_text(workflow_content, encoding="utf-8")

        if license_choice == "mit":
            license_text = self._render_init_template(
                "LICENSE.mit.tpl", context, template=normalized_template
            )
            (target_path / "LICENSE").write_text(license_text, encoding="utf-8")

        if git:
            git_dir = target_path / ".git"
            if not git_dir.exists():
                try:
                    subprocess.run(
                        ["git", "init"],
                        cwd=target_path,
                        check=True,
                        capture_output=True,
                    )
                    subprocess.run(
                        ["git", "add", "."],
                        cwd=target_path,
                        check=True,
                        capture_output=True,
                    )
                    subprocess.run(
                        [
                            "git",
                            "-c",
                            "user.name=Douglas",
                            "-c",
                            "user.email=douglas@example.com",
                            "commit",
                            "-m",
                            "chore: initial scaffold",
                        ],
                        cwd=target_path,
                        check=True,
                        capture_output=True,
                    )
                except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                    logger.warning("Unable to initialize git repository: %s", exc)

        print("Project initialized with Douglas scaffolding.")


def _friendly_plan_reason(reason: Optional[str]) -> str:
    if not reason:
        return "backlog already prepared"
    normalized = reason.lower()
    if normalized == "existing_backlog":
        return "backlog already exists"
    if normalized == "no_llm":
        return "no planning model available"
    if normalized == "llm_error":
        return "planning model unavailable"
    if normalized == "fallback":
        return "generated default backlog"
    return normalized
