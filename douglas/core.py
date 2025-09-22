import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import yaml

from douglas.cadence_manager import CadenceManager
from douglas.controls import run_state as run_state_control
from douglas.integrations.github import GitHub
from douglas.providers.llm_provider import LLMProvider
from douglas.journal import questions as question_journal
from douglas.pipelines import demo as demopipe, lint, typecheck, test as testpipe
from douglas.sprint_manager import CadenceDecision, SprintManager


@dataclass
class StepExecutionResult:
    executed: bool
    success: bool
    override_event: Optional[str] = None
    already_recorded: bool = False
    failure_reported: bool = False
    failure_details: Optional[str] = None

class Douglas:
    DEFAULT_COMMIT_MESSAGE = "chore: automated commit"
    SUPPORTED_PUSH_POLICIES = {
        'per_feature',
        'per_bug',
        'per_epic',
        'per_sprint',
    }
    
    MAX_LOG_EXCERPT_LENGTH = 4000 # Maximum number of characters kept from the end of CI logs and bug report excerpts.
    def __init__(self, config_path='douglas.yaml'):
        self.config_path = Path(config_path)
        self.config = self.load_config(self.config_path)
        self.project_root = self.config_path.resolve().parent
        self.project_name = self.config.get('project', {}).get('name', '')
        self.lm_provider = self.create_llm_provider()
        self.sprint_manager = SprintManager(sprint_length_days=self._resolve_sprint_length())
        self.cadence_manager = CadenceManager(self.config.get('cadence'), self.sprint_manager)
        self.push_policy = self._resolve_push_policy()
        self.history_path = self.project_root / 'ai-inbox' / 'history.jsonl'
        self._loop_outcomes: Dict[str, Optional[bool]] = {}
        self._ci_status: Optional[str] = None
        self._configured_steps: set[str] = set()
        self._executed_step_names: set[str] = set()
        self._blocking_questions_by_role: Dict[str, List[question_journal.Question]] = {}
        self._run_state_path = self._resolve_run_state_path()
        self._soft_stop_pending = False

    def load_config(self, path):
        try:
            with open(path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config {path}: {e}")
            sys.exit(1)

    def create_llm_provider(self):
        provider_name = self.config.get('ai', {}).get('provider', 'openai')
        if provider_name == 'openai':
            return LLMProvider.create_provider('openai')
        else:
            print(f"LLM provider '{provider_name}' not supported.")
            sys.exit(1)

    def _resolve_sprint_length(self) -> Optional[int]:
        sprint_config = self.config.get('sprint', {}) or {}
        if 'length' in sprint_config:
            print("Warning: 'length' is deprecated. Please use 'length_days' in the sprint configuration.")
        raw_length = sprint_config.get('length_days')
        if raw_length is None:
            return None
        try:
            length = int(raw_length)
        except (TypeError, ValueError):
            print(f"Warning: Invalid sprint length '{raw_length}'; falling back to default cadence.")
            return None
        if length <= 0:
            print("Warning: Sprint length must be positive; defaulting to 10 days.")
            return None
        return length

    def _resolve_push_policy(self) -> str:
        candidate = self.config.get('push_policy')
        if not candidate:
            candidate = self.config.get('vcs', {}).get('push_policy')
        if not candidate:
            return 'per_feature'

        normalized = str(candidate).strip().lower()
        if normalized not in self.SUPPORTED_PUSH_POLICIES:
            print(f"Warning: Unsupported push_policy '{candidate}'; defaulting to per_feature.")
            return 'per_feature'
        return normalized

    def _resolve_run_state_path(self) -> Path:
        paths_cfg = self.config.get('paths', {}) or {}

        candidate = paths_cfg.get('run_state_file')
        if candidate:
            run_state_path = Path(candidate)
            if not run_state_path.is_absolute():
                run_state_path = self.project_root / run_state_path
            return run_state_path

        portal_dir = paths_cfg.get('user_portal_dir')
        if portal_dir:
            portal_path = Path(portal_dir)
            if not portal_path.is_absolute():
                portal_path = self.project_root / portal_path
            return portal_path / 'run-state.txt'

        return self.project_root / 'user-portal' / 'run-state.txt'

    def run_loop(self):
        print("Starting Douglas AI development loop...")
        print(
            f"Sprint status: {self.sprint_manager.describe_day()} (iteration {self.sprint_manager.current_iteration})"
        )
        print(f"Push/PR policy: {self.push_policy}")

        self._soft_stop_pending = False
        self._check_run_state(phase='loop_start', allow_soft_stop_exit=False)

        error: Optional[BaseException] = None
        try:
            self._refresh_question_state()

            steps = self._normalize_step_configs(self.config.get('loop', {}).get('steps', []))
            executed_steps: List[str] = []
            self._executed_step_names = set()
            self._configured_steps = {str(step['name']).lower() for step in steps}
            commit_step_present = 'commit' in self._configured_steps
            self._loop_outcomes = {}
            self._ci_status = None

            for step_config in steps:
                if self._exit_conditions_met(executed_steps):
                    print("Exit condition satisfied; ending loop early.")
                    break

                step_name = step_config['name']
                decision = self.cadence_manager.evaluate_step(step_name, step_config)
                if not decision.should_run:
                    print(f"Skipping {step_name} step: {decision.reason}")
                    self._record_step_outcome(step_name, executed=False, success=False)
                    continue

                role_for_step = self._resolve_step_role(step_name)
                if self._should_defer_for_questions(role_for_step, step_name):
                    self._record_step_outcome(step_name, executed=False, success=False)
                    continue

                try:
                    result = self._execute_step(step_name, step_config, decision)
                except SystemExit as exc:
                    self._record_step_outcome(step_name, executed=True, success=False)
                    self._handle_step_failure(
                        step_name,
                        f"{step_name} step exited with status {exc.code}.",
                        None,
                    )
                    raise
                except Exception as exc:
                    self._record_step_outcome(step_name, executed=True, success=False)
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
                        self.sprint_manager.record_step_execution(step_name, event_type)
                    self._executed_step_names.add(step_name.lower())
                    executed_steps.append(step_name)
                    self._record_step_outcome(step_name, executed=True, success=result.success)
                    if not result.success and not result.failure_reported:
                        self._handle_step_failure(
                            step_name,
                            result.failure_details or f"{step_name} step reported failure.",
                            None,
                        )
                else:
                    self._record_step_outcome(step_name, executed=False, success=False)

                if self._exit_conditions_met(executed_steps):
                    print("Exit condition satisfied; ending loop early.")
                    break

            if not commit_step_present:
                committed, _ = self._commit_if_needed()
                if committed:
                    self.sprint_manager.record_step_execution('commit', None)

            self.sprint_manager.finish_iteration()
            print("Douglas loop completed.")
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._check_run_state(
                phase='loop_end',
                allow_soft_stop_exit=True,
                enforce_exit=error is None,
            )

    def _normalize_step_configs(self, raw_steps: List[Any]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for entry in raw_steps or []:
            if isinstance(entry, str):
                normalized.append({'name': entry, 'cadence': None})
                continue

            if isinstance(entry, dict):
                step_entry: Dict[str, Any] = dict(entry)
                name = step_entry.get('name') or step_entry.get('step')
                if not name and len(step_entry) == 1:
                    sole_key = next(iter(step_entry))
                    value = step_entry[sole_key]
                    if isinstance(value, dict):
                        name = sole_key
                        merged = dict(value)
                        merged['name'] = name
                        step_entry = merged
                if not name:
                    print("Warning: Skipping loop step without a name.")
                    continue
                step_entry['name'] = str(name)
                normalized.append(step_entry)
                continue

            print(f"Warning: Unsupported loop step configuration '{entry}'; skipping.")

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
                'phase': phase,
                'allow_soft_stop_exit': allow_soft_stop_exit,
                'soft_stop_pending': self._soft_stop_pending,
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
            phase='agent_start',
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
                phase='agent_end',
                agent_label=descriptor,
                allow_soft_stop_exit=False,
                enforce_exit=error is None,
            )

    def _question_context(self, **overrides: Any) -> Dict[str, Any]:
        context: Dict[str, Any] = {
            'project_root': self.project_root,
            'config': self.config,
            'sprint': self.sprint_manager.sprint_index,
        }
        context.update(overrides)
        return context

    def _refresh_question_state(self) -> None:
        try:
            open_questions = question_journal.scan_for_answers(self._question_context())
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Warning: Unable to scan for user questions: {exc}")
            self._blocking_questions_by_role = {}
            return

        blocking: Dict[str, List[question_journal.Question]] = {}
        for item in open_questions:
            answer = (item.user_answer or "").strip()
            if answer:
                print(f"Processing user answer for {item.id}: {item.topic}")
                item.agent_follow_up = self._format_question_follow_up(item)
                try:
                    question_journal.archive_question(item)
                except Exception as exc:  # pragma: no cover - defensive logging
                    print(f"Warning: Failed to archive answered question {item.id}: {exc}")
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

    def _record_step_outcome(self, step_name: str, executed: bool, success: bool) -> None:
        if executed:
            self._loop_outcomes[step_name] = bool(success)
        else:
            self._loop_outcomes[step_name] = None

    def _handle_step_failure(
        self,
        step_name: str,
        message: str,
        logs: Optional[str],
    ) -> None:
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
            'step_failure',
            {
                'step': step_name,
                'message': message,
                'bug_id': bug_id,
            },
        )

    def _exit_conditions_met(self, executed_steps: List[str]) -> bool:
        exit_conditions = self.config.get('loop', {}).get('exit_conditions') or []
        for condition in exit_conditions:
            if condition == 'sprint_demo_complete' and self.sprint_manager.has_step_run('demo'):
                return True
            if condition == 'tests_pass' and self._loop_outcomes.get('test') is True:
                return True
            if condition == 'lint_pass' and self._loop_outcomes.get('lint') is True:
                return True
            if condition == 'typecheck_pass' and self._loop_outcomes.get('typecheck') is True:
                return True
            if condition == 'local_checks_pass' and self._loop_outcomes.get('local_checks') is True:
                return True
            if condition == 'push_complete' and self._loop_outcomes.get('push') is True:
                return True
            if condition == 'pr_created' and self._loop_outcomes.get('pr') is True:
                return True
            if condition == 'ci_pass' and self._ci_status == 'success':
                return True
        return False

    def _execute_step(
        self,
        step_name: str,
        step_config: Dict[str, Any],
        cadence_decision: CadenceDecision,
    ) -> StepExecutionResult:
        override_event: Optional[str] = None
        already_recorded = False

        if step_name == 'generate':
            print("Running generate step...")
            self.generate()
            return StepExecutionResult(True, True, override_event, already_recorded)

        if step_name == 'lint':
            print("Running lint step...")
            try:
                lint.run_lint()
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    print("Lint step failed; aborting remaining steps.")
                raise
            return StepExecutionResult(True, True, override_event, already_recorded)

        if step_name == 'typecheck':
            print("Running typecheck step...")
            typecheck.run_typecheck()
            return StepExecutionResult(True, True, override_event, already_recorded)

        if step_name == 'test':
            print("Running test step...")
            testpipe.run_tests()
            return StepExecutionResult(True, True, override_event, already_recorded)

        if step_name == 'review':
            print("Running review step...")
            self.review()
            return StepExecutionResult(True, True, override_event, already_recorded)

        if step_name == 'demo':
            print("Running demo step...")
            demo_context = {
                'project_root': self.project_root,
                'config': self.config,
                'sprint_manager': self.sprint_manager,
                'history_path': self.history_path,
                'loop_outcomes': dict(self._loop_outcomes),
            }
            try:
                metadata = demopipe.write_demo_pack(demo_context)
            except Exception as exc:
                message = f"Demo generation failed: {exc}"
                print(message)
                return StepExecutionResult(
                    True,
                    False,
                    None,
                    already_recorded,
                    failure_details=message,
                )

            self._write_history_event('demo_pack_generated', metadata.as_event_payload())
            return StepExecutionResult(True, True, override_event, already_recorded)

        if step_name == 'commit':
            print("Running commit step...")
            committed, _ = self._commit_if_needed()
            if committed:
                override_event = cadence_decision.event_type
            return StepExecutionResult(True, True, override_event, already_recorded)

        if step_name == 'push':
            print("Running push step...")
            if (
                self.push_policy == 'per_sprint'
                and 'demo' in self._configured_steps
                and 'demo' not in self._executed_step_names
            ):
                print("Skipping push step: awaiting sprint demo before release.")
                self._loop_outcomes['local_checks'] = None
                return StepExecutionResult(False, False, None, already_recorded)
            decision = self.sprint_manager.should_run_push(self.push_policy)
            if not decision.should_run:
                print(f"Skipping push step: {decision.reason}")
                self._loop_outcomes['local_checks'] = None
                return StepExecutionResult(False, False, None, already_recorded)
            if not self._commits_ready_for_push(decision.event_type):
                print("No commits ready for push; skipping push step.")
                self._loop_outcomes['local_checks'] = None
                return StepExecutionResult(False, False, None, already_recorded)

            local_checks_ok, local_logs = self._run_local_checks()
            if not local_checks_ok:
                print("Local checks failed; aborting push.")
                self._handle_step_failure('local_checks', 'Local checks failed before push.', local_logs)
                return StepExecutionResult(
                    True,
                    False,
                    None,
                    already_recorded,
                    failure_reported=True,
                    failure_details='Local checks failed before push.',
                )

            success, push_logs = self._run_git_push()
            if success:
                print("Push completed according to policy.")
                self.sprint_manager.record_push(decision.event_type, self.push_policy)
                already_recorded = True
                self._write_history_event(
                    'push',
                    {
                        'policy': self.push_policy,
                        'event_type': decision.event_type,
                        'details': push_logs,
                    },
                )
                return StepExecutionResult(True, True, decision.event_type, already_recorded)

            print("Push step failed; leaving commits local.")
            self._handle_step_failure('push', 'git push failed.', push_logs)
            return StepExecutionResult(
                True,
                False,
                None,
                already_recorded,
                failure_reported=True,
                failure_details='Push failed; see logs for details.',
            )

        if step_name == 'pr':
            print("Running pr step...")
            if (
                self.push_policy == 'per_sprint'
                and 'demo' in self._configured_steps
                and 'demo' not in self._executed_step_names
            ):
                print("Skipping pr step: awaiting sprint demo before opening PR.")
                return StepExecutionResult(False, False, None, already_recorded)
            decision = self.sprint_manager.should_open_pr(self.push_policy)
            if not decision.should_run:
                print(f"Skipping pr step: {decision.reason}")
                return StepExecutionResult(False, False, None, already_recorded)
            if not self._commits_ready_for_pr(decision.event_type):
                print("No commits ready for PR; skipping pr step.")
                return StepExecutionResult(False, False, None, already_recorded)

            pr_created, pr_metadata = self._open_pull_request()
            if pr_created:
                print("Pull request created according to policy.")
                self.sprint_manager.record_pr(decision.event_type, self.push_policy)
                already_recorded = True
                self._write_history_event(
                    'pr_created',
                    {
                        'event_type': decision.event_type,
                        'policy': self.push_policy,
                        'metadata': pr_metadata,
                    },
                )
                self._monitor_ci()
                return StepExecutionResult(True, True, decision.event_type, already_recorded)

            print("Failed to create pull request; leaving for manual follow-up.")
            self._handle_step_failure('pr', 'Pull request creation failed.', pr_metadata)
            return StepExecutionResult(
                True,
                False,
                None,
                already_recorded,
                failure_reported=True,
                failure_details='Failed to create pull request.',
            )

        print(f"Step '{step_name}' is not automated; remember to handle it manually when prompted.")
        return StepExecutionResult(True, True, cadence_decision.event_type, already_recorded)

    def _commits_ready_for_push(self, event_type: Optional[str]) -> bool:
        if event_type in {'feature', 'bug', 'epic'}:
            return True
        return self.sprint_manager.commits_since_last_push > 0

    def _commits_ready_for_pr(self, event_type: Optional[str]) -> bool:
        if event_type in {'feature', 'bug', 'epic'}:
            return True
        return self.sprint_manager.commits_since_last_pr > 0

    def _discover_local_check_commands(self) -> List[List[str]]:
        commands: list[list[str]] = []
        ci_config = self.config.get('ci', {}) or {}

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
            if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, str)):
                for item in value:
                    collect(item)
                return
            command = [str(value)]
            commands.append(command)

        for key in ('additional_local_checks', 'local_checks'):
            collect(ci_config.get(key))

        existing = {tuple(cmd) for cmd in commands}
        default_candidates = [
            ('black', ['black', '--check', '.']),
            ('bandit', ['bandit', '-r', str(self.project_root)]),
            ('semgrep', ['semgrep', '--config', 'auto']),
        ]

        for tool, command in default_candidates:
            if shutil.which(tool) and tuple(command) not in existing:
                existing.add(tuple(command))
                commands.append(command)

        return commands

    def _run_local_checks(self) -> Tuple[bool, str]:
        commands = self._discover_local_check_commands()
        if not commands:
            self._loop_outcomes['local_checks'] = True
            self._write_history_event('local_checks_pass', {'commands': []})
            return True, 'No additional local checks configured.'

        logs: list[str] = []
        for command in commands:
            display = ' '.join(command)
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
                self._loop_outcomes['local_checks'] = False
                return False, "\n\n".join(logs)

            logs.append(
                self._format_command_output(
                    display, result.stdout, result.stderr, result.returncode
                )
            )

            if result.returncode != 0:
                self._loop_outcomes['local_checks'] = False
                return False, "\n\n".join(logs)

        self._loop_outcomes['local_checks'] = True
        self._write_history_event(
            'local_checks_pass',
            {
                'commands': [' '.join(command) for command in commands],
            },
        )
        return True, "\n\n".join(logs)

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

    def _run_git_push(self) -> Tuple[bool, str]:
        remote = self.config.get('vcs', {}).get('remote', 'origin')
        branch = self.config.get('vcs', {}).get('current_branch') or self._get_current_branch()
        command = ['git', 'push', '-u', remote, branch]
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
            print(f"Warning: {message}")
            return False, message

        logs.append(
            self._format_command_output(
                ' '.join(command), result.stdout, result.stderr, result.returncode
            )
        )

        if result.returncode == 0:
            return True, "\n\n".join(logs)

        error_msg = result.stderr.strip() or result.stdout.strip()
        if error_msg:
            print(f"Warning: git push failed: {error_msg}")
        else:
            print("Warning: git push failed without diagnostics.")

        lowered_error = (error_msg or '').lower()
        if 'rejected' in lowered_error or 'non-fast-forward' in lowered_error:
            print("Push rejected; attempting fast-forward pull.")
            pull_cmd = ['git', 'pull', '--ff-only', remote, branch]
            try:
                pull_result = subprocess.run(
                    pull_cmd,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError as exc:
                message = f"git not available to pull changes: {exc}"
                print(f"Warning: {message}")
                logs.append(message)
                return False, "\n\n".join(logs)

            logs.append(
                self._format_command_output(
                    ' '.join(pull_cmd),
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
                    print(f"Warning: {message}")
                    logs.append(message)
                    return False, "\n\n".join(logs)

                logs.append(
                    self._format_command_output(
                        ' '.join(command),
                        retry_result.stdout,
                        retry_result.stderr,
                        retry_result.returncode,
                    )
                )

                if retry_result.returncode == 0:
                    return True, "\n\n".join(logs)

                retry_error = retry_result.stderr.strip() or retry_result.stdout.strip()
                if retry_error:
                    print(f"Warning: git push retry failed: {retry_error}")

        return False, "\n\n".join(logs)

    def _open_pull_request(self) -> Tuple[bool, Optional[str]]:
        title, body = self._build_pr_content()
        base_branch = self.config.get('vcs', {}).get('default_branch', 'main')
        head_branch = self._get_current_branch()
        try:
            metadata = GitHub.create_pull_request(
                title=title,
                body=body,
                base=base_branch,
                head=head_branch,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Warning: Unable to create pull request: {exc}")
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
                ['git', 'log', '-1', '--pretty=%s'],
                cwd=self.project_root,
                text=True,
            ).strip()
            body = subprocess.check_output(
                ['git', 'log', '-1', '--pretty=%b'],
                cwd=self.project_root,
                text=True,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"Warning: Unable to collect latest commit summary: {exc}")
            return "", ""
        return subject, body

    def _get_current_branch(self) -> str:
        configured = self.config.get('vcs', {}).get('current_branch')
        if configured:
            return str(configured)
        try:
            branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=self.project_root,
                text=True,
            ).strip()
            return branch or 'HEAD'
        except (subprocess.CalledProcessError, FileNotFoundError):
            return 'HEAD'

    def _get_current_commit(self) -> Optional[str]:
        try:
            return subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.project_root,
                text=True,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def _monitor_ci(self, max_attempts: int = 10, poll_interval: int = 10) -> Optional[bool]:
        if shutil.which('gh') is None:
            print('GitHub CLI not available; skipping CI monitoring.')
            self._ci_status = None
            return None

        commit_sha = self._get_current_commit()
        if not commit_sha:
            print('Unable to determine latest commit SHA for CI monitoring.')
            self._ci_status = None
            return None

        branch = self._get_current_branch()
        for attempt in range(max_attempts):
            try:
                result = subprocess.run(
                    [
                        'gh',
                        'run',
                        'list',
                        '--limit',
                        '20',
                        '--json',
                        'databaseId,headSha,status,conclusion,url',
                        '--branch',
                        branch,
                    ],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError as exc:
                print(f'Warning: GitHub CLI not available for CI monitoring: {exc}')
                self._ci_status = None
                return None

            if result.returncode != 0:
                message = result.stderr.strip() or result.stdout.strip() or 'unknown error'
                print(f'Warning: Unable to list GitHub runs: {message}')
                time.sleep(poll_interval)
                continue

            try:
                runs = json.loads(result.stdout or '[]')
            except json.JSONDecodeError as exc:
                print(f'Warning: Unable to parse GitHub run list: {exc}')
                time.sleep(poll_interval)
                continue

            target_run = next((run for run in runs if run.get('headSha') == commit_sha), None)
            if not target_run:
                time.sleep(poll_interval)
                continue

            status = target_run.get('status')
            conclusion = target_run.get('conclusion')
            run_id = target_run.get('databaseId')
            run_url = target_run.get('url')

            if status != 'completed':
                print(f'Waiting for CI run {run_id} to complete (status: {status}).')
                time.sleep(poll_interval)
                continue

            if conclusion == 'success':
                print('CI checks succeeded for the latest commit.')
                self._ci_status = 'success'
                self._write_history_event(
                    'ci_pass',
                    {
                        'run_id': run_id,
                        'url': run_url,
                        'commit': commit_sha,
                    },
                )
                return True

            summary = f'CI run {run_id} failed with conclusion {conclusion}.'
            log_path = self._download_ci_logs(run_id)
            excerpt = None
            if log_path and log_path.exists():
                try:
                    content = log_path.read_text(encoding='utf-8')
                    excerpt = content[-self.MAX_LOG_EXCERPT_LENGTH:]
                except OSError as exc:
                    excerpt = f'Unable to read CI log file: {exc}'

            self._ci_status = 'failure'
            self._write_history_event(
                'ci_fail',
                {
                    'run_id': run_id,
                    'url': run_url,
                    'commit': commit_sha,
                    'conclusion': conclusion,
                },
            )
            self._handle_step_failure('ci', summary, excerpt)
            return False

        print('CI run not found or did not complete within the monitoring window.')
        self._ci_status = None
        return None

    def _download_ci_logs(self, run_id: Optional[int]) -> Optional[Path]:
        if not run_id:
            return None
        if shutil.which('gh') is None:
            print('GitHub CLI not available; cannot download CI logs.')
            return None

        log_dir = self.project_root / 'ai-inbox' / 'ci'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f'ci-run-{run_id}.log'

        try:
            result = subprocess.run(
                ['gh', 'run', 'view', str(run_id), '--log'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            print(f'Warning: GitHub CLI not available to download CI logs: {exc}')
            return None

        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip() or 'unknown error'
            print(f'Warning: Unable to download CI logs: {message}')
            return None

        try:
            log_path.write_text(result.stdout, encoding='utf-8')
        except OSError as exc:
            print(f'Warning: Unable to write CI log file: {exc}')
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
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
        bug_id = f'FEAT-BUG-{timestamp}'

        inbox_dir = self.project_root / 'ai-inbox'
        inbox_dir.mkdir(parents=True, exist_ok=True)
        bug_file = inbox_dir / 'bugs.md'

        entry_lines = []
        if not bug_file.exists():
            entry_lines.append('# Automated Bug Tickets\n')

        entry_lines.append(f'## {bug_id} - {summary}\n')
        entry_lines.append(f'*Category*: {category}\n')

        commit_sha = self._get_current_commit()
        branch = self._get_current_branch()
        if commit_sha:
            entry_lines.append(f'*Commit*: `{commit_sha}`\n')
        if branch:
            entry_lines.append(f'*Branch*: `{branch}`\n')

        cleaned_details = (details or summary).strip()
        if cleaned_details:
            entry_lines.append(cleaned_details + '\n')

        if log_excerpt:
            snippet = log_excerpt.strip()
            if len(snippet) > self.MAX_LOG_EXCERPT_LENGTH:
                snippet = snippet[-self.MAX_LOG_EXCERPT_LENGTH:]
            entry_lines.append('### Log Excerpt\n')
            entry_lines.append('```\n' + snippet + '\n```\n')

        entry_lines.append('\n')

        try:
            with bug_file.open('a', encoding='utf-8') as handle:
                handle.write(''.join(entry_lines))
        except OSError as exc:
            print(f'Warning: Unable to write bug ticket: {exc}')

        self._write_history_event(
            'bug_reported',
            {
                'bug_id': bug_id,
                'category': category,
                'summary': summary,
                'commit': commit_sha,
            },
        )
        return bug_id

    def _write_history_event(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        record: Dict[str, Any] = {
            'timestamp': timestamp,
            'event': event_type,
        }
        if payload:
            record.update(payload)

        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            self._ensure_history_is_git_ignored()
            with self.history_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, sort_keys=False))
                handle.write('\n')
        except OSError as exc:
            print(f'Warning: Unable to write history event: {exc}')

    def _ensure_history_is_git_ignored(self) -> None:
        git_dir = self.project_root / '.git'
        if not git_dir.is_dir():
            return

        exclude_path = git_dir / 'info' / 'exclude'
        try:
            exclude_path.parent.mkdir(parents=True, exist_ok=True)
            existing = ''
            if exclude_path.exists():
                existing = exclude_path.read_text(encoding='utf-8')
            if 'ai-inbox/' in existing:
                return
            with exclude_path.open('a', encoding='utf-8') as handle:
                if existing and not existing.endswith('\n'):
                    handle.write('\n')
                handle.write('ai-inbox/\n')
        except OSError:
            pass

    def generate(self):
        self._run_agent_with_state('Developer', 'generate', self._generate_impl)

    def _generate_impl(self):
        prompt = self._build_generation_prompt()
        if not prompt:
            print("No prompt constructed for generation step; skipping.")
            return

        print("Invoking language model to propose code changes...")
        try:
            llm_output = self.lm_provider.generate_code(prompt)
        except Exception as exc:
            print(f"Error while invoking language model: {exc}")
            return

        if not llm_output or not llm_output.strip():
            print("Language model returned an empty response; no changes applied.")
            return

        applied_paths = self._apply_llm_output(llm_output)
        if applied_paths:
            self._stage_changes(applied_paths)
        else:
            print("Model output did not yield any actionable changes.")

    def review(self):
        self._run_agent_with_state('Developer', 'review', self._review_impl)

    def _review_impl(self):
        diff_text = self._get_pending_diff()
        if not diff_text:
            print("No code changes detected for review; skipping.")
            return

        prompt = self._build_review_prompt(diff_text)
        if not prompt:
            print("Unable to construct review prompt; skipping review step.")
            return

        print("Requesting language model review of recent changes...")
        try:
            feedback = self.lm_provider.generate_code(prompt)
        except Exception as exc:
            print(f"Error while invoking language model for review: {exc}")
            return

        if not feedback or not feedback.strip():
            print("Language model returned empty review feedback.")
            return

        self._record_review_feedback(feedback)

    def _build_generation_prompt(self):
        sections = []

        system_prompt = self._read_system_prompt()
        if system_prompt:
            sections.append(f"SYSTEM PROMPT:\n{system_prompt.strip()}")

        project_cfg = self.config.get('project', {})
        metadata_lines = []
        if self.project_name:
            metadata_lines.append(f"Name: {self.project_name}")
        description = project_cfg.get('description')
        if description:
            metadata_lines.append(f"Description: {description}")
        language = project_cfg.get('language')
        if language:
            metadata_lines.append(f"Primary language: {language}")
        license_name = project_cfg.get('license')
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

        project_cfg = self.config.get('project', {})
        metadata_lines = []
        name = project_cfg.get('name')
        if name:
            metadata_lines.append(f"Project: {name}")
        language = project_cfg.get('language')
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
        prompt_path_config = self.config.get('ai', {}).get('prompt')
        if not prompt_path_config:
            return ""

        prompt_path = Path(prompt_path_config)
        if not prompt_path.is_absolute():
            prompt_path = (self.project_root / prompt_path).resolve()
        try:
            return prompt_path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError) as exc:
            print(f"Warning: Unable to read system prompt '{prompt_path}': {exc}")
            return ""

    def _get_recent_commits(self, limit=5):
        try:
            result = subprocess.run(
                ['git', 'log', f'-{limit}', '--pretty=format:%h %s'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"Warning: Unable to retrieve recent commits: {exc}")
            return ""

    def _get_git_status(self):
        try:
            result = subprocess.run(
                ['git', 'status', '--short'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"Warning: Unable to determine git status: {exc}")
            return ""

    def _collect_open_tasks(self, limit=5):
        todos = []
        skip_dirs = {
            '.git',
            '.hg',
            '.svn',
            '.venv',
            'venv',
            '__pycache__',
            'node_modules',
            '.mypy_cache',
            '.pytest_cache',
            'dist',
            'build',
        }
        allowed_suffixes = {
            '.py',
            '.md',
            '.txt',
            '.rst',
            '.js',
            '.ts',
            '.tsx',
            '.jsx',
            '.json',
            '.yaml',
            '.yml',
            '.toml',
            '.ini',
            '.cfg',
        }
        try:
            for path in self.project_root.rglob('*'):
                if len(todos) >= limit:
                    break
                if path.is_dir():
                    continue
                if any(part in skip_dirs for part in path.parts):
                    continue
                suffix = path.suffix.lower()
                if suffix and suffix not in allowed_suffixes:
                    continue
                if not suffix and path.name not in {'Dockerfile', 'Makefile'}:
                    continue
                try:
                    content = path.read_text(encoding='utf-8')
                except (UnicodeDecodeError, OSError):
                    continue
                for idx, line in enumerate(content.splitlines()):
                    if 'TODO' in line or 'todo' in line:
                        todos.append(f"{path.relative_to(self.project_root)}:{idx + 1} {line.strip()}")
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
        if stripped and ('diff --git' in stripped or stripped.startswith('--- ')):
            candidates.append(stripped)

        pattern = re.compile(r"```(?P<header>[^\n]*)\n(?P<body>.*?)```", re.DOTALL)
        for match in pattern.finditer(output):
            header = match.group('header').strip().lower()
            body = match.group('body').strip()
            if not body:
                continue
            if header in {'diff', 'patch'} or 'diff --git' in body or body.startswith('--- '):
                candidates.append(body)

        unique_candidates = []
        seen = set()
        for candidate in candidates:
            if candidate not in seen:
                unique_candidates.append(candidate)
                seen.add(candidate)
        return unique_candidates

    def _apply_diff(self, diff_text):
        if 'diff --git' not in diff_text and not diff_text.lstrip().startswith('--- '):
            return set()

        if not diff_text.endswith('\n'):
            diff_text += '\n'

        try:
            result = subprocess.run(
                ['git', 'apply', '--whitespace=nowarn', '-'],
                cwd=self.project_root,
                input=diff_text,
                text=True,
                capture_output=True,
            )
        except FileNotFoundError as exc:
            print(f"Warning: git not available to apply diff: {exc}")
            return set()

        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            if error_msg:
                print(f"git apply failed: {error_msg}")
            else:
                print("git apply failed without diagnostics.")
            return set()

        print("Applied diff from model output.")
        return self._extract_paths_from_diff(diff_text)

    def _extract_paths_from_diff(self, diff_text):
        paths = set()
        for line in diff_text.splitlines():
            if line.startswith('diff --git'):
                try:
                    parts = shlex.split(line)
                except ValueError:
                    parts = line.split()
                if len(parts) >= 4:
                    for token in parts[2:4]:
                        token = token.strip('"')
                        if token.startswith('a/') or token.startswith('b/'):
                            paths.add(token[2:])
            elif line.startswith('+++ ') or line.startswith('--- '):
                token = line[4:].strip()
                if token == '/dev/null':
                    continue
                token = token.strip('"')
                if token.startswith('a/') or token.startswith('b/'):
                    token = token[2:]
                paths.add(token)
        cleaned = {p for p in paths if p and p != '/dev/null'}
        return cleaned

    def _apply_code_blocks(self, output):
        pattern = re.compile(r"```(?P<header>[^\n]*)\n(?P<body>.*?)```", re.DOTALL)
        updated_paths = set()

        for match in pattern.finditer(output):
            header = match.group('header').strip()
            if header.lower() in {'diff', 'patch'}:
                continue
            body = match.group('body')
            path, content = self._extract_file_update_from_block(header, body)
            if not path:
                continue
            resolved_path = self._resolve_project_path(Path(path))
            if not resolved_path:
                print(f"Skipping invalid path in model output: {path}")
                continue
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            if content and not content.endswith('\n'):
                content += '\n'
            try:
                resolved_path.write_text(content, encoding='utf-8')
            except OSError as exc:
                print(f"Failed to write generated content to {resolved_path}: {exc}")
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
            'python', 'py', 'javascript', 'js', 'typescript', 'ts', 'tsx', 'jsx',
            'json', 'yaml', 'yml', 'markdown', 'md', 'text', 'txt', 'bash', 'sh',
            'shell', 'go', 'java', 'c', 'cpp', 'c++', 'rust', 'rb', 'ruby', 'php',
            'html', 'css', 'sql', 'diff', 'patch', 'toml', 'ini', 'cfg'
        }
        if lowered in language_tokens or lowered.startswith('lang='):
            return False
        if '/' in header or '\\' in header:
            return True
        if '.' in header and ' ' not in header:
            return True
        return False

    def _split_first_line(self, body):
        if '\n' in body:
            first, remainder = body.split('\n', 1)
            return first, remainder
        return body, ''

    def _extract_path_marker(self, line):
        cleaned = line.strip()
        if not cleaned:
            return None

        prefixes = ('#', '//', '/*', '<!--')
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break

        for suffix in ('-->', '*/'):
            if cleaned.endswith(suffix):
                cleaned = cleaned[:-len(suffix)].strip()

        lower_cleaned = cleaned.lower()
        if lower_cleaned.startswith('file:') or lower_cleaned.startswith('path:'):
            return cleaned.split(':', 1)[1].strip()
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
                ['git', 'add', '--'] + sorted_paths,
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            print(f"Warning: git not available to stage changes: {exc}")
            return

        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            if error_msg:
                print(f"Warning: git add failed: {error_msg}")
            else:
                print("Warning: git add failed without diagnostics.")
            return

        print("Staged generated changes: " + ", ".join(sorted_paths))

    def _get_pending_diff(self):
        commands = [
            ['git', 'diff', '--cached'],
            ['git', 'diff'],
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
            print(f"Warning: Unable to collect diff for review: {collected_error}")
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

        review_path = self.project_root / 'douglas_review.md'
        try:
            review_path.parent.mkdir(parents=True, exist_ok=True)
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            header = f"## Latest Feedback ({now})\n\n"
            if review_path.exists():
                with review_path.open('r', encoding='utf-8') as fh:
                    content = fh.read()
                # Remove previous "## Latest Feedback" section and its content
                # Keep everything before the section, and everything after the next "## " header (if any)
                pattern = r"(## Latest Feedback.*?)(?=^## |\Z)"  # non-greedy up to next section or end
                content_new = re.sub(pattern, "", content, flags=re.DOTALL | re.MULTILINE)
                # Ensure file starts with the main header
                if not content_new.lstrip().startswith("# Douglas Review Feedback"):
                    content_new = "# Douglas Review Feedback\n\n" + content_new.lstrip()
            else:
                content_new = "# Douglas Review Feedback\n\n"
            # Write new content with latest feedback at the top
            with review_path.open('w', encoding='utf-8') as fh:
                fh.write(content_new)
                fh.write(header)
                fh.write(cleaned)
                fh.write("\n\n")
            print(f"Saved review feedback to {review_path.relative_to(self.project_root)}.")
        except OSError as exc:
            print(f"Warning: Unable to save review feedback to {review_path}: {exc}")

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
                'commit',
                {
                    'message': commit_message,
                    'commit': commit_sha,
                },
            )
            return True, commit_message

        return False, None

    def _has_uncommitted_changes(self):
        try:
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            error_msg = exc.stderr or exc.stdout or ''
            error_msg = error_msg.strip()
            if error_msg:
                print(f"Warning: Unable to determine git status for commit: {error_msg}")
            else:
                print("Warning: Unable to determine git status for commit.")
            return False
        except FileNotFoundError as exc:
            print(f"Warning: git not available to detect changes: {exc}")
            return False

        return bool(result.stdout.strip())

    def _stage_all_changes(self):
        try:
            result = subprocess.run(
                ['git', 'add', '-A'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            print(f"Warning: git not available to stage changes: {exc}")
            return False

        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            if error_msg:
                print(f"Warning: git add -A failed: {error_msg}")
            else:
                print("Warning: git add -A failed without diagnostics.")
            return False

        return True

    def _get_staged_paths(self):
        try:
            result = subprocess.run(
                ['git', 'diff', '--cached', '--name-only'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"Warning: Unable to list staged files: {exc}")
            return []

        paths = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return paths

    def _get_staged_diff(self):
        try:
            result = subprocess.run(
                ['git', 'diff', '--cached'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            error_msg = exc.stderr or exc.stdout or ''
            error_msg = error_msg.strip()
            if error_msg:
                print(f"Warning: Unable to collect staged diff: {error_msg}")
            else:
                print("Warning: Unable to collect staged diff.")
            return ""
        except FileNotFoundError as exc:
            print(f"Warning: git not available to collect staged diff: {exc}")
            return ""

        diff_text = result.stdout
        max_length = 20000
        if len(diff_text) > max_length:
            # Truncate at the last complete line before max_length
            last_newline = diff_text.rfind('\n', 0, max_length)
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

        try:
            response = self.lm_provider.generate_code(prompt)
        except Exception as exc:
            print(f"Warning: Unable to generate commit message via language model: {exc}")
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
        language = self.config.get('project', {}).get('language')
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
        while first_line.endswith(('.', '!', '?')):
            first_line = first_line[:-1].rstrip()

        max_length = 72
        if len(first_line) > max_length:
            first_line = first_line[:max_length].rstrip()

        return first_line

    def _run_git_commit(self, message):
        try:
            result = subprocess.run(
                ['git', 'commit', '-m', message],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            print(f"Warning: git not available to create commit: {exc}")
            return False

        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            if error_msg:
                print(f"Warning: git commit failed: {error_msg}")
            else:
                print("Warning: git commit failed without diagnostics.")
            return False

        return True


    def check(self):
        print("Checking Douglas configuration and environment...")
        print(f"Configuration loaded. Project name: {self.project_name}")

    def doctor(self):
        print("Diagnosing environment...")
        try:
            subprocess.run(['python', '--version'], check=True)
            subprocess.run(['git', '--version'], check=True)
        except subprocess.CalledProcessError:
            print("Error: Required tools are missing.")
        print("Douglas doctor complete.")

    def init_project(self, project_name: str, non_interactive: bool = False):
        print(f"Initializing new project '{project_name}' with Douglas...")
        project_dir = Path(project_name)
        project_dir.mkdir(parents=True, exist_ok=True)
        config_template = {
            'project': {
                'name': project_name,
                'description': f'Project scaffolded by Douglas for {project_name}.',
                'language': self.config.get('project', {}).get('language', 'python'),
            },
            'ai': {
                'provider': self.config.get('ai', {}).get('provider', 'openai'),
                'model': self.config.get('ai', {}).get('model', 'gpt-4'),
                'prompt': 'system_prompt.md',
            },
            'loop': {
                'steps': [
                    {'name': 'generate'},
                    {'name': 'lint'},
                    {'name': 'typecheck'},
                    {'name': 'test'},
                    {'name': 'commit'},
                    {'name': 'push'},
                    {'name': 'pr'},
                ],
                'exit_conditions': ['ci_pass'],
            },
            'push_policy': 'per_feature',
            'sprint': {'length_days': 10},
            'vcs': {'default_branch': 'main'},
        }

        (project_dir / 'douglas.yaml').write_text(
            yaml.safe_dump(config_template, sort_keys=False),
            encoding='utf-8',
        )

        readme_content = (
            f"# {project_name}\n\n"
            "This project was generated by Douglas with a ready-to-run Python scaffold.\n"
            "Update the configuration in `douglas.yaml` to match your workflow and goals.\n"
        )
        (project_dir / 'README.md').write_text(readme_content, encoding='utf-8')

        system_prompt = (
            "You are Douglas, an AI assistant that helps implement high-quality software. "
            "Follow best practices, write tests, and keep responses actionable.\n"
        )
        (project_dir / 'system_prompt.md').write_text(system_prompt, encoding='utf-8')

        gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]

# Virtual environments
.venv/
venv/

# Distribution / packaging
build/
dist/
*.egg-info/

# Douglas artifacts
ai-inbox/
douglas_review.md
"""
        (project_dir / '.gitignore').write_text(gitignore_content, encoding='utf-8')

        src_dir = project_dir / 'src'
        src_dir.mkdir(parents=True, exist_ok=True)
        (src_dir / '__init__.py').write_text('', encoding='utf-8')
        main_py = (
            """
from __future__ import annotations


def greet(name: str = "world") -> str:
    '''Return a friendly greeting.'''

    return f"Hello, {name}!"


def main() -> None:
    '''Entrypoint for the scaffolded application.'''

    print(greet())


if __name__ == "__main__":
    main()
"""
        ).strip()
        (src_dir / 'main.py').write_text(main_py + '\n', encoding='utf-8')

        tests_dir = project_dir / 'tests'
        tests_dir.mkdir(parents=True, exist_ok=True)
        (tests_dir / '__init__.py').write_text('', encoding='utf-8')
        test_main = (
            """
from src import main


def test_greet_returns_custom_message():
    assert main.greet("Douglas") == "Hello, Douglas!"
"""
        ).strip()
        (tests_dir / 'test_main.py').write_text(test_main + '\n', encoding='utf-8')

        workflow_dir = project_dir / '.github' / 'workflows'
        workflow_dir.mkdir(parents=True, exist_ok=True)
        workflow_content = (
            """
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt || true
          pip install pytest
      - name: Lint
        run: |
          pip install ruff black isort
          ruff check .
          black --check .
          isort --check-only .
      - name: Tests
        run: pytest
"""
        ).strip()
        (workflow_dir / 'ci.yml').write_text(workflow_content + '\n', encoding='utf-8')

        print(
            "Project initialized with configuration, sample source files, tests, and CI workflow."
        )
