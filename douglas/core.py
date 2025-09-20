import re
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

from douglas.providers.llm_provider import LLMProvider
from douglas.pipelines import lint, typecheck, test as testpipe

class Douglas:
    def __init__(self, config_path='douglas.yaml'):
        self.config_path = Path(config_path)
        self.config = self.load_config(self.config_path)
        self.project_root = self.config_path.resolve().parent
        self.project_name = self.config.get('project', {}).get('name', '')
        self.lm_provider = self.create_llm_provider()

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

    def run_loop(self):
        print("Starting Douglas AI development loop...")
        steps = self.config.get('loop', {}).get('steps', [])
        for step in steps:
            if step == 'generate':
                print("Running generate step...")
                self.generate()
                continue
            elif step == 'lint':
                print("Running lint step...")
                try:
                    lint.run_lint()
                except SystemExit as exc:
                    if exc.code not in (None, 0):
                        print("Lint step failed; aborting remaining steps.")
                    raise
            elif step == 'typecheck':
                print("Running typecheck step...")
                typecheck.run_typecheck()
            elif step == 'test':
                print("Running test step...")
                testpipe.run_tests()
            elif step == 'review':
                print("Running review step...")
                self.review()
            else:
                print(f"Unknown step '{step}'; skipping.")
        print("Douglas loop completed.")
        # In later versions, commit/push/PR steps would follow.

    def generate(self):
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
            new_file = not review_path.exists()
            with review_path.open('a', encoding='utf-8') as fh:
                if new_file:
                    fh.write("# Douglas Review Feedback\n\n")
                fh.write("## Latest Feedback\n\n")
                fh.write(cleaned)
                fh.write("\n\n")
            print(f"Saved review feedback to {review_path.relative_to(self.project_root)}.")
        except OSError as exc:
            print(f"Warning: Unable to save review feedback to {review_path}: {exc}")


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
        (project_dir / 'douglas.yaml').write_text('project:\n  name: "' + project_name + '"\n', encoding='utf-8')
        (project_dir / 'README.md').write_text(f"# {project_name}\nCreated with Douglas\n", encoding='utf-8')
        print("Project initialized. Please review and customize douglas.yaml.")
