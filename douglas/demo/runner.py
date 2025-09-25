"""Demo script execution engine."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from douglas.logging import get_logger, log_action


logger = get_logger(__name__)


@dataclass
class DemoStepResult:
    """Outcome of executing a single demo step."""

    name: str
    step_type: str
    status: str
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DemoReport:
    """Aggregate demo execution report."""

    name: str
    steps: list[DemoStepResult]
    started_at: float
    finished_at: float
    artifacts_dir: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "duration": self.finished_at - self.started_at,
            "steps": [
                {
                    "name": step.name,
                    "type": step.step_type,
                    "status": step.status,
                    "stdout": step.stdout,
                    "stderr": step.stderr,
                    "duration": step.duration,
                    "metadata": step.metadata,
                }
                for step in self.steps
            ],
        }


class DemoRunner:
    """Run Douglas demo scripts defined via YAML or JSON DSL."""

    def __init__(self, *, workspace: Path | str | None = None) -> None:
        self.workspace = Path(workspace or ".douglas/demos")
        self.workspace.mkdir(parents=True, exist_ok=True)
        (self.workspace / "reports").mkdir(exist_ok=True)

    @log_action("demo-run", logger_factory=lambda: logger)
    def run(self, script_path: Path | str, *, output_dir: Path | str | None = None) -> DemoReport:
        script = self._load_script(Path(script_path))
        report_dir = Path(output_dir or (self.workspace / "reports" / script["name"]))
        report_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        results: list[DemoStepResult] = []

        sandbox_process: Optional[subprocess.Popen[str]] = None
        try:
            sandbox_config = script.get("sandbox")
            if sandbox_config:
                sandbox_process = self._launch_sandbox(sandbox_config, cwd=report_dir)

            for step in script.get("steps", []):
                result = self._execute_step(step, cwd=report_dir)
                results.append(result)
        finally:
            if sandbox_process is not None:
                sandbox_process.terminate()
                try:
                    sandbox_process.wait(timeout=5)
                except Exception:
                    sandbox_process.kill()

        finished = time.perf_counter()
        report = DemoReport(
            name=script["name"],
            steps=results,
            started_at=started,
            finished_at=finished,
            artifacts_dir=report_dir,
        )
        self._write_report(report)
        return report

    def _load_script(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(path)
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Demo script must be a mapping")
        payload.setdefault("name", path.stem)
        payload.setdefault("steps", [])
        return payload

    def _launch_sandbox(self, config: dict[str, Any], *, cwd: Path) -> subprocess.Popen[str]:
        command = config.get("command")
        if not command:
            raise ValueError("Sandbox configuration requires a command")
        env = os.environ.copy()
        env.update(config.get("env", {}))
        logger.info("Launching sandbox command: %s", command)
        return subprocess.Popen(
            command if isinstance(command, list) else ["/bin/sh", "-c", str(command)],
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def _execute_step(self, step: dict[str, Any], *, cwd: Path) -> DemoStepResult:
        step_type = step.get("type", "cli")
        name = step.get("name", step_type)
        started = time.perf_counter()
        try:
            if step_type == "cli":
                result = self._run_cli_step(step, cwd=cwd)
            elif step_type == "api":
                result = self._run_api_step(step)
            elif step_type == "gui":
                result = self._run_gui_step(step)
            else:
                raise ValueError(f"Unknown demo step type: {step_type}")
            status = "ok" if result["returncode"] == 0 else "failed"
            metadata = {k: v for k, v in result.items() if k not in {"stdout", "stderr", "returncode"}}
            return DemoStepResult(
                name=name,
                step_type=step_type,
                status=status,
                stdout=result.get("stdout", ""),
                stderr=result.get("stderr", ""),
                duration=time.perf_counter() - started,
                metadata=metadata,
            )
        except Exception as exc:
            logger.exception("Demo step %s failed", name)
            return DemoStepResult(
                name=name,
                step_type=step_type,
                status="error",
                stderr=str(exc),
                duration=time.perf_counter() - started,
            )

    def _run_cli_step(self, step: dict[str, Any], *, cwd: Path) -> dict[str, Any]:
        command = step.get("command")
        if not command:
            raise ValueError("CLI step requires a command")
        timeout = step.get("timeout")
        process = subprocess.run(
            command if isinstance(command, list) else ["/bin/sh", "-c", str(command)],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "stdout": process.stdout,
            "stderr": process.stderr,
            "returncode": process.returncode,
        }

    def _run_api_step(self, step: dict[str, Any]) -> dict[str, Any]:
        request = step.get("request") or {}
        url = request.get("url")
        if not url:
            raise ValueError("API step requires request.url")
        method = request.get("method", "GET").upper()
        data = request.get("json")
        headers = request.get("headers", {})
        req = urllib.request.Request(url, method=method)
        for key, value in headers.items():
            req.add_header(key, value)
        body = None
        if data is not None:
            body = json.dumps(data).encode("utf-8")
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, data=body, timeout=request.get("timeout", 30)) as response:
                payload = response.read().decode("utf-8")
                return {
                    "stdout": payload,
                    "stderr": "",
                    "returncode": 0,
                    "status": response.status,
                }
        except Exception as exc:
            return {
                "stdout": "",
                "stderr": str(exc),
                "returncode": 1,
            }

    def _run_gui_step(self, step: dict[str, Any]) -> dict[str, Any]:
        note = (
            "GUI steps are not yet implemented. Provide a Playwright script path in `script` "
            "to prepare for future support."
        )
        return {
            "stdout": note,
            "stderr": "",
            "returncode": 0,
            "pending": step.get("script"),
        }

    def _write_report(self, report: DemoReport) -> Path:
        target = report.artifacts_dir / "report.json"
        target.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        html_path = report.artifacts_dir / "report.html"
        html_path.write_text(_render_html_report(report), encoding="utf-8")
        logger.info("Demo report written to %s", html_path)
        return target


def _render_html_report(report: DemoReport) -> str:
    rows = []
    for step in report.steps:
        rows.append(
            "<tr>"
            f"<td>{step.name}</td>"
            f"<td>{step.step_type}</td>"
            f"<td>{step.status}</td>"
            f"<td><pre>{_escape(step.stdout)}</pre></td>"
            f"<td><pre>{_escape(step.stderr)}</pre></td>"
            f"<td>{step.duration:.2f}s</td>"
            "</tr>"
        )
    return (
        "<html><head><meta charset='utf-8'><title>Douglas Demo Report</title>"
        "<style>body{font-family:Arial,sans-serif;background:#f4f6fb;padding:20px;}"
        "table{width:100%;border-collapse:collapse;}th,td{border:1px solid #ccc;padding:8px;}"
        "pre{white-space:pre-wrap;word-break:break-word;background:#1e1e2f;color:#f6f8ff;padding:8px;border-radius:4px;}"
        "</style></head><body>"
        f"<h1>Demo report: {report.name}</h1>"
        f"<p>Duration: {report.finished_at - report.started_at:.2f}s</p>"
        "<table><thead><tr><th>Name</th><th>Type</th><th>Status</th><th>Stdout</th><th>Stderr</th><th>Duration</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></body></html>"
    )


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\"", "&quot;")
    )


__all__ = ["DemoRunner", "DemoReport", "DemoStepResult"]
