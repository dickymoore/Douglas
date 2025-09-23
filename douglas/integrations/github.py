import subprocess
from typing import Optional


class GitHub:
    @staticmethod
    def create_pull_request(
        title: str,
        body: str,
        base: str = "main",
        head: Optional[str] = None,
    ) -> str:
        if head is None:
            head = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
            ).strip()

        command = [
            "gh",
            "pr",
            "create",
            "--title",
            title,
            "--body",
            body,
            "--base",
            base,
            "--head",
            head,
        ]

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Failed to create PR: {exc.stderr or exc.stdout}"
            ) from exc

        output = result.stdout.strip() or result.stderr.strip()
        return output or f"Pull request created for {head} -> {base}"
