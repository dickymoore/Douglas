"""Repository provider abstractions for pull-request style interactions."""

from __future__ import annotations

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from douglas.integrations.github import GitHub


class RepositoryIntegration(Protocol):
    """Lightweight protocol implemented by VCS providers."""

    name: str
    supports_ci_monitoring: bool

    def create_pull_request(self, title: str, body: str, base: str, head: str) -> str:
        """Create a pull request (or equivalent) and return provider metadata."""


@dataclass
class GitHubIntegration:
    name: str = "github"
    supports_ci_monitoring: bool = True

    def create_pull_request(self, title: str, body: str, base: str, head: str) -> str:
        return GitHub.create_pull_request(title=title, body=body, base=base, head=head)


@dataclass
class GitLabIntegration:
    name: str = "gitlab"
    supports_ci_monitoring: bool = False

    def create_pull_request(self, title: str, body: str, base: str, head: str) -> str:
        print(
            "GitLab integration is currently stubbed. Use the GitLab CLI (glab) or UI "
            "to open a merge request."
        )
        return "GitLab merge request stub created; manual follow-up required."


@dataclass
class AzureDevOpsIntegration:
    name: str = "azure-devops"
    supports_ci_monitoring: bool = False

    def create_pull_request(self, title: str, body: str, base: str, head: str) -> str:
        print(
            "Azure DevOps integration is currently stubbed. Use the Azure CLI or UI "
            "to open a pull request."
        )
        return "Azure DevOps pull request stub created; manual follow-up required."


def resolve_repository_integration(name: str | None) -> RepositoryIntegration:
    normalized = (name or "github").strip().lower()
    if normalized in {"github", "gh"}:
        return GitHubIntegration()
    if normalized in {"gitlab", "gl"}:
        return GitLabIntegration()
    if normalized in {"azure", "azure-devops", "azure_devops", "azuredevops"}:
        return AzureDevOpsIntegration()

    print(
        f"Warning: Unknown repository provider '{name}'. Falling back to GitHub integration."
    )
    return GitHubIntegration()
