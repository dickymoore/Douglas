from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate_code(self, prompt: str) -> str:
        pass

    @staticmethod
    def create_provider(name: str, **options):
        normalized = (name or "").strip().lower()

        model = options.get("model") or options.get("model_name")
        api_key = options.get("api_key") or options.get("token")
        base_url = options.get("base_url") or options.get("api_base")

        if normalized in {"openai", "gpt", "gpt-4", "gpt4"}:
            from douglas.providers.openai_provider import OpenAIProvider

            return OpenAIProvider(model_name=model, api_key=api_key, base_url=base_url)

        if normalized in {"codex", "openai-codex", "codex-openai", "code-davinci"}:
            from douglas.providers.codex_provider import CodexProvider

            return CodexProvider(model_name=model, api_key=api_key, base_url=base_url)

        if normalized in {"claude", "claude_code", "claude-code", "anthropic"}:
            from douglas.providers.claude_code_provider import ClaudeCodeProvider

            return ClaudeCodeProvider(model_name=model, api_key=api_key)

        if normalized in {"gemini", "google", "google-ai", "googleai"}:
            from douglas.providers.gemini_provider import GeminiProvider

            return GeminiProvider(model_name=model, api_key=api_key)

        if normalized in {"copilot", "github-copilot", "github"}:
            from douglas.providers.copilot_provider import CopilotProvider

            token = options.get("token") or api_key
            return CopilotProvider(model_name=model, token=token)

        raise ValueError(f"Unsupported LLM provider: {name}")
