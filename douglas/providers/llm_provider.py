from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate_code(self, prompt: str) -> str:
        pass

    @staticmethod
    def create_provider(name: str, **options):
        normalized = (name or "").lower()
        if normalized == "openai":
            from douglas.providers.openai_provider import OpenAIProvider

            model = options.get("model")
            api_key = options.get("api_key")
            base_url = options.get("base_url") or options.get("api_base")
            return OpenAIProvider(model_name=model, api_key=api_key, base_url=base_url)

        raise ValueError(f"Unsupported LLM provider: {name}")
