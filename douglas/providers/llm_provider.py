from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate_code(self, prompt: str) -> str:
        pass

    @staticmethod
    def create_provider(name: str):
        if name.lower() == "openai":
            from douglas.providers.openai_provider import OpenAIProvider

            return OpenAIProvider()
        else:
            raise ValueError(f"Unsupported LLM provider: {name}")
