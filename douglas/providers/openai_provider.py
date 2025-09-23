import os

from douglas.providers.llm_provider import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not set. OpenAI calls will fail.")
        # In a real implementation, initialize the OpenAI SDK here.

    def generate_code(self, prompt: str) -> str:
        # Placeholder for real OpenAI API call
        print("[OpenAIProvider] Generating code with prompt:")
        print(prompt[:200])
        return "# Generated code placeholder"
