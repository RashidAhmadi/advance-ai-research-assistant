from abc import ABC, abstractmethod

import ollama


class LLMProvider(ABC):
    """Provider-independent interface for language models."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        """Generate a response from system and user prompts."""
        pass


class OllamaProvider(LLMProvider):
    """Local LLM provider using Ollama."""

    def __init__(
        self,
        model_name: str = "qwen2.5:3b",
        host: str = "http://localhost:11434",
        temperature: float = 0.1,
        num_predict: int = 600,
        num_ctx: int = 4096
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.num_predict = num_predict
        self.num_ctx = num_ctx

        self.client = ollama.Client(host=host)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        """Generate an answer using the configured Ollama model."""

        response = self.client.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            options={
                "temperature": self.temperature,
                "num_predict": self.num_predict,
                "num_ctx": self.num_ctx
            }
        )

        return response.message.content.strip()
