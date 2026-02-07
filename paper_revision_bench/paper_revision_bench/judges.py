"""
Judge models for evaluation.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import json
import re

from paper_revision_bench.results import SampleResult, Winner
from paper_revision_bench.criteria import format_prompt


def list_judges() -> List[str]:
    """Return list of supported judge model types."""
    return [
        "gpt-4-turbo",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-3.5-turbo",
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-3-haiku",
        "ollama/<model_name>",
        "vllm/<model_name>",
    ]


def create_judge(
    model: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> "BaseJudge":
    """
    Create a judge instance based on model name.

    Args:
        model: Model name (e.g., "gpt-4-turbo", "claude-3-opus", "ollama/llama3")
        api_key: API key for the model
        api_base: Base URL for API (for local models)
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response

    Returns:
        Judge instance
    """
    model_lower = model.lower()

    if model_lower.startswith("ollama/"):
        model_name = model.split("/", 1)[1]
        return OllamaJudge(
            model=model_name,
            api_base=api_base or "http://localhost:11434",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if model_lower.startswith("vllm/"):
        model_name = model.split("/", 1)[1]
        return VLLMJudge(
            model=model_name,
            api_base=api_base or "http://localhost:8000",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if "claude" in model_lower or model_lower.startswith("anthropic/"):
        model_name = model.replace("anthropic/", "")
        return AnthropicJudge(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # Default to OpenAI
    return OpenAIJudge(
        model=model,
        api_key=api_key,
        api_base=api_base,
        temperature=temperature,
        max_tokens=max_tokens,
    )


class BaseJudge(ABC):
    """Base class for judge models."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    async def _call_model(self, prompt: str) -> str:
        """Call the model and return raw response."""
        pass

    async def evaluate(
        self,
        original: str,
        revised: str,
        context: Optional[str],
        prompt_template: str,
        index: int,
    ) -> SampleResult:
        """
        Evaluate a single sample.

        Args:
            original: Original text
            revised: Revised text
            context: Optional paper context
            prompt_template: Prompt template
            index: Sample index

        Returns:
            SampleResult
        """
        # Format the prompt
        prompt = format_prompt(
            template=prompt_template,
            original=original,
            revised=revised,
            context=context,
        )

        # Call the model
        try:
            response = await self._call_model(prompt)
            result = self._parse_response(response)

            return SampleResult(
                index=index,
                winner=result["winner"],
                score=result["score"],
                explanation=result["explanation"],
                original_text=original,
                revised_text=revised,
                raw_response=response,
            )
        except Exception as e:
            # Return a tie with error explanation on failure
            return SampleResult(
                index=index,
                winner=Winner.TIE,
                score=0.5,
                explanation=f"Error during evaluation: {str(e)}",
                original_text=original,
                revised_text=revised,
                raw_response="",
            )

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the model response into structured result."""
        # Try to extract JSON from response
        try:
            # Look for JSON in the response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)

            # Parse winner
            winner_str = data.get("winner", "tie").lower().strip()
            if winner_str == "revised":
                winner = Winner.REVISED
            elif winner_str == "original":
                winner = Winner.ORIGINAL
            else:
                winner = Winner.TIE

            # Parse score
            score = float(data.get("score", 0.5))
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]

            # Parse explanation
            explanation = data.get("explanation", "No explanation provided.")

            return {
                "winner": winner,
                "score": score,
                "explanation": explanation,
            }

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback: try to infer from text
            response_lower = response.lower()
            if "revised is better" in response_lower or "revised text is better" in response_lower:
                return {"winner": Winner.REVISED, "score": 0.75, "explanation": response[:500]}
            elif "original is better" in response_lower or "original text is better" in response_lower:
                return {"winner": Winner.ORIGINAL, "score": 0.25, "explanation": response[:500]}
            else:
                return {"winner": Winner.TIE, "score": 0.5, "explanation": f"Could not parse response: {response[:500]}"}


class OpenAIJudge(BaseJudge):
    """Judge using OpenAI models."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        super().__init__(model, temperature, max_tokens)
        self.api_key = api_key
        self.api_base = api_base
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.api_base:
                kwargs["base_url"] = self.api_base
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def _call_model(self, prompt: str) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content


class AnthropicJudge(BaseJudge):
    """Judge using Anthropic Claude models."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        super().__init__(model, temperature, max_tokens)
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic
            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._client = AsyncAnthropic(**kwargs)
        return self._client

    async def _call_model(self, prompt: str) -> str:
        client = self._get_client()
        response = await client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.content[0].text


class OllamaJudge(BaseJudge):
    """Judge using local Ollama models."""

    def __init__(
        self,
        model: str,
        api_base: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        super().__init__(model, temperature, max_tokens)
        self.api_base = api_base.rstrip("/")

    async def _call_model(self, prompt: str) -> str:
        import httpx

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.api_base}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                },
            )
            response.raise_for_status()
            return response.json()["response"]


class VLLMJudge(BaseJudge):
    """Judge using vLLM-served models (OpenAI-compatible API)."""

    def __init__(
        self,
        model: str,
        api_base: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        super().__init__(model, temperature, max_tokens)
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key or "EMPTY"  # vLLM accepts any key
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=f"{self.api_base}/v1",
            )
        return self._client

    async def _call_model(self, prompt: str) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content
