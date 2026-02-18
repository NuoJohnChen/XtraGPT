"""
Judge models for evaluation.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import json
import re

from paper_revision_bench.results import SampleResult, Winner


# AlpacaEval function calling tool definition (matches paper's configs.yaml exactly)
MAKE_PARTIAL_LEADERBOARD_TOOL = {
    "type": "function",
    "function": {
        "name": "make_partial_leaderboard",
        "description": "Make a leaderboard of models given a list of the models ordered by the preference of their outputs.",
        "strict": True,
        "parameters": {
            "type": "object",
            "required": ["ordered_models"],
            "additionalProperties": False,
            "properties": {
                "ordered_models": {
                    "type": "array",
                    "description": "A list of models ordered by the preference of their outputs. The first model in the list has the best output.",
                    "items": {
                        "type": "object",
                        "required": ["model", "rank"],
                        "additionalProperties": False,
                        "properties": {
                            "model": {"type": "string", "description": "The name of the model"},
                            "rank": {"type": "number", "description": "Order of preference of the model, 1 has the best output"},
                        },
                    },
                }
            },
        },
    },
}


def list_judges() -> List[str]:
    """Return list of supported judge model types."""
    return [
        "gpt-4-1106-preview",
        "gpt-4-turbo",
        "gpt-4o",
        "gpt-4o-mini",
    ]


def parse_alpaca_eval_template(template: str) -> List[Dict[str, str]]:
    """Parse <|im_start|>/<|im_end|> template into OpenAI messages."""
    messages = []
    pattern = r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>"
    for match in re.finditer(pattern, template, re.DOTALL):
        role = match.group(1).strip()
        content = match.group(2).strip()
        messages.append({"role": role, "content": content})
    return messages


class AlpacaEvalJudge:
    """Judge using AlpacaEval's function calling format (matches paper methodology exactly)."""

    def __init__(
        self,
        model: str = "gpt-4-1106-preview",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 200,
    ):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def evaluate(
        self,
        original: str,
        revised: str,
        instruction: str,
        prompt_template: str,
        index: int,
    ) -> SampleResult:
        """Evaluate a single sample using AlpacaEval function calling.

        In the prompt template: output_1 → "m" (original), output_2 → "M" (revised).
        """
        # Format the prompt template
        formatted = prompt_template.format(
            instruction=instruction,
            output_1=original,
            output_2=revised,
        )

        # Parse into messages
        messages = parse_alpaca_eval_template(formatted)
        if not messages:
            messages = [{"role": "user", "content": formatted}]

        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=[MAKE_PARTIAL_LEADERBOARD_TOOL],
                tool_choice={"type": "function", "function": {"name": "make_partial_leaderboard"}},
            )

            # Parse function call response
            tool_call = response.choices[0].message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            ordered_models = args["ordered_models"]

            # Determine winner from ranks
            # "m" = output_1 = original, "M" = output_2 = revised
            ranks = {item["model"]: item["rank"] for item in ordered_models}
            rank_m = ranks.get("m", 2)  # original
            rank_M = ranks.get("M", 2)  # revised

            if rank_M < rank_m:
                winner = Winner.REVISED
                score = 1.0
            elif rank_m < rank_M:
                winner = Winner.ORIGINAL
                score = 0.0
            else:
                winner = Winner.TIE
                score = 0.5

            return SampleResult(
                index=index,
                winner=winner,
                score=score,
                explanation=f"Ranking: m={rank_m}, M={rank_M}",
                original_text=original,
                revised_text=revised,
                raw_response=tool_call.function.arguments,
            )

        except Exception as e:
            return SampleResult(
                index=index,
                winner=Winner.TIE,
                score=0.5,
                explanation=f"Error during evaluation: {str(e)}",
                original_text=original,
                revised_text=revised,
                raw_response="",
            )
