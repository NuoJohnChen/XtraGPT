"""
Core evaluation functions.
Uses AlpacaEval's function calling format to match the paper's methodology.
"""

import asyncio
import os
from typing import List, Optional

from paper_revision_bench.results import EvaluationResult, SampleResult
from paper_revision_bench.paper_prompts import get_paper_eval_prompt, list_paper_sections
from paper_revision_bench.judges import AlpacaEvalJudge, list_judges


def evaluate(
    original_texts: List[str],
    revised_texts: List[str],
    instructions: Optional[List[str]] = None,
    section: str = "abstract",
    judge_model: str = "gpt-4-1106-preview",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 200,
    max_concurrent: int = 5,
    show_progress: bool = True,
    **kwargs,
) -> EvaluationResult:
    """
    Evaluate paper revision quality using AlpacaEval's function calling format.

    This matches the paper's methodology exactly: GPT-4-Turbo ranks two outputs
    via the make_partial_leaderboard function call, using section-specific criteria.

    Args:
        original_texts: List of original/baseline texts (mapped to model "m" / output_1)
        revised_texts: List of revised/model texts (mapped to model "M" / output_2)
        instructions: Optional list of revision instructions. If None, uses a default.
        section: Paper section (title, abstract, introduction, background, evaluation, conclusion)
        judge_model: OpenAI model for judging (default: gpt-4-1106-preview, matches paper)
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        temperature: Judge temperature (default: 0.0 for deterministic)
        max_tokens: Max tokens for judge response (default: 200, matches paper)
        max_concurrent: Maximum concurrent API calls
        show_progress: Whether to show progress bar

    Returns:
        EvaluationResult object with win_rate, details, etc.
    """
    return asyncio.run(
        evaluate_async(
            original_texts=original_texts,
            revised_texts=revised_texts,
            instructions=instructions,
            section=section,
            judge_model=judge_model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_concurrent=max_concurrent,
            show_progress=show_progress,
        )
    )


async def evaluate_async(
    original_texts: List[str],
    revised_texts: List[str],
    instructions: Optional[List[str]] = None,
    section: str = "abstract",
    judge_model: str = "gpt-4-1106-preview",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 200,
    max_concurrent: int = 5,
    show_progress: bool = True,
) -> EvaluationResult:
    """Async version of evaluate()."""
    if len(original_texts) != len(revised_texts):
        raise ValueError(
            f"Length mismatch: original_texts ({len(original_texts)}) != revised_texts ({len(revised_texts)})"
        )

    if instructions is not None and len(instructions) != len(original_texts):
        raise ValueError(
            f"Length mismatch: instructions ({len(instructions)}) != original_texts ({len(original_texts)})"
        )

    section = section.lower()
    if section not in list_paper_sections():
        raise ValueError(f"Unknown section: {section}. Valid: {list_paper_sections()}")

    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set. Please set it or pass api_key parameter.")

    # Get the paper's prompt template for this section
    prompt_template = get_paper_eval_prompt(section)

    # Create judge
    judge = AlpacaEvalJudge(
        model=judge_model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Default instruction if not provided
    default_instruction = f"Improve the {section} section of this academic paper."

    # Run evaluation with concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)

    async def evaluate_single(i: int) -> SampleResult:
        async with semaphore:
            instruction = instructions[i] if instructions else default_instruction
            return await judge.evaluate(
                original=original_texts[i],
                revised=revised_texts[i],
                instruction=instruction,
                prompt_template=prompt_template,
                index=i,
            )

    if show_progress:
        from tqdm.asyncio import tqdm_asyncio
        results = await tqdm_asyncio.gather(
            *[evaluate_single(i) for i in range(len(original_texts))],
            desc=f"Evaluating {section} ({judge_model})",
        )
    else:
        results = await asyncio.gather(*[evaluate_single(i) for i in range(len(original_texts))])

    return EvaluationResult(
        details=list(results),
        metadata={
            "judge_model": judge_model,
            "section": section,
            "temperature": temperature,
            "total_samples": len(original_texts),
        },
    )
