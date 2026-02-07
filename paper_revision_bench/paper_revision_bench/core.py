"""
Core evaluation functions.
"""

import asyncio
import os
from typing import List, Optional, Union

from paper_revision_bench.results import EvaluationResult, SampleResult
from paper_revision_bench.criteria import get_criterion_prompt, validate_section, validate_criterion
from paper_revision_bench.judges import create_judge


def evaluate(
    original_texts: List[str],
    revised_texts: List[str],
    contexts: Optional[List[str]] = None,
    section: str = "abstract",
    criterion: str = "overall",
    judge_model: str = "gpt-4-turbo",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    show_progress: bool = True,
) -> EvaluationResult:
    """
    Evaluate paper revision quality using LLM-as-a-judge.

    Args:
        original_texts: List of original texts before revision
        revised_texts: List of revised texts after revision
        contexts: Optional list of full paper contexts
        section: Paper section type (title, abstract, introduction, background, evaluation, conclusion)
        criterion: Evaluation criterion (conciseness, clarity, impact, overall, etc.)
        judge_model: Model to use as judge (e.g., "gpt-4-turbo", "claude-3-opus", "ollama/llama3")
        api_key: API key (or set via environment variable)
        api_base: API base URL for local models
        temperature: Judge temperature (default: 0.0 for deterministic)
        max_tokens: Max tokens for judge response
        show_progress: Whether to show progress bar

    Returns:
        EvaluationResult object with win_rate, details, etc.
    """
    return asyncio.run(
        evaluate_async(
            original_texts=original_texts,
            revised_texts=revised_texts,
            contexts=contexts,
            section=section,
            criterion=criterion,
            judge_model=judge_model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            batch_size=10,
            max_concurrent=5,
            show_progress=show_progress,
        )
    )


async def evaluate_async(
    original_texts: List[str],
    revised_texts: List[str],
    contexts: Optional[List[str]] = None,
    section: str = "abstract",
    criterion: str = "overall",
    judge_model: str = "gpt-4-turbo",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    batch_size: int = 10,
    max_concurrent: int = 5,
    show_progress: bool = True,
) -> EvaluationResult:
    """
    Async version of evaluate() for batch processing.

    Additional Args:
        batch_size: Number of samples per batch
        max_concurrent: Maximum concurrent API calls
    """
    # Validate inputs
    if len(original_texts) != len(revised_texts):
        raise ValueError(
            f"Length mismatch: original_texts ({len(original_texts)}) != revised_texts ({len(revised_texts)})"
        )

    if contexts is not None and len(contexts) != len(original_texts):
        raise ValueError(
            f"Length mismatch: contexts ({len(contexts)}) != original_texts ({len(original_texts)})"
        )

    validate_section(section)
    validate_criterion(section, criterion)

    # Get API key from environment if not provided
    if api_key is None:
        api_key = _get_api_key_from_env(judge_model)

    # Create judge
    judge = create_judge(
        model=judge_model,
        api_key=api_key,
        api_base=api_base,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Get evaluation prompt template
    prompt_template = get_criterion_prompt(section, criterion)

    # Prepare samples
    samples = []
    for i in range(len(original_texts)):
        context = contexts[i] if contexts else None
        samples.append({
            "index": i,
            "original": original_texts[i],
            "revised": revised_texts[i],
            "context": context,
        })

    # Run evaluation with concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)

    async def evaluate_single(sample: dict) -> SampleResult:
        async with semaphore:
            return await judge.evaluate(
                original=sample["original"],
                revised=sample["revised"],
                context=sample["context"],
                prompt_template=prompt_template,
                index=sample["index"],
            )

    # Progress tracking
    if show_progress:
        from tqdm.asyncio import tqdm_asyncio
        results = await tqdm_asyncio.gather(
            *[evaluate_single(s) for s in samples],
            desc=f"Evaluating ({judge_model})",
        )
    else:
        results = await asyncio.gather(*[evaluate_single(s) for s in samples])

    # Build result
    return EvaluationResult(
        details=list(results),
        metadata={
            "judge_model": judge_model,
            "section": section,
            "criterion": criterion,
            "temperature": temperature,
            "total_samples": len(samples),
        }
    )


def _get_api_key_from_env(model: str) -> Optional[str]:
    """Get API key from environment based on model type."""
    model_lower = model.lower()

    if model_lower.startswith("ollama/") or model_lower.startswith("vllm/"):
        return None  # Local models don't need API key

    if "claude" in model_lower or model_lower.startswith("anthropic/"):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Please set it or pass api_key parameter."
            )
        return key

    # Default to OpenAI
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it or pass api_key parameter."
        )
    return key
