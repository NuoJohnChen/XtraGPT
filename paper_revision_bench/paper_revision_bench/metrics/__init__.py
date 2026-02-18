"""
Metrics module: AlpacaEval-style length-controlled win rate
and weighted overall score aggregation.

Converts EvaluationResult to alpaca_eval format and computes
GLM-based length-controlled win rate.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import pandas as pd

if TYPE_CHECKING:
    from paper_revision_bench.results import EvaluationResult

__all__ = ["get_length_controlled_winrate", "compute_weighted_overall"]

_WINNER_TO_PREFERENCE = {
    "original": 1.0,   # baseline wins
    "revised": 2.0,    # model wins
    "tie": 1.5,
}

# Paper section weights for overall score (Table 2 in paper)
# title:abstract:introduction:background:evaluation:conclusion = 2:4:6:3:3:2
SECTION_WEIGHTS = {
    "title": 2,
    "abstract": 4,
    "introduction": 6,
    "background": 3,
    "evaluation": 3,
    "conclusion": 2,
}


def _result_to_alpaca_df(
    result: "EvaluationResult",
    model_name: str = "revised",
    baseline_name: str = "original",
    annotator: str = "paper_revision_bench",
) -> pd.DataFrame:
    """Convert EvaluationResult to alpaca_eval-format DataFrame."""
    rows = []
    for sample in result.details:
        rows.append({
            "index": sample.index,
            "preference": _WINNER_TO_PREFERENCE[sample.winner.value],
            "output_1": sample.original_text,
            "output_2": sample.revised_text,
            "generator_1": baseline_name,
            "generator_2": model_name,
            "annotator": annotator,
        })
    return pd.DataFrame(rows)


def get_length_controlled_winrate(
    result: "EvaluationResult",
    model_name: str = "revised",
    baseline_name: str = "original",
    annotator: str = "paper_revision_bench",
    glm_name: str = "length_controlled_no_instruction_difficulty",
    save_weights_dir: Optional[str] = None,
) -> dict:
    """
    Compute AlpacaEval-style length-controlled win rate from an EvaluationResult.

    Requires optional dependencies: pip install paper-revision-bench[alpaca]

    Args:
        result: EvaluationResult from evaluate()
        model_name: Name for the revised/model outputs (generator_2)
        baseline_name: Name for the original/baseline outputs (generator_1)
        annotator: Annotator name used for weight saving
        glm_name: GLM variant to use
        save_weights_dir: Directory to save GLM weights. None = don't save.

    Returns:
        dict with keys: win_rate, length_controlled_winrate, lc_standard_error,
        n_wins, n_losses, n_ties, n_total
    """
    try:
        from paper_revision_bench.metrics.glm_winrate import (
            get_length_controlled_winrate as _glm_lcwr,
        )
    except ImportError as e:
        raise ImportError(
            "Length-controlled win rate requires additional dependencies. "
            "Install with: pip install paper-revision-bench[alpaca]"
        ) from e

    df = _result_to_alpaca_df(result, model_name, baseline_name, annotator)
    return _glm_lcwr(
        annotations=df,
        glm_name=glm_name,
        save_weights_dir=save_weights_dir,
    )


def compute_weighted_overall(
    section_results: Dict[str, "EvaluationResult"],
    use_lc_winrate: bool = False,
    model_name: str = "revised",
    baseline_name: str = "original",
) -> dict:
    """
    Compute the paper's weighted overall win rate across all 6 sections.

    Weights: title:abstract:introduction:background:evaluation:conclusion = 2:4:6:3:3:2

    Args:
        section_results: dict mapping section name -> EvaluationResult
        use_lc_winrate: if True, use length-controlled win rate (requires [alpaca] extras)
        model_name: name for the revised outputs (used for LC win rate)
        baseline_name: name for the original outputs (used for LC win rate)

    Returns:
        dict with keys:
            weighted_win_rate: float (0-1), weighted overall win rate
            section_win_rates: dict of per-section win rates
            section_weights: dict of weights used
            total_weight: int

    Example:
        results = {}
        for section in ["title", "abstract", "introduction", "background", "evaluation", "conclusion"]:
            results[section] = evaluate(original_texts, revised_texts, section=section)
        overall = compute_weighted_overall(results)
        print(f"Overall Win Rate: {overall['weighted_win_rate']:.1%}")
    """
    section_win_rates = {}
    weighted_sum = 0.0
    total_weight = 0

    for section, result in section_results.items():
        weight = SECTION_WEIGHTS.get(section, 1)
        if use_lc_winrate:
            lc = get_length_controlled_winrate(result, model_name=model_name, baseline_name=baseline_name)
            win_rate = lc["length_controlled_winrate"] / 100.0
        else:
            win_rate = result.win_rate
        section_win_rates[section] = win_rate
        weighted_sum += win_rate * weight
        total_weight += weight

    weighted_win_rate = weighted_sum / total_weight if total_weight > 0 else 0.0

    return {
        "weighted_win_rate": weighted_win_rate,
        "section_win_rates": section_win_rates,
        "section_weights": {s: SECTION_WEIGHTS.get(s, 1) for s in section_results},
        "total_weight": total_weight,
    }
