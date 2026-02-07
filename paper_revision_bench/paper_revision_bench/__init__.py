"""
Paper Revision Bench - A benchmarking toolkit for evaluating paper revision quality.
"""

from paper_revision_bench.core import evaluate, evaluate_async
from paper_revision_bench.results import EvaluationResult, SampleResult
from paper_revision_bench.criteria import list_criteria, list_sections, get_criterion_prompt
from paper_revision_bench.judges import list_judges
from paper_revision_bench.paper_prompts import get_paper_eval_prompt, list_paper_sections

__version__ = "0.1.0"
__all__ = [
    "evaluate",
    "evaluate_async",
    "EvaluationResult",
    "SampleResult",
    "list_criteria",
    "list_sections",
    "list_judges",
    "get_criterion_prompt",
    "get_paper_eval_prompt",
    "list_paper_sections",
]
