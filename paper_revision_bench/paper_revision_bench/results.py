"""
Evaluation results and data structures.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from enum import Enum


class Winner(str, Enum):
    """Winner of a pairwise comparison."""
    REVISED = "revised"  # Revised text is better
    ORIGINAL = "original"  # Original text is better
    TIE = "tie"  # Both are equally good


@dataclass
class SampleResult:
    """Result for a single sample evaluation."""
    index: int
    winner: Winner
    score: float  # 0.0 = original wins, 0.5 = tie, 1.0 = revised wins
    explanation: str
    original_text: str = ""
    revised_text: str = ""
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "winner": self.winner.value,
            "score": self.score,
            "explanation": self.explanation,
            "original_text": self.original_text,
            "revised_text": self.revised_text,
        }


@dataclass
class EvaluationResult:
    """Aggregated evaluation results."""
    details: List[SampleResult]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        """Proportion of samples where revised text wins."""
        if not self.details:
            return 0.0
        wins = sum(1 for d in self.details if d.winner == Winner.REVISED)
        return wins / len(self.details)

    @property
    def lose_rate(self) -> float:
        """Proportion of samples where original text wins."""
        if not self.details:
            return 0.0
        losses = sum(1 for d in self.details if d.winner == Winner.ORIGINAL)
        return losses / len(self.details)

    @property
    def tie_rate(self) -> float:
        """Proportion of samples that are ties."""
        if not self.details:
            return 0.0
        ties = sum(1 for d in self.details if d.winner == Winner.TIE)
        return ties / len(self.details)

    @property
    def average_score(self) -> float:
        """Average score (0-1 scale, higher = revised is better)."""
        if not self.details:
            return 0.0
        return sum(d.score for d in self.details) / len(self.details)

    @property
    def n_wins(self) -> int:
        """Number of wins for revised text."""
        return sum(1 for d in self.details if d.winner == Winner.REVISED)

    @property
    def n_losses(self) -> int:
        """Number of losses (original wins)."""
        return sum(1 for d in self.details if d.winner == Winner.ORIGINAL)

    @property
    def n_ties(self) -> int:
        """Number of ties."""
        return sum(1 for d in self.details if d.winner == Winner.TIE)

    @property
    def total(self) -> int:
        """Total number of samples."""
        return len(self.details)

    def summary(self) -> str:
        """Return a summary string."""
        return (
            f"Evaluation Results ({self.total} samples)\n"
            f"{'=' * 40}\n"
            f"Win Rate:  {self.win_rate:.1%} ({self.n_wins}/{self.total})\n"
            f"Lose Rate: {self.lose_rate:.1%} ({self.n_losses}/{self.total})\n"
            f"Tie Rate:  {self.tie_rate:.1%} ({self.n_ties}/{self.total})\n"
            f"Avg Score: {self.average_score:.3f}\n"
            f"{'=' * 40}\n"
            f"Judge: {self.metadata.get('judge_model', 'N/A')}\n"
            f"Section: {self.metadata.get('section', 'N/A')}\n"
            f"Criterion: {self.metadata.get('criterion', 'N/A')}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "win_rate": self.win_rate,
            "lose_rate": self.lose_rate,
            "tie_rate": self.tie_rate,
            "average_score": self.average_score,
            "n_wins": self.n_wins,
            "n_losses": self.n_losses,
            "n_ties": self.n_ties,
            "total": self.total,
            "metadata": self.metadata,
            "details": [d.to_dict() for d in self.details],
        }

    def to_json(self, path: str, indent: int = 2) -> None:
        """Export results to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent, ensure_ascii=False)

    def to_csv(self, path: str) -> None:
        """Export results to CSV file."""
        import pandas as pd

        rows = []
        for d in self.details:
            rows.append({
                "index": d.index,
                "winner": d.winner.value,
                "score": d.score,
                "explanation": d.explanation,
                "original_text": d.original_text[:200] + "..." if len(d.original_text) > 200 else d.original_text,
                "revised_text": d.revised_text[:200] + "..." if len(d.revised_text) > 200 else d.revised_text,
            })

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False, encoding="utf-8")

    def __repr__(self) -> str:
        return f"EvaluationResult(win_rate={self.win_rate:.1%}, total={self.total})"
