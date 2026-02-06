"""
Analyze improvement in paper scores before/after XtraGPT revision.

Usage:
    python analyze_results.py --before before_scores.json --after after_scores.json
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict


def load_scores(filepath: Path) -> list:
    """Load scores from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_stats(scores: list, metric: str) -> dict:
    """Calculate statistics for a metric."""
    values = [s.get(metric) for s in scores if s.get(metric) is not None]
    if not values:
        return {"mean": None, "count": 0}

    # Handle string values (like decisions)
    if isinstance(values[0], str):
        counts = defaultdict(int)
        for v in values:
            counts[v] += 1
        return {"distribution": dict(counts), "count": len(values)}

    # Handle numeric values
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "count": len(values)
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze score improvements")
    parser.add_argument("--before", type=Path, required=True,
                        help="Scores before revision")
    parser.add_argument("--after", type=Path, required=True,
                        help="Scores after revision")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output file for analysis (optional)")
    args = parser.parse_args()

    before_scores = load_scores(args.before)
    after_scores = load_scores(args.after)

    metrics = ["overall", "soundness", "presentation", "contribution", "confidence"]

    print("=" * 60)
    print("XtraGPT Revision Impact Analysis")
    print("=" * 60)
    print(f"Papers analyzed: {len(before_scores)}")
    print()

    results = {}

    for metric in metrics:
        before_stats = calculate_stats(before_scores, metric)
        after_stats = calculate_stats(after_scores, metric)

        if before_stats.get("mean") is not None and after_stats.get("mean") is not None:
            improvement = after_stats["mean"] - before_stats["mean"]
            pct_change = (improvement / before_stats["mean"]) * 100 if before_stats["mean"] != 0 else 0

            print(f"{metric.upper()}")
            print(f"  Before: {before_stats['mean']:.2f}")
            print(f"  After:  {after_stats['mean']:.2f}")
            print(f"  Change: {improvement:+.2f} ({pct_change:+.1f}%)")
            print()

            results[metric] = {
                "before": before_stats["mean"],
                "after": after_stats["mean"],
                "improvement": improvement,
                "pct_change": pct_change
            }

    # Analyze decision changes
    print("DECISION ANALYSIS")
    before_decisions = calculate_stats(before_scores, "decision")
    after_decisions = calculate_stats(after_scores, "decision")

    if "distribution" in before_decisions:
        print(f"  Before: {before_decisions['distribution']}")
    if "distribution" in after_decisions:
        print(f"  After:  {after_decisions['distribution']}")

    # Calculate acceptance rate change
    before_accept = before_decisions.get("distribution", {}).get("Accept", 0)
    after_accept = after_decisions.get("distribution", {}).get("Accept", 0)
    total = before_decisions.get("count", 1)

    before_rate = (before_accept / total) * 100
    after_rate = (after_accept / total) * 100

    print(f"  Acceptance Rate: {before_rate:.1f}% -> {after_rate:.1f}% ({after_rate - before_rate:+.1f}%)")

    results["acceptance_rate"] = {
        "before": before_rate,
        "after": after_rate,
        "improvement": after_rate - before_rate
    }

    print()
    print("=" * 60)

    # Save results if output specified
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Analysis saved to: {args.output}")


if __name__ == "__main__":
    main()
