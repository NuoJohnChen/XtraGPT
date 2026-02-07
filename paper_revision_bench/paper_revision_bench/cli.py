"""
Command-line interface for paper-revision-bench.
"""

import argparse
import asyncio
import json
import sys
from typing import Optional

from paper_revision_bench import evaluate, list_sections, list_criteria


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark paper revision quality using LLM-as-a-judge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate from JSON file
  paper-revision-bench --input data.json --judge gpt-4-turbo

  # Evaluate with specific section and criterion
  paper-revision-bench --input data.json --section abstract --criterion conciseness

  # Use local model
  paper-revision-bench --input data.json --judge ollama/llama3 --api-base http://localhost:11434

  # List available sections and criteria
  paper-revision-bench --list-sections
  paper-revision-bench --list-criteria abstract
        """,
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input JSON/JSONL file with 'original' and 'revised' fields",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results.json",
        help="Output file for results (default: results.json)",
    )
    parser.add_argument(
        "--judge", "-j",
        type=str,
        default="gpt-4-turbo",
        help="Judge model (default: gpt-4-turbo)",
    )
    parser.add_argument(
        "--section", "-s",
        type=str,
        default="abstract",
        help="Paper section (default: abstract)",
    )
    parser.add_argument(
        "--criterion", "-c",
        type=str,
        default="overall",
        help="Evaluation criterion (default: overall)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key (or set via environment variable)",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        help="API base URL for local models",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Judge temperature (default: 0.0)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent API calls (default: 5)",
    )
    parser.add_argument(
        "--list-sections",
        action="store_true",
        help="List available paper sections",
    )
    parser.add_argument(
        "--list-criteria",
        type=str,
        nargs="?",
        const="all",
        help="List available criteria (optionally for a specific section)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Handle list commands
    if args.list_sections:
        print("Available sections:")
        for section in list_sections():
            print(f"  - {section}")
        return 0

    if args.list_criteria:
        if args.list_criteria == "all":
            print("Available criteria (all sections):")
            for criterion in list_criteria():
                print(f"  - {criterion}")
        else:
            print(f"Available criteria for '{args.list_criteria}':")
            try:
                for criterion in list_criteria(args.list_criteria):
                    print(f"  - {criterion}")
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1
        return 0

    # Require input file for evaluation
    if not args.input:
        parser.error("--input is required for evaluation")

    # Load input data
    try:
        data = load_input(args.input)
    except Exception as e:
        print(f"Error loading input: {e}", file=sys.stderr)
        return 1

    # Extract texts
    original_texts = [d.get("original", d.get("original_text", "")) for d in data]
    revised_texts = [d.get("revised", d.get("revised_text", "")) for d in data]
    contexts = [d.get("context", d.get("paper_content")) for d in data]

    # Check if contexts are all None
    if all(c is None for c in contexts):
        contexts = None

    # Run evaluation
    try:
        results = evaluate(
            original_texts=original_texts,
            revised_texts=revised_texts,
            contexts=contexts,
            section=args.section,
            criterion=args.criterion,
            judge_model=args.judge,
            api_key=args.api_key,
            api_base=args.api_base,
            temperature=args.temperature,
            show_progress=not args.quiet,
        )
    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        return 1

    # Print summary
    if not args.quiet:
        print("\n" + results.summary())

    # Save results
    results.to_json(args.output)
    if not args.quiet:
        print(f"\nResults saved to: {args.output}")

    return 0


def load_input(path: str):
    """Load input data from JSON or JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Try JSON first
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        else:
            return [data]
    except json.JSONDecodeError:
        pass

    # Try JSONL
    data = []
    for line in content.split("\n"):
        line = line.strip()
        if line:
            data.append(json.loads(line))
    return data


if __name__ == "__main__":
    sys.exit(main())
