"""
Full Paper Evaluation using AI-Scientist framework.

Usage:
    python ai_scientist_eval.py --paper_path ./paper.pdf --output results.json
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Evaluate papers using AI-Scientist")
    parser.add_argument("--paper_path", type=Path, required=True,
                        help="Path to paper PDF file")
    parser.add_argument("--output", type=Path, default=Path("review_results.json"),
                        help="Output file for review results")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="OpenAI model to use for review")
    parser.add_argument("--num_reflections", type=int, default=1,
                        help="Number of reflection rounds")
    parser.add_argument("--num_reviews", type=int, default=1,
                        help="Number of review ensembles")
    args = parser.parse_args()

    # Check if AI-Scientist is installed
    try:
        from ai_scientist.perform_review import load_paper, perform_review
    except ImportError:
        print("Error: AI-Scientist not found.")
        print("Please install it first:")
        print("  git clone https://github.com/SakanaAI/AI-Scientist.git")
        print("  cd AI-Scientist && pip install -e .")
        return 1

    import openai
    import os

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return 1

    client = openai.OpenAI()

    print(f"Loading paper: {args.paper_path}")
    paper_txt = load_paper(str(args.paper_path))

    print(f"Running review with model: {args.model}")
    review = perform_review(
        paper_txt,
        args.model,
        client,
        num_reflections=args.num_reflections,
        num_fs_examples=1,
        num_reviews_ensemble=args.num_reviews,
        temperature=0.1,
    )

    # Extract key metrics
    result = {
        "paper": str(args.paper_path),
        "model": args.model,
        "overall": review.get("Overall"),
        "decision": review.get("Decision"),
        "soundness": review.get("Soundness"),
        "presentation": review.get("Presentation"),
        "contribution": review.get("Contribution"),
        "confidence": review.get("Confidence"),
        "strengths": review.get("Strengths"),
        "weaknesses": review.get("Weaknesses"),
    }

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nReview completed!")
    print(f"  Overall Score: {result['overall']}")
    print(f"  Decision: {result['decision']}")
    print(f"  Soundness: {result['soundness']}")
    print(f"  Presentation: {result['presentation']}")
    print(f"  Contribution: {result['contribution']}")
    print(f"\nFull results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
