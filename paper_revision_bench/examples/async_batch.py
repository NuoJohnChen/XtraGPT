"""
Example: Async batch evaluation for large datasets
"""

import asyncio
import json
from paper_revision_bench import evaluate_async


async def main():
    # Load sample data
    with open("sample_data.json", "r") as f:
        data = json.load(f)

    original_texts = [d["original"] for d in data]
    revised_texts = [d["revised"] for d in data]
    contexts = [d.get("context") for d in data]

    print("=" * 60)
    print("Async Batch Evaluation Example")
    print("=" * 60)
    print(f"Total samples: {len(original_texts)}")

    # Run async evaluation
    # Note: Set OPENAI_API_KEY environment variable first
    results = await evaluate_async(
        original_texts=original_texts,
        revised_texts=revised_texts,
        contexts=contexts,
        section="abstract",
        criterion="overall",
        judge_model="gpt-4-turbo",
        batch_size=10,
        max_concurrent=5,  # Control API rate
        show_progress=True,
    )

    # Print results
    print("\n" + results.summary())

    # Export results
    results.to_json("batch_results.json")
    results.to_csv("batch_results.csv")
    print("\nResults exported to batch_results.json and batch_results.csv")

    # Print per-sample details
    print("\nPer-sample results:")
    for detail in results.details:
        print(f"  Sample {detail.index}: {detail.winner.value} (score: {detail.score:.2f})")
        print(f"    Explanation: {detail.explanation[:100]}...")


if __name__ == "__main__":
    # Uncomment to run (requires API key):
    # asyncio.run(main())
    print("Set OPENAI_API_KEY and uncomment asyncio.run(main()) to run this example")
