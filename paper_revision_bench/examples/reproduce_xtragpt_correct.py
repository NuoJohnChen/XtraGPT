"""
Reproduce XtraGPT vs Original Text results from the paper using paper-revision-bench

Paper methodology: Compare model's revised output against the ORIGINAL text (before revision)
"""

import json
import sys
sys.path.insert(0, '/shared/hdd/andre/predictions/claude_try/paper_revision_bench')

from paper_revision_bench import evaluate


def load_predictions_with_labels(model_path, section, limit=10):
    """Load predictions and original labels from jsonl file."""
    revised = []
    original = []
    filepath = f"{model_path}/{section}_predictions.jsonl"
    with open(filepath, 'r') as f:
        for idx, line in enumerate(f):
            if idx >= limit:
                break
            data = json.loads(line)
            revised.append(data.get('predict', ''))
            original.append(data.get('label', ''))  # label is the original text
    return original, revised


def main():
    # Path to XtraGPT predictions
    xtragpt_path = "/shared/hdd/andre/predictions/XtraGPT-1.5B"

    # Test on title section
    section = "title"
    limit = 10  # Use 10 samples for quick test

    print("=" * 60)
    print("Reproducing XtraGPT Results (Paper Methodology)")
    print("=" * 60)
    print(f"Section: {section}")
    print(f"Samples: {limit}")
    print(f"Comparison: XtraGPT output vs Original text (before revision)")
    print()

    # Load data - original is the text BEFORE revision, revised is XtraGPT output
    print("Loading predictions...")
    original_texts, xtragpt_outputs = load_predictions_with_labels(xtragpt_path, section, limit)

    print(f"  Original texts: {len(original_texts)} samples")
    print(f"  XtraGPT outputs: {len(xtragpt_outputs)} samples")

    # Show sample
    print(f"\nSample 0:")
    print(f"  Original: {original_texts[0][:100]}...")
    print(f"  XtraGPT:  {xtragpt_outputs[0][:100]}...")

    # Run evaluation using paper-revision-bench
    print("\nRunning evaluation with paper-revision-bench...")

    results = evaluate(
        original_texts=original_texts,   # Text BEFORE revision
        revised_texts=xtragpt_outputs,   # XtraGPT's revised output
        section="title",
        criterion="overall",
        judge_model="gpt-4-turbo",
        show_progress=True,
    )

    # Print results
    print("\n" + results.summary())

    # Compare with paper results
    print("\n" + "=" * 60)
    print("Comparison with Paper Results")
    print("=" * 60)
    print(f"Paper (Table 2) XtraGPT-1.5B Title Win Rate: 89.9%")
    print(f"Our reproduction Win Rate: {results.win_rate:.1%}")

    if abs(results.win_rate - 0.899) < 0.1:
        print("✓ Results are consistent with the paper!")
    else:
        print(f"Note: Small sample size ({limit}) may cause variance")
    print("=" * 60)

    # Save results
    results.to_json("reproduction_results_correct.json")
    print("\nResults saved to reproduction_results_correct.json")

    # Show per-sample details
    print("\nPer-sample results:")
    for d in results.details:
        print(f"  [{d.index}] {d.winner.value:8s} (score: {d.score:.2f}) - {d.explanation[:60]}...")


if __name__ == "__main__":
    main()
