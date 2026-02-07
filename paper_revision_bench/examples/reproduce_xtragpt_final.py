"""
Reproduce XtraGPT vs Original Text results - with more samples and clean data
"""

import json
import re
import sys
sys.path.insert(0, '/shared/hdd/andre/predictions/claude_try/paper_revision_bench')

from paper_revision_bench import evaluate


def clean_text(text):
    """Remove chat template artifacts."""
    # Remove common chat template markers
    text = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|im_end\|>', '', text)
    text = re.sub(r'<\|im_start\|>', '', text)
    text = text.strip()
    return text


def load_predictions_with_labels(model_path, section, limit=50):
    """Load predictions and original labels from jsonl file."""
    revised = []
    original = []
    filepath = f"{model_path}/{section}_predictions.jsonl"
    with open(filepath, 'r') as f:
        for idx, line in enumerate(f):
            if idx >= limit:
                break
            data = json.loads(line)
            rev = clean_text(data.get('predict', ''))
            orig = clean_text(data.get('label', ''))

            # Skip if either is empty or they're identical
            if rev and orig and rev != orig:
                revised.append(rev)
                original.append(orig)

    return original, revised


def main():
    # Path to XtraGPT predictions
    xtragpt_path = "/shared/hdd/andre/predictions/XtraGPT-1.5B"

    # Test on title section
    section = "title"
    limit = 50  # Use 50 samples

    print("=" * 60)
    print("Reproducing XtraGPT Results (Paper Methodology)")
    print("=" * 60)
    print(f"Section: {section}")
    print(f"Max samples: {limit}")
    print(f"Comparison: XtraGPT output vs Original text (before revision)")
    print()

    # Load data
    print("Loading and cleaning predictions...")
    original_texts, xtragpt_outputs = load_predictions_with_labels(xtragpt_path, section, limit)

    print(f"  Valid samples after cleaning: {len(original_texts)}")

    # Show sample
    print(f"\nSample 0:")
    print(f"  Original: {original_texts[0][:80]}...")
    print(f"  XtraGPT:  {xtragpt_outputs[0][:80]}...")

    # Run evaluation
    print("\nRunning evaluation with paper-revision-bench...")

    results = evaluate(
        original_texts=original_texts,
        revised_texts=xtragpt_outputs,
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

    diff = abs(results.win_rate - 0.899)
    if diff < 0.05:
        print("✓ Results match the paper! (within 5%)")
    elif diff < 0.10:
        print("~ Results are close to the paper (within 10%)")
    else:
        print(f"△ Difference: {diff:.1%}")
    print("=" * 60)

    # Save results
    results.to_json("reproduction_results_final.json")
    print(f"\nResults saved to reproduction_results_final.json")


if __name__ == "__main__":
    main()
