"""
Reproduce XtraGPT vs Qwen2.5-7B-Instruct results from the paper using paper-revision-bench
"""

import json
import sys
sys.path.insert(0, '/shared/hdd/andre/predictions/claude_try/paper_revision_bench')

from paper_revision_bench import evaluate

# Load prediction data
def load_predictions(model_path, section, limit=10):
    """Load predictions from jsonl file."""
    results = []
    filepath = f"{model_path}/{section}_predictions.jsonl"
    with open(filepath, 'r') as f:
        for idx, line in enumerate(f):
            if idx >= limit:
                break
            data = json.loads(line)
            results.append(data.get('predict', ''))
    return results

# Load prompts (as original text baseline)
def load_prompts(section, limit=10):
    """Load original prompts."""
    filepath = f"/shared/hdd/andre/predictions/formatted/prompts_noexplain/{section}.json"
    with open(filepath, 'r') as f:
        prompts = json.load(f)
    return prompts[:limit]


def main():
    # Paths
    xtragpt_path = "/shared/hdd/andre/predictions/XtraGPT-1.5B"
    baseline_path = "/shared/hdd/andre/predictions/experimental/Qwen2.5-7B-Instruct-noexplain"

    # Test on title section (smallest, 700 samples total)
    section = "title"
    limit = 10  # Use 10 samples for quick test

    print("=" * 60)
    print("Reproducing XtraGPT vs Qwen2.5-7B-Instruct (Title Section)")
    print("=" * 60)
    print(f"Section: {section}")
    print(f"Samples: {limit}")
    print()

    # Load data
    print("Loading predictions...")
    xtragpt_outputs = load_predictions(xtragpt_path, section, limit)
    baseline_outputs = load_predictions(baseline_path, section, limit)

    print(f"  XtraGPT-1.5B: {len(xtragpt_outputs)} samples")
    print(f"  Qwen2.5-7B-Instruct: {len(baseline_outputs)} samples")

    # Run evaluation using paper-revision-bench
    print("\nRunning evaluation with paper-revision-bench...")
    print("(Using Qwen2.5-7B-Instruct as 'original', XtraGPT as 'revised')")

    results = evaluate(
        original_texts=baseline_outputs,  # Baseline model outputs
        revised_texts=xtragpt_outputs,    # XtraGPT outputs (should be better)
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
    print("=" * 60)

    # Save results
    results.to_json("reproduction_results.json")
    print("\nResults saved to reproduction_results.json")


if __name__ == "__main__":
    main()
