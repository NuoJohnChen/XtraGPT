"""
Convert model predictions to AlpacaEval format.

Usage:
    python convert_predictions.py --input_dir ./predictions --output_dir ./formatted --model_name XtraGPT-7B
"""
import argparse
import json
import re
from pathlib import Path


def clean_response(text: str) -> str:
    """Remove thinking tags from response (for reasoning models)."""
    pattern = re.compile(r"<think>.*?</think>\n\n", re.DOTALL)
    return pattern.sub("", text).strip()


def convert_section(
    input_file: Path,
    prompt_file: Path,
    output_file: Path,
    model_name: str
) -> int:
    """Convert a single section's predictions to AlpacaEval format."""
    results = []

    # Load prompts (instructions)
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = json.load(f)

    # Load predictions
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line)
                entry = {
                    "instruction": prompts[idx] if idx < len(prompts) else data.get("prompt", ""),
                    "input": "",
                    "generator": model_name,
                    "output": clean_response(data.get("predict", ""))
                }
                results.append(entry)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {idx}: {e}")
                continue

    # Save formatted output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return len(results)


def main():
    parser = argparse.ArgumentParser(description="Convert predictions to AlpacaEval format")
    parser.add_argument("--input_dir", type=Path, required=True,
                        help="Directory containing *_predictions.jsonl files")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Output directory for formatted JSON files")
    parser.add_argument("--prompt_dir", type=Path, default=None,
                        help="Directory containing prompt JSON files (optional)")
    parser.add_argument("--model_name", type=str, default="XtraGPT",
                        help="Model name for generator field")
    args = parser.parse_args()

    sections = ['abstract', 'background', 'conclusion', 'evaluation', 'introduction', 'title']

    print(f"Converting predictions from: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model name: {args.model_name}")
    print("-" * 50)

    total = 0
    for section in sections:
        input_file = args.input_dir / f"{section}_predictions.jsonl"
        output_file = args.output_dir / f"{section}.json"

        if not input_file.exists():
            print(f"Warning: {input_file} not found, skipping {section}")
            continue

        # Use prompt file if provided
        if args.prompt_dir:
            prompt_file = args.prompt_dir / f"{section}.json"
        else:
            prompt_file = None

        if prompt_file and prompt_file.exists():
            count = convert_section(input_file, prompt_file, output_file, args.model_name)
        else:
            # Fallback: extract prompts from prediction file
            results = []
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    results.append({
                        "instruction": data.get("prompt", ""),
                        "input": "",
                        "generator": args.model_name,
                        "output": clean_response(data.get("predict", ""))
                    })

            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            count = len(results)

        print(f"  {section}: {count} samples -> {output_file}")
        total += count

    print("-" * 50)
    print(f"Total: {total} samples converted")


if __name__ == "__main__":
    main()
