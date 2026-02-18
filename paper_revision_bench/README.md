# Paper Revision Bench

A Python package for benchmarking paper revision quality, using the exact evaluation methodology from the XtraGPT paper.

[![PyPI version](https://badge.fury.io/py/paper-revision-bench.svg)](https://badge.fury.io/py/paper-revision-bench)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Installation

```bash
pip install paper-revision-bench
```

## Quick Start

```python
from paper_revision_bench import evaluate

results = evaluate(
    original_texts=["The dominant sequence transduction models are based on complex recurrent or convolutional neural networks."],
    revised_texts=["Sequence transduction models typically use complex RNNs or CNNs."],
    section="abstract",
    judge_model="gpt-4-1106-preview",  # matches paper
    api_key="sk-xxx",  # or set OPENAI_API_KEY env var
)

print(f"Win Rate: {results.win_rate:.1%}")
print(results.summary())
```

## How It Works

The evaluation uses AlpacaEval's function calling format, exactly as in the paper:

1. GPT-4-Turbo receives both texts and ranks them via the `make_partial_leaderboard` function call
2. Section-specific criteria (20 total across 6 sections) guide the ranking
3. The result is a win/lose/tie for the revised text vs the original

`original_texts` maps to model `"m"` (output_1), `revised_texts` maps to model `"M"` (output_2).

## Usage

### Basic Evaluation

```python
from paper_revision_bench import evaluate

results = evaluate(
    original_texts=["Original text 1", "Original text 2"],
    revised_texts=["Revised text 1", "Revised text 2"],
    instructions=["Improve clarity", "Improve clarity"],  # optional
    section="abstract",  # title, abstract, introduction, background, evaluation, conclusion
    judge_model="gpt-4-1106-preview",
)
```

### Async Batch Evaluation

```python
from paper_revision_bench import evaluate_async
import asyncio

async def main():
    results = await evaluate_async(
        original_texts=large_original_list,
        revised_texts=large_revised_list,
        section="introduction",
        max_concurrent=5,
    )
    print(results.summary())

asyncio.run(main())
```

### Detailed Reports

```python
results = evaluate(...)

print(results.win_rate)    # 0.85
print(results.lose_rate)   # 0.10
print(results.tie_rate)    # 0.05

for detail in results.details:
    print(f"Sample {detail.index}: {detail.winner} - {detail.explanation}")

results.to_json("report.json")
results.to_csv("report.csv")
```

### Weighted Overall Score

The paper computes an overall win rate as a weighted average across 6 sections (title:abstract:introduction:background:evaluation:conclusion = 2:4:6:3:3:2):

```python
from paper_revision_bench import evaluate, compute_weighted_overall

section_results = {}
for section in ["title", "abstract", "introduction", "background", "evaluation", "conclusion"]:
    section_results[section] = evaluate(
        original_texts=original_list,
        revised_texts=revised_list,
        section=section,
    )

overall = compute_weighted_overall(section_results)
print(f"Overall Win Rate: {overall['weighted_win_rate']:.1%}")
```

### Length-Controlled Win Rate

To reproduce the paper's length-controlled win rate (corrects for length bias via GLM):

```bash
pip install paper-revision-bench[alpaca]
```

```python
lc = results.length_controlled_winrate(model_name="XtraGPT-7B", baseline_name="original")
print(f"LC Win Rate: {lc['length_controlled_winrate']:.1f}% ± {lc['lc_standard_error']:.1f}%")
```

First call downloads ~50KB of data from HuggingFace to `~/.cache/alpaca_eval/`.

## API Reference

### `evaluate()`

**Parameters:**
- `original_texts` (List[str]): Baseline texts (model "m" / output_1)
- `revised_texts` (List[str]): Model outputs to evaluate (model "M" / output_2)
- `instructions` (List[str], optional): Revision instructions per sample
- `section` (str): Paper section. Default: `"abstract"`
- `judge_model` (str): OpenAI model. Default: `"gpt-4-1106-preview"` (matches paper)
- `api_key` (str, optional): OpenAI API key (or set `OPENAI_API_KEY`)
- `temperature` (float): Default: `0.0`
- `max_tokens` (int): Default: `200` (matches paper)
- `max_concurrent` (int): Concurrent API calls. Default: `5`

**Returns:** `EvaluationResult`

### `EvaluationResult`

**Attributes:** `win_rate`, `lose_rate`, `tie_rate`, `average_score`, `n_wins`, `n_losses`, `n_ties`, `total`, `details`, `metadata`

**Methods:** `summary()`, `to_json(path)`, `to_csv(path)`, `length_controlled_winrate(...)`

## Citation

```bibtex
@misc{nuo2025xtragpt,
      title={XtraGPT: LLMs for Human-AI Collaboration on Controllable Academic Paper Revision},
      author={Nuo Chen and Andre Lin HuiKai and Jiaying Wu and Junyi Hou and Zining Zhang and Qian Wang and Xidong Wang and Bingsheng He},
      year={2025},
      eprint={2505.11336},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
}
```

## License

Apache 2.0
