# Paper Revision Bench

A Python package for benchmarking paper revision quality using LLM-as-a-judge.

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
    judge_model="gpt-4-turbo",
    api_key="sk-xxx",  # or set OPENAI_API_KEY env var
)

print(f"Win Rate: {results.win_rate:.1%}")
print(f"Details: {results.summary()}")
```

## Features

- **Multiple Judge Models**: OpenAI, Anthropic, local models via vLLM/Ollama
- **6 Paper Sections**: Title, Abstract, Introduction, Background, Evaluation, Conclusion
- **20 Evaluation Criteria**: Conciseness, clarity, impact, coherence, etc.
- **Async Batch Processing**: Fast evaluation of large datasets
- **Detailed Reports**: Win/lose/tie breakdown, per-sample scores

## Usage

### Basic Evaluation

```python
from paper_revision_bench import evaluate

results = evaluate(
    original_texts=["Original text 1", "Original text 2"],
    revised_texts=["Revised text 1", "Revised text 2"],
    contexts=["Full paper 1", "Full paper 2"],  # optional
    section="abstract",  # title, abstract, introduction, background, evaluation, conclusion
    criterion="conciseness",  # or "clarity", "impact", etc.
    judge_model="gpt-4-turbo",
)
```

### Using Different Judge Models

```python
from paper_revision_bench import evaluate

# OpenAI
results = evaluate(..., judge_model="gpt-4-turbo", api_key="sk-xxx")

# Anthropic
results = evaluate(..., judge_model="claude-3-opus", api_key="sk-ant-xxx")

# Local model via Ollama
results = evaluate(..., judge_model="ollama/llama3:70b", api_base="http://localhost:11434")

# Local model via vLLM
results = evaluate(..., judge_model="vllm/meta-llama/Llama-3-70b", api_base="http://localhost:8000")
```

### Batch Evaluation with Async

```python
from paper_revision_bench import evaluate_async
import asyncio

async def main():
    results = await evaluate_async(
        original_texts=large_original_list,
        revised_texts=large_revised_list,
        judge_model="gpt-4-turbo",
        batch_size=10,
        max_concurrent=5,
    )
    print(results.summary())

asyncio.run(main())
```

### Detailed Reports

```python
results = evaluate(...)

# Summary statistics
print(results.win_rate)        # 0.85
print(results.lose_rate)       # 0.10
print(results.tie_rate)        # 0.05

# Per-sample details
for detail in results.details:
    print(f"Sample {detail.index}: {detail.winner} - {detail.explanation}")

# Export report
results.to_json("report.json")
results.to_csv("report.csv")
```

### Available Criteria

```python
from paper_revision_bench import list_criteria, list_sections

print(list_sections())
# ['title', 'abstract', 'introduction', 'background', 'evaluation', 'conclusion']

print(list_criteria("abstract"))
# ['conciseness', 'clarity', 'impact', 'completeness', 'coherence', ...]
```

## API Reference

### `evaluate()`

Main evaluation function.

**Parameters:**
- `original_texts` (List[str]): Original texts before revision
- `revised_texts` (List[str]): Revised texts after revision
- `contexts` (List[str], optional): Full paper contexts
- `section` (str, optional): Paper section type. Default: "abstract"
- `criterion` (str, optional): Evaluation criterion. Default: "overall"
- `judge_model` (str): Model to use as judge
- `api_key` (str, optional): API key (or use env var)
- `api_base` (str, optional): API base URL for local models
- `temperature` (float): Judge temperature. Default: 0.0
- `max_tokens` (int): Max tokens for judge response. Default: 1024

**Returns:** `EvaluationResult` object

### `EvaluationResult`

**Attributes:**
- `win_rate` (float): Proportion of wins for revised text
- `lose_rate` (float): Proportion of losses
- `tie_rate` (float): Proportion of ties
- `details` (List[SampleResult]): Per-sample results
- `metadata` (dict): Evaluation metadata

**Methods:**
- `summary()`: Return summary string
- `to_json(path)`: Export to JSON
- `to_csv(path)`: Export to CSV

## Citation

If you use this package, please cite:

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
