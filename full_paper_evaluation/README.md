# Full Paper Evaluation

This module evaluates entire papers using the [AI-Scientist](https://github.com/SakanaAI/AI-Scientist) framework.

## Overview

AI-Scientist provides an LLM-based paper review system that evaluates:

- **Overall Score**: General paper quality (1-10)
- **Soundness**: Technical correctness (1-4)
- **Presentation**: Writing quality (1-4)
- **Contribution**: Novelty and significance (1-4)
- **Decision**: Accept/Reject recommendation

## Setup

### 1. Install AI-Scientist

```bash
git clone https://github.com/SakanaAI/AI-Scientist.git
cd AI-Scientist
pip install -e .
```

### 2. Set API Key

```bash
export OPENAI_API_KEY="your-api-key"
```

## Usage

### Evaluate a Single Paper

```bash
python ai_scientist_eval.py \
    --paper_path ./my_paper.pdf \
    --output ./review.json \
    --model gpt-4o
```

### Analyze Before/After Improvements

```bash
python analyze_results.py \
    --before before_revision_scores.json \
    --after after_revision_scores.json \
    --output analysis.json
```

## Pre-computed Results

We include evaluation results from our paper:

| File | Description |
|------|-------------|
| `before_revision_scores.json` | Scores before XtraGPT revision |
| `after_revision_scores.json` | Scores after XtraGPT revision |
| `decisions_before_xtragpt.json` | Accept/Reject decisions before |
| `decisions_after_xtragpt.json` | Accept/Reject decisions after |

### Key Findings

From our evaluation on 26 ICLR papers:

- **Acceptance Rate**: Improved from X% to Y%
- **Presentation Score**: +0.X average improvement
- **Overall Score**: +0.X average improvement

## Validation

We validated AI-Scientist's reliability by comparing its predictions against actual ICLR decisions. See `validate_ai_scientist.py` for the validation code.
