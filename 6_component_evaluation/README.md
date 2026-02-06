# Component-wise Evaluation

This module evaluates paper revisions for 6 sections using a modified [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) framework.

## Sections Evaluated

| Section | Description |
|---------|-------------|
| Title | Paper title improvements |
| Abstract | Abstract clarity and impact |
| Introduction | Introduction structure and flow |
| Background | Related work and context |
| Evaluation | Experimental methodology |
| Conclusion | Summary and future work |

## Setup

### 1. Install AlpacaEval

```bash
git clone https://github.com/tatsu-lab/alpaca_eval.git
cd alpaca_eval
pip install -e .
```

### 2. Copy Modified Configs

Replace the default AlpacaEval configs with our modified versions:

```bash
cp -r alpaca_eval_configs/* \
    /path/to/alpaca_eval/src/alpaca_eval/evaluators_configs/alpaca_eval_gpt4_turbo_fn/
```

### 3. Set API Key

```bash
export OPENAI_API_KEY="your-api-key"
```

## Usage

### Step 1: Generate Predictions

First, run inference with your model to generate predictions:

```bash
# Using XtraGPT
MODEL_PATH="Xtra-Computing/XtraGPT-7B" bash ../../scripts/predict.sh
```

This creates `*_predictions.jsonl` files for each section.

### Step 2: Convert to AlpacaEval Format

```bash
python convert_predictions.py \
    --input_dir /path/to/predictions \
    --output_dir /path/to/formatted \
    --model_name "XtraGPT-7B"
```

### Step 3: Run Evaluation

```bash
bash run_eval.sh \
    /path/to/model/formatted \
    /path/to/baseline/formatted \
    /path/to/output
```

## Output Format

Results are saved in JSON format:

```json
{
  "win_rate": 65.2,
  "standard_error": 1.3,
  "n_total": 1400
}
```

## Modified Prompts

We modified the AlpacaEval prompts to focus on academic writing quality:

- `alpaca_eval_fn_abstract.txt` - Abstract-specific evaluation criteria
- `alpaca_eval_fn_introduction.txt` - Introduction evaluation criteria
- etc.

See `alpaca_eval_configs/` for all modified prompts.
