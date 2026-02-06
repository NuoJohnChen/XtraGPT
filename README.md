# XtraGPT

**XtraGPT** is a family of open-source LLMs for human-AI collaborative academic paper revision.

[![Models](https://img.shields.io/badge/🤗%20Models-XtraGPT-blue)](https://huggingface.co/Xtra-Computing/XtraGPT-7B)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-ReviseQA-green)](https://huggingface.co/datasets/Xtra-Computing/ReviseQA)
[![Paper](https://img.shields.io/badge/📄%20Paper-arXiv-red)](https://arxiv.org/abs/2505.11336)

## Quick Start

### Option 1: Use Pre-trained Models (Recommended)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Xtra-Computing/XtraGPT-7B"  # Also: XtraGPT-1.5B, XtraGPT-3B, XtraGPT-14B

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = """Act as an expert model for improving articles **PAPER_CONTENT**.
The output needs to answer the **QUESTION** on **SELECTED_CONTENT** in the input.
<PAPER_CONTENT>
{your_paper_content}
</PAPER_CONTENT>
<SELECTED_CONTENT>
{text_to_revise}
</SELECTED_CONTENT>
<QUESTION>
{revision_instruction}
</QUESTION>"""

messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.1)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

### Option 2: Train Your Own Model

See [Training Guide](#training).

---

## Table of Contents

- [Installation](#installation)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
  - [Component-wise Evaluation](#component-wise-evaluation)
  - [Full Paper Evaluation](#full-paper-evaluation)
- [Citation](#citation)

---

## Installation

```bash
# Clone repository
git clone https://github.com/Xtra-Computing/XtraGPT.git
cd XtraGPT

# Install dependencies
pip install -r requirements.txt

# For training, also install LLaMA-Factory
pip install llamafactory
```

---

## Training

We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for fine-tuning.

### Step 1: Prepare Dataset Configuration

Copy `configs/dataset_info.json` to your LLaMA-Factory data directory:

```bash
cp configs/dataset_info.json /path/to/LLaMA-Factory/data/
```

### Step 2: Run Training

```bash
# Set environment variables
export MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"  # Base model
export OUTPUT_DIR="./output/xtragpt-7b"        # Output directory

# Run training
bash scripts/train.sh
```

Or use LLaMA-Factory directly:

```bash
llamafactory-cli train configs/train_config.yaml
```

### Training Configuration

Key hyperparameters (from paper):

| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-6 |
| Epochs | 4 |
| Batch Size | 1 (per device) |
| Gradient Accumulation | 4 |
| Max Length | 16384 |
| Warmup Ratio | 0.1 |

---

## Inference

### Batch Inference

```bash
# Set model path (HuggingFace or local)
export MODEL_PATH="Xtra-Computing/XtraGPT-7B"
export INPUT_FILE="./data/test_inputs.jsonl"
export OUTPUT_FILE="./predictions.jsonl"

bash scripts/predict.sh
```

### Input Format

```json
{
  "paper_content": "Full paper text...",
  "selected_content": "Sentence or paragraph to revise",
  "instruction": "Make this more concise"
}
```

---

## Evaluation

### Component-wise Evaluation

Evaluates revisions for 6 paper sections: **Title**, **Abstract**, **Introduction**, **Background**, **Evaluation**, **Conclusion**.

Uses modified [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) for pairwise comparison.

#### Step 1: Setup AlpacaEval

```bash
# Clone and install AlpacaEval
git clone https://github.com/tatsu-lab/alpaca_eval.git
cd alpaca_eval && pip install -e .

# Copy our modified configs
cp -r ../6_component_evaluation/alpaca_eval_gpt4_turbo_fn/* \
    src/alpaca_eval/evaluators_configs/alpaca_eval_gpt4_turbo_fn/
```

#### Step 2: Convert Predictions

```bash
python 6_component_evaluation/convert_predictions.py \
    --input_dir ./predictions \
    --output_dir ./formatted_predictions \
    --model_name "XtraGPT-7B"
```

#### Step 3: Run Evaluation

```bash
export OPENAI_API_KEY="your-api-key"

bash 6_component_evaluation/run_eval.sh \
    ./formatted_predictions/xtragpt \
    ./formatted_predictions/baseline \
    ./eval_results
```

### Full Paper Evaluation

Uses [AI-Scientist](https://github.com/SakanaAI/AI-Scientist) to evaluate entire papers.

#### Setup

```bash
git clone https://github.com/SakanaAI/AI-Scientist.git
cd AI-Scientist && pip install -e .
```

#### Run Evaluation

```bash
export OPENAI_API_KEY="your-api-key"

python full_paper_evaluation/ai_scientist_eval.py \
    --paper_path ./papers/my_paper.pdf \
    --output ./review_results.json \
    --model "gpt-4o"
```

---

## Project Structure

```
XtraGPT/
├── configs/
│   ├── train_config.yaml      # Training configuration
│   └── dataset_info.json      # Dataset configuration for LLaMA-Factory
├── scripts/
│   ├── train.sh               # Training script
│   └── predict.sh             # Inference script
├── 6_component_evaluation/    # Component-wise evaluation
│   ├── alpaca_eval_gpt4_turbo_fn/
│   ├── convert_predictions.py
│   └── run_eval.sh
├── full_paper_evaluation/     # Full paper evaluation
│   ├── ai_scientist_eval.py
│   ├── analyze_results.py
│   └── paper_results/
├── train/
│   ├── data/
│   └── README.md
├── examples/
│   └── inference_example.py
├── requirements.txt
└── README.md
```

---

## Model Zoo

| Model | Size | HuggingFace |
|-------|------|-------------|
| XtraGPT-1.5B | 1.5B | [Link](https://huggingface.co/Xtra-Computing/XtraGPT-1.5B) |
| XtraGPT-3B | 3B | [Link](https://huggingface.co/Xtra-Computing/XtraGPT-3B) |
| XtraGPT-7B | 7B | [Link](https://huggingface.co/Xtra-Computing/XtraGPT-7B) |
| XtraGPT-14B | 14B | [Link](https://huggingface.co/Xtra-Computing/XtraGPT-14B) |

---

## Citation

```bibtex
@misc{nuo2025xtragpt,
      title={XtraGPT: LLMs for Human-AI Collaboration on Controllable Academic Paper Revision},
      author={Nuo Chen and Andre Lin HuiKai and Jiaying Wu and Junyi Hou and Zining Zhang and Qian Wang and Xidong Wang and Bingsheng He},
      year={2025},
      eprint={2505.11336},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.11336},
}
```

## License

This project is released under the [ModelGo Zero License 2.0 (MG0-2.0)](https://www.modelgo.li/).

## Acknowledgements

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval)
- [AI-Scientist](https://github.com/SakanaAI/AI-Scientist)
