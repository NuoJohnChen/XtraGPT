# XtraGPT

[![Models](https://img.shields.io/badge/🤗%20Models-XtraGPT-blue)](https://huggingface.co/Xtra-Computing/XtraGPT-7B)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-ReviseQA-green)](https://huggingface.co/datasets/Xtra-Computing/ReviseQA)
[![Paper](https://img.shields.io/badge/📄%20Paper-arXiv-red)](https://arxiv.org/abs/2505.11336)

## Overview

**XtraGPT** is a family of open-source Large Language Models (LLMs) designed specifically for **human-AI collaborative academic paper revision**. Unlike general-purpose models that often perform surface-level polishing, XtraGPT is fine-tuned to **understand the full context** of a research paper and execute specific, **criteria-guided** revision instructions. XtraGPT is the refiner of [Friend Project: PaperDebugger](https://github.com/PaperDebugger/paperdebugger)

The models were trained on a dataset of 140,000 high-quality instruction-revision pairs derived from top-tier conference papers (ICLR).

**Key Features:**

* **Context-Aware:** Processes the full paper context to ensure revisions maintain consistency with the global narrative.
* **Controllable:** Follows specific user instructions aligned with 20 academic writing criteria across 6 sections (Abstract, Introduction, etc.).
* **Iterative Workflow:** Designed to support the "Human-AI Collaborative" (HAC) lifecycle where authors retain creative control.

---

## Inference with Transformers

To use XtraGPT with the standard Hugging Face `transformers` library, ensure you format your input using the specific tags `<PAPER_CONTENT>`, `<SELECTED_CONTENT>`, and `<QUESTION>`.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Select the model size: "XtraGPT-1.5B", "XtraGPT-3B", "XtraGPT-7B", or "XtraGPT-14B"
model_name = "Xtra-Computing/XtraGPT-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Define the Prompt Template tailored for XtraGPT
prompt_template = """Act as an expert model for improving articles **PAPER_CONTENT**.
The output needs to answer the **QUESTION** on **SELECTED_CONTENT** in the input. Avoid adding unnecessary length, unrelated details, overclaims, or vague statements.
Focus on clear, concise, and evidence-based improvements that align with the overall context of the paper.
<PAPER_CONTENT>
{paper_content}
</PAPER_CONTENT>
<SELECTED_CONTENT>
{selected_content}
</SELECTED_CONTENT>
<QUESTION>
{user_question}
</QUESTION>"""

# Example Data (from the "Attention Is All You Need" paper)
paper_content = "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train."
selected_content = "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration."
user_question = "help me make it more concise."

# Format the input
formatted_prompt = prompt_template.format(
    paper_content=paper_content,
    selected_content=selected_content,
    user_question=user_question
)

messages = [
    {"role": "user", "content": formatted_prompt}
]

# Apply chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Generate
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=16384,
    temperature=0.1
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

---

## Table of Contents

- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
  - [Component-wise Evaluation](#component-wise-evaluation)
  - [Full Paper Evaluation](#full-paper-evaluation)
- [Model Zoo](#model-zoo)
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

---

## Model License

This model is released under the **ModelGo Zero License 2.0 (MG0-2.0)**.

MG0-2.0 is a highly permissive open model license designed to facilitate the widest possible adoption and collaboration. It allows for **unrestricted use**, reproduction, distribution, and the creation of derivative works including for commercial purposes, without requiring attribution or imposing copyleft restrictions.

For more details on the license terms, please visit [ModelGo.li](https://www.modelgo.li/) or refer to the `LICENSE` file in the repository.

---

## Acknowledgements

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval)
- [AI-Scientist](https://github.com/SakanaAI/AI-Scientist)
