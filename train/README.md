# Training XtraGPT

We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for fine-tuning.

## Setup

### 1. Install LLaMA-Factory

```bash
pip install llamafactory
```

### 2. Configure Dataset

Copy the dataset configuration to LLaMA-Factory's data directory:

```bash
cp data/data.json /path/to/LLaMA-Factory/data/dataset_info.json
```

Or merge it with the existing `dataset_info.json` if you have other datasets.

## Training

### Option 1: Use the provided script

```bash
# Set your base model and output directory
export MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
export OUTPUT_DIR="./output/xtragpt-7b"

bash ../scripts/train.sh
```

### Option 2: Use LLaMA-Factory directly

```bash
llamafactory-cli train ../configs/train_config.yaml
```

### Option 3: Custom configuration

Modify `../configs/train_config.yaml` to customize:
- `model_name_or_path`: Base model (e.g., Qwen/Qwen2.5-7B-Instruct)
- `output_dir`: Where to save the trained model
- `num_train_epochs`: Number of training epochs
- `learning_rate`: Learning rate (default: 1e-6)

## Training Configuration

Key hyperparameters used in the paper:

| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-6 |
| Epochs | 4 |
| Batch Size | 1 (per device) |
| Gradient Accumulation | 4 |
| Max Length | 16384 |
| Warmup Ratio | 0.1 |
| Precision | bf16 |

## Dataset

The training data is automatically downloaded from HuggingFace:
- Dataset: [Xtra-Computing/ReviseQA](https://huggingface.co/datasets/Xtra-Computing/ReviseQA)
- Size: ~140,000 instruction-revision pairs
- Format: Alpaca format (instruction, input, output)
