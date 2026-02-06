#!/bin/bash
# XtraGPT Training Script
# Usage: bash scripts/train.sh

set -e

# Configuration (override via environment variables)
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/xtragpt}"
CONFIG_FILE="${CONFIG_FILE:-configs/train_config.yaml}"

echo "============================================"
echo "XtraGPT Training"
echo "============================================"
echo "Base Model: $MODEL_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "Config: $CONFIG_FILE"
echo "============================================"

# Check if LLaMA-Factory is installed
if ! command -v llamafactory-cli &> /dev/null; then
    echo "Error: LLaMA-Factory not found."
    echo "Please install it first: pip install llamafactory"
    exit 1
fi

# Copy dataset config to LLaMA-Factory data directory
LLAMA_FACTORY_PATH=$(python -c "import llamafactory; print(llamafactory.__path__[0])" 2>/dev/null || echo "")
if [ -n "$LLAMA_FACTORY_PATH" ]; then
    DATA_DIR="$(dirname $LLAMA_FACTORY_PATH)/data"
    if [ -d "$DATA_DIR" ]; then
        cp configs/dataset_info.json "$DATA_DIR/"
        echo "Copied dataset_info.json to $DATA_DIR"
    fi
fi

# Run training
llamafactory-cli train "$CONFIG_FILE" \
    --model_name_or_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR"

echo "============================================"
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "============================================"
