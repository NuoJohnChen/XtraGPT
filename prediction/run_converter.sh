#!/bin/bash

INPUT_BASE_DIR="/shared/hdd/andre/predictions"
LOG_PATH="/shared/hdd/andre/alpaca_evaluation/converter_log.txt"
OUTPUT_BASE_PATH="$INPUT_BASE_DIR/formatted"
DATASET_PROPORTION=1

# models=(Llama-3.1-8B-Instruct Qwen2.5-7B-Instruct-paperall6687-epoch2-sft Qwen2.5-7B-Instruct-paperall6687-sft Qwen2.5-7B-Instruct Llama-3.2-3B-Instruct Qwen2-72B-Instruct QwQ-32B-Preview WhizReviewer-ML-Llama3.1-8B Yi-1.5-9B-Chat deepseek-v3-671b)
# models=(Qwen2.5-7B-Instruct Llama-3.2-3B-Instruct Qwen2-72B-Instruct QwQ-32B-Preview WhizReviewer-ML-Llama3.1-8B Yi-1.5-9B-Chat)
# models=(phi4-14b-paperfullall-new/checkpoint-57 phi4-14b-paperfullall-new/checkpoint-114 phi4-14b-paperfullall-new/checkpoint-171 phi4-14b-paperfullall-new/checkpoint-228 phi4-14b-paperfullall-new/checkpoint-285 phi4-14b-paperfullall-new/checkpoint-342 phi4-14b-paperfullall-new/checkpoint-399 phi4-14b-paperfullall-new/checkpoint-456 phi4-14b-paperfullall-new/checkpoint-513 phi4-14b-paperfullall-new/checkpoint-570 phi4-14b-paperfullall-new/checkpoint-627 phi4-14b-paperfullall-new/checkpoint-684 phi4-14b-paperfullall-new/checkpoint-741 phi4-14b-paperfullall-new/checkpoint-798)
# models=(Qwen2.5-7B-Instruct-paperfull214/checkpoint-62 Qwen2.5-7B-Instruct-paperfull214/checkpoint-124 Qwen2.5-7B-Instruct-paperfull214/checkpoint-186 Qwen2.5-7B-Instruct-paperfull214/checkpoint-248 Qwen2.5-7B-Instruct-paperfull214/checkpoint-310 Qwen2.5-7B-Instruct-paperfull214/checkpoint-372 Qwen2.5-7B-Instruct-paperfull214/checkpoint-434 Qwen2.5-7B-Instruct-paperfull214/checkpoint-496 Qwen2.5-7B-Instruct-paperfull214/checkpoint-558 Qwen2.5-7B-Instruct-paperfull214/checkpoint-610)
# models=(Qwen2.5-7B-Instruct-paperfull214/checkpoint-434 Qwen2.5-7B-Instruct-paperfull214/checkpoint-496)
# models=(phi4-14b-paperfullall-new/checkpoint-399 phi4-14b-paperfullall-new/checkpoint-228 phi4-14b-paperfullall-new/checkpoint-513)
# models=(Llama-3.1-8B-Instruct DeepSeek-R1-Distill-Qwen-7B)
# models=(phi4-14b-paperfullall-new/checkpoint-1000)
# models=(experimental/deepseek-v3-671b/3500_reduced_deepseek-v3-671b)
# models=(experimental/original_text/3500_reduced_original_text)
models=(experimental/qwen2.5-72b-instruct/3500_reduced_qwen2.5-72b-instruct)
if [ ! -d "$OUTPUT_BASE_PATH" ]; then
    mkdir -p "$OUTPUT_BASE_PATH"
    chmod -R a+rwx "$OUTPUT_BASE_PATH"
fi

# iterate through placeholders
for model in "${models[@]}"
do
    echo "Converting for [${model}].."
    echo -e "\n[$(date)] Converting for for [${model}].." >> "$LOG_PATH"
    INPUT_DIR="$INPUT_BASE_DIR/$model"
    OUTPUT_PATH="$OUTPUT_BASE_PATH/$model"

    if [ ! -d "$INPUT_DIR" ]; then
        echo "[$(date)] Error: Input directory not found for $model." >> "$LOG_PATH"
        continue
    fi

    python convert.py \
        --input_base_directory "$INPUT_DIR" \
        --output_dir "$OUTPUT_PATH" \
        --dataset_proportion $DATASET_PROPORTION

    # check success
    if [ $? -ne 0 ]; then
        echo "Error: Command for $model failed. Logging error and continuing."
        echo "[$(date)] Error: Command for $model failed." >> "$LOG_PATH"
        continue
    else
        echo "[$(date)] Success: Conversion completed for $model. Output saved to $OUTPUT_PATH." >> "$LOG_PATH"
    fi

done

echo "All commands executed. Check $LOG_PATH for more details."

