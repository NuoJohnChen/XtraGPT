#!/bin/bash

API_KEY=...

# MY_MODEL="XtraGPT-7B"
# MY_MODEL=Qwen2.5-7B-Instruct-paperfull214/checkpoint-3000-new
# MY_MODEL="phi3.5-3.8b-paperfullall-new/checkpoint-3000"
# MY_MODEL="phi4-14b-paperfullall-new/checkpoint-1000"
MY_MODEL="experimental/Qwen2.5-7B-Instruct-paperfull214/checkpoint-3000/3500_reduced_Qwen2.5-7B-Instruct-paperfull214"
# MY_MODEL="XtraGPT-1.5B"
# MY_MODEL="Qwen2.5-7B-Instruct"
# MY_MODEL="DeepSeek-R1-Distill-Qwen-7B"
INPUT_BASE_DIR="/shared/hdd/andre/predictions/formatted"
OUTPUT_BASE_DIR="/shared/hdd/andre/results/$MY_MODEL"
LOG_PATH="/shared/hdd/andre/alpaca_evaluation/scorer_ep1_log.txt"

# models=(DeepSeek-R1-Distill-Llama-8B)
# models=(Llama-3.1-8B-Instruct Qwen2.5-7B-Instruct Qwen2-72B-Instruct QwQ-32B-Preview Yi-1.5-9B-Chat Llama-3.2-3B-Instruct deepseek-v3-671b)
# models=(Qwen2.5-7B-Instruct-paperfull214/checkpoint-62 Qwen2.5-7B-Instruct-paperfull214/checkpoint-124 Qwen2.5-7B-Instruct-paperfull214/checkpoint-186 Qwen2.5-7B-Instruct-paperfull214/checkpoint-248 Qwen2.5-7B-Instruct-paperfull214/checkpoint-310 Qwen2.5-7B-Instruct-paperfull214/checkpoint-372 Qwen2.5-7B-Instruct-paperfull214/checkpoint-434 Qwen2.5-7B-Instruct-paperfull214/checkpoint-496 Qwen2.5-7B-Instruct-paperfull214/checkpoint-558 Qwen2.5-7B-Instruct-paperfull214/checkpoint-610)
# models=(Qwen2.5-7B-Instruct-paperfull214/checkpoint-434)
# models=(DeepSeek-R1-Distill-Qwen-7B)
#  Qwen2.5-7B-Instruct-paperfull214/checkpoint-496
# models=(Phi-3.5-mini-instruct)
# models=(phi-4)
# models=(DeepSeek-R1-Distill-Llama-8B)
models=(experimental/original_text/3500_reduced_original_text experimental/qwen2.5-72b-instruct/3500_reduced_qwen2.5-72b-instruct)
# models=(DeepSeek-R1-Distill-Llama-8B)
# models=(gpt-4o-mini)
# models=(phi4-14b-paperfullall-new/checkpoint-228 phi4-14b-paperfullall-new/checkpoint-399 phi4-14b-paperfullall-new/checkpoint-513)
# models=(Qwen2.5-7B-Instruct-paperfull214/checkpoint-62)
# models=(phi4-14b-paperfullall-new/checkpoint-228 phi4-14b-paperfullall-new/checkpoint-399 phi4-14b-paperfullall-new/checkpoint-513)
# models=(phi4-14b-paperfullall-new/checkpoint-171 phi4-14b-paperfullall-new/checkpoint-228 phi4-14b-paperfullall-new/checkpoint-285 phi4-14b-paperfullall-new/checkpoint-342 phi4-14b-paperfullall-new/checkpoint-399 phi4-14b-paperfullall-new/checkpoint-456 phi4-14b-paperfullall-new/checkpoint-513 phi4-14b-paperfullall-new/checkpoint-570 phi4-14b-paperfullall-new/checkpoint-627 phi4-14b-paperfullall-new/checkpoint-684 phi4-14b-paperfullall-new/checkpoint-114)
# models=(phi4-14b-paperfullall-new/checkpoint-114  phi4-14b-paperfullall-new/checkpoint-342 phi4-14b-paperfullall-new/checkpoint-399 phi4-14b-paperfullall-new/checkpoint-456 phi4-14b-paperfullall-new/checkpoint-513 phi4-14b-paperfullall-new/checkpoint-570 phi4-14b-paperfullall-new/checkpoint-627 phi4-14b-paperfullall-new/checkpoint-684)
sections=(abstract background conclusion evaluation introduction title)

if [ ! -d "$OUTPUT_BASE_DIR" ]; then
    mkdir -p "$OUTPUT_BASE_DIR"
    chmod -R a+rwx "$OUTPUT_BASE_DIR"
fi

# export OPENAI_API_KEY="$API_KEY"

export OPENAI_API_KEY=...

# iterate through placeholders
for model in "${models[@]}"
do
    echo "Comparing against [${model}].."
    echo -e "\n[$(date)] Comparing against [${model}].." >> "$LOG_PATH"
    OUTPUT_MODEL_DIR="$OUTPUT_BASE_DIR/$model"
    # ensure output directory exists
    mkdir -p "$OUTPUT_MODEL_DIR"
    chmod -R a+rwx "$OUTPUT_MODEL_DIR"

    for section_name in "${sections[@]}"
    do
        python ./change_to_json_new.py --section $section_name
        
        FINETUNED_MODEL_PREDICTION="$INPUT_BASE_DIR/$MY_MODEL/$section_name.json"
        OTHER_MODEL_PREDICTION="$INPUT_BASE_DIR/$model/$section_name.json"
        OUTPUT_PATH="$OUTPUT_MODEL_DIR/$section_name"

        # ensure output directory exists
        mkdir -p "$OUTPUT_PATH"
        chmod -R a+rwx "$OUTPUT_PATH"

        # Check if the prediction files exist
        if [ ! -f "$FINETUNED_MODEL_PREDICTION" ]; then
            echo "[$(date)] Error: Fine-tuned model $section_name prediction file not found for $MY_MODEL." >> "$LOG_PATH"
            continue
        fi
        
        if [ ! -f "$OTHER_MODEL_PREDICTION" ]; then
            echo "[$(date)] Error: Against model $section_name prediction file not found for $model." >> "$LOG_PATH"
            continue
        fi

        echo "[$(date)] Running comparison on $section_name.." >> "$LOG_PATH"

        PYTHONPATH="/YOUR/PATH/TO/alpaca_eval" python -m src.alpaca_eval.main evaluate \
            --model_outputs "$OTHER_MODEL_PREDICTION" \
            --reference_outputs "$FINETUNED_MODEL_PREDICTION" \
            --annotators_config alpaca_eval_gpt4_turbo_fn \
            --output_path "$OUTPUT_PATH" \
            --metric_kwargs "{'glm_name':'length_controlled_v1'}" 

        # check success
        if [ $? -ne 0 ]; then
            echo "Error: Command for $model failed. Logging error and continuing."
            echo "[$(date)] Error: Command for $model failed." >> "$LOG_PATH"
            continue
        else
            echo "[$(date)] Success: Scoring completed against $model for $section_name. Output saved to $OUTPUT_PATH." >> "$LOG_PATH"
        fi
    done
done

echo "All commands executed. Check $LOG_PATH for more details."

# alpaca_eval evaluate \
#     --model_outputs /shared/hdd/andre/predictions/formatted/temp/Qwen2.5-7B-Instruct-paperall6687-epoch2-sft.json \
#     --reference_outputs /shared/hdd/andre/predictions/formatted/temp/Llama-3.1-8B-Instruct.json \
#     --annotators_config alpaca_eval_gpt4_turbo_fn \
#     --output_path /shared/hdd/andre/predictions/formatted/temp/Qwen2.5-7B-Instruct-epoch2-paperall6687-sft
