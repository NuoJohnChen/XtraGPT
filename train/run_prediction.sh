#!/bin/bash

MODEL_NAME="Qwen2-72B-Instruct" # model name, e.g., Llama-3.2-3B-Instruct
TEMPLATE="qwen" # template name, e.g., llama3; check llamafactory readme for mapping

MODEL_PATH="/shared/ssd/models/$MODEL_NAME"
OUTPUT_BASE_PATH="/shared/hdd/andre/predictions/experimental/$MODEL_NAME"
ERROR_LOG_PATH="/shared/hdd/andre/predictions/error_log.txt"
CUDA_VISIBLE_DEVICES="0,2,3,4"

# 6 evlauation metrics
placeholders=(abstract background conclusion evaluation introduction title)

echo -e "\n[$(date)] New run started for [${MODEL_NAME}].." >> "$ERROR_LOG_PATH"

if [ ! -d "$OUTPUT_BASE_PATH" ]; then
    mkdir -p "$OUTPUT_BASE_PATH"
fi
# permissions
chmod -R a+rwx "$OUTPUT_BASE_PATH"

# iterate through placeholders
for placeholder in "${placeholders[@]}"
do
    DATASET="${placeholder}_custom_data_list_noexplain"
    OUTPUT_PATH="$OUTPUT_BASE_PATH/${placeholder}_predictions.jsonl"

    echo "Running inference for $placeholder..."
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python scripts/vllm_infer.py \
        --model_name_or_path "$MODEL_PATH" \
        --dataset "$DATASET" \
        --template "$TEMPLATE" \
        --save_name "$OUTPUT_PATH" \
        --max_new_tokens 2048

    # check success
    if [ $? -ne 0 ]; then
        echo "Error: Command for $placeholder failed. Logging error and continuing."
        echo "[$(date)] Error: Command for $placeholder failed." >> "$ERROR_LOG_PATH"
        continue
    else
        echo -e "[$(date)] Completed run for [${MODEL_NAME}] on $placeholder" >> "$ERROR_LOG_PATH"
    fi

done

echo "All commands executed. Check $ERROR_LOG_PATH for any errors."




# MODEL_NAME="Llama-3.2-3B-Instruct" # model name, e.g., Llama-3.2-3B-Instruct
# TEMPLATE="llama3" # template name, e.g., llama3; check llamafactory readme for mapping

# MODEL_PATH="/disk1/nuo/models/$MODEL_NAME"
# OUTPUT_BASE_PATH="/shared/hdd/andre/predictions/experimental/$MODEL_NAME"
# ERROR_LOG_PATH="/shared/hdd/andre/predictions/error_log.txt"
# CUDA_VISIBLE_DEVICES="4,5,6,7"

# # 6 evlauation metrics
# placeholders=(abstract background conclusion evaluation introduction title)

# echo -e "\n[$(date)] New run started for [${MODEL_NAME}].." >> "$ERROR_LOG_PATH"

# if [ ! -d "$OUTPUT_BASE_PATH" ]; then
#     mkdir -p "$OUTPUT_BASE_PATH"
# fi
# # permissions
# chmod -R a+rwx "$OUTPUT_BASE_PATH"

# # iterate through placeholders
# for placeholder in "${placeholders[@]}"
# do
#     DATASET="${placeholder}_custom_data_list_noexplain"
#     OUTPUT_PATH="$OUTPUT_BASE_PATH/${placeholder}_predictions.jsonl"

#     echo "Running inference for $placeholder..."
#     CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python scripts/vllm_infer.py \
#         --model_name_or_path "$MODEL_PATH" \
#         --dataset "$DATASET" \
#         --template "$TEMPLATE" \
#         --save_name "$OUTPUT_PATH"

#     # check success
#     if [ $? -ne 0 ]; then
#         echo "Error: Command for $placeholder failed. Logging error and continuing."
#         echo "[$(date)] Error: Command for $placeholder failed." >> "$ERROR_LOG_PATH"
#         continue
#     else
#         echo -e "[$(date)] Completed run for [${MODEL_NAME}] on $placeholder" >> "$ERROR_LOG_PATH"
#     fi

# done

# echo "All commands executed. Check $ERROR_LOG_PATH for any errors."