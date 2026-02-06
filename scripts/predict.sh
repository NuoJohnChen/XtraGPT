#!/bin/bash
# XtraGPT Batch Inference Script
# Usage: bash scripts/predict.sh [input_file] [output_dir]

set -e

# Configuration
MODEL_PATH="${MODEL_PATH:-Xtra-Computing/XtraGPT-7B}"
TEMPLATE="${TEMPLATE:-qwen}"
INPUT_FILE="${1:-}"
OUTPUT_DIR="${2:-./predictions}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"

# Sections for evaluation
SECTIONS=(abstract background conclusion evaluation introduction title)

echo "============================================"
echo "XtraGPT Inference"
echo "============================================"
echo "Model: $MODEL_PATH"
echo "Template: $TEMPLATE"
echo "Output Dir: $OUTPUT_DIR"
echo "============================================"

# Check if LLaMA-Factory is installed
if ! command -v llamafactory-cli &> /dev/null; then
    echo "Error: LLaMA-Factory not found."
    echo "Please install it first: pip install llamafactory"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# If specific input file provided, run single inference
if [ -n "$INPUT_FILE" ]; then
    echo "Running inference on: $INPUT_FILE"
    OUTPUT_FILE="$OUTPUT_DIR/predictions.jsonl"

    python -c "
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = '$MODEL_PATH'
input_file = '$INPUT_FILE'
output_file = '$OUTPUT_FILE'
max_new_tokens = $MAX_NEW_TOKENS

print(f'Loading model: {model_path}')
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map='auto'
)

prompt_template = '''Act as an expert model for improving articles **PAPER_CONTENT**.
The output needs to answer the **QUESTION** on **SELECTED_CONTENT** in the input.
<PAPER_CONTENT>
{paper_content}
</PAPER_CONTENT>
<SELECTED_CONTENT>
{selected_content}
</SELECTED_CONTENT>
<QUESTION>
{instruction}
</QUESTION>'''

results = []
with open(input_file, 'r') as f:
    for idx, line in enumerate(f):
        data = json.loads(line)
        prompt = prompt_template.format(
            paper_content=data.get('paper_content', ''),
            selected_content=data.get('selected_content', ''),
            instruction=data.get('instruction', '')
        )

        messages = [{'role': 'user', 'content': prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors='pt').to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.1)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        results.append({
            'prompt': prompt,
            'predict': response,
            'label': data.get('label', '')
        })

        if (idx + 1) % 10 == 0:
            print(f'Processed {idx + 1} samples')

with open(output_file, 'w') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f'Results saved to {output_file}')
"
    exit 0
fi

# Otherwise, run inference for all sections (evaluation mode)
echo "Running inference for all sections..."

for section in "${SECTIONS[@]}"; do
    DATASET="${section}_test"
    OUTPUT_FILE="$OUTPUT_DIR/${section}_predictions.jsonl"

    echo "Processing section: $section"

    # Using vLLM for faster inference (if available)
    if command -v python -c "import vllm" &> /dev/null 2>&1; then
        python -m vllm.entrypoints.openai.run_batch \
            --model "$MODEL_PATH" \
            --input "$DATASET" \
            --output "$OUTPUT_FILE"
    else
        # Fallback to LLaMA-Factory
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        llamafactory-cli predict \
            --model_name_or_path "$MODEL_PATH" \
            --template "$TEMPLATE" \
            --dataset "$DATASET" \
            --output_dir "$OUTPUT_DIR" \
            --max_new_tokens "$MAX_NEW_TOKENS"
    fi

    echo "Completed: $section -> $OUTPUT_FILE"
done

echo "============================================"
echo "All predictions completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================"
