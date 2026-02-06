#!/bin/bash
# Component-wise Evaluation Script using AlpacaEval
# Usage: bash run_eval.sh <model_predictions_dir> <baseline_predictions_dir> <output_dir>

set -e

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable not set"
    echo "Please run: export OPENAI_API_KEY='your-api-key'"
    exit 1
fi

# Arguments
MODEL_DIR="${1:?Error: Please provide model predictions directory}"
BASELINE_DIR="${2:?Error: Please provide baseline predictions directory}"
OUTPUT_DIR="${3:-./eval_results}"

# Sections to evaluate
SECTIONS=(abstract background conclusion evaluation introduction title)

echo "============================================"
echo "Component-wise Evaluation (AlpacaEval)"
echo "============================================"
echo "Model predictions: $MODEL_DIR"
echo "Baseline predictions: $BASELINE_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "============================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if alpaca_eval is installed
if ! python -c "import alpaca_eval" 2>/dev/null; then
    echo "Error: alpaca_eval not found."
    echo "Please install it first:"
    echo "  git clone https://github.com/tatsu-lab/alpaca_eval.git"
    echo "  cd alpaca_eval && pip install -e ."
    exit 1
fi

# Run evaluation for each section
for section in "${SECTIONS[@]}"; do
    echo ""
    echo "Evaluating: $section"
    echo "-" * 40

    MODEL_FILE="$MODEL_DIR/${section}.json"
    BASELINE_FILE="$BASELINE_DIR/${section}.json"
    SECTION_OUTPUT="$OUTPUT_DIR/$section"

    # Check if files exist
    if [ ! -f "$MODEL_FILE" ]; then
        echo "Warning: $MODEL_FILE not found, skipping"
        continue
    fi

    if [ ! -f "$BASELINE_FILE" ]; then
        echo "Warning: $BASELINE_FILE not found, skipping"
        continue
    fi

    mkdir -p "$SECTION_OUTPUT"

    # Run AlpacaEval
    python -m alpaca_eval.main evaluate \
        --model_outputs "$MODEL_FILE" \
        --reference_outputs "$BASELINE_FILE" \
        --annotators_config alpaca_eval_gpt4_turbo_fn \
        --output_path "$SECTION_OUTPUT"

    echo "Results saved to: $SECTION_OUTPUT"
done

echo ""
echo "============================================"
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================"

# Print summary
echo ""
echo "Summary:"
for section in "${SECTIONS[@]}"; do
    RESULT_FILE="$OUTPUT_DIR/$section/leaderboard.json"
    if [ -f "$RESULT_FILE" ]; then
        WIN_RATE=$(python -c "import json; d=json.load(open('$RESULT_FILE')); print(f'{d.get(\"win_rate\", 0):.2f}%')" 2>/dev/null || echo "N/A")
        echo "  $section: $WIN_RATE"
    fi
done
