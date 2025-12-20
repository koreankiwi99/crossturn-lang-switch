#!/bin/bash
# System Prompt Ablation Experiment
# Tests whether explicit language instructions improve fidelity on code-switching conditions

SYSTEM_PROMPT="Always respond in the same language the user uses in their most recent message."
WORKERS=32
BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_DIR="$BASE_DIR/data/experiments"

MODELS=("gpt-5" "claude-opus-4.5")
DATASETS=(
    "en_to_de"
    "en_to_es"
    "en_to_zh"
    "en_to_ar"
    "de_to_en"
    "es_to_en"
    "zh_to_en"
    "ar_to_en"
)

echo "============================================================"
echo "SYSTEM PROMPT ABLATION EXPERIMENT"
echo "============================================================"
echo "System prompt: $SYSTEM_PROMPT"
echo "Models: ${MODELS[*]}"
echo "Datasets: ${#DATASETS[@]} code-switching conditions"
echo "Workers: $WORKERS"
echo "============================================================"
echo ""

for model in "${MODELS[@]}"; do
    echo "========================================"
    echo "MODEL: $model"
    echo "========================================"

    RESULTS_DIR="$BASE_DIR/results/$model"

    for dataset in "${DATASETS[@]}"; do
        echo ""
        echo "--- Running $model on $dataset with system prompt ---"
        python "$BASE_DIR/src/scripts/run_experiment.py" \
            --model "$model" \
            --data "$DATA_DIR/${dataset}.jsonl" \
            --workers "$WORKERS" \
            --system-prompt "$SYSTEM_PROMPT"

        if [ $? -ne 0 ]; then
            echo "ERROR: Failed on $model / $dataset"
            continue
        fi
    done

    echo ""
    echo "--- Running Layer 1 (Language Fidelity) evaluation for $model ---"
    python "$BASE_DIR/src/scripts/evaluation/language_fidelity.py" \
        --input "$RESULTS_DIR/responses_*_sysprompt_*.jsonl"

    echo ""
    echo "--- Running Layer 2 (Task Accuracy) evaluation for $model ---"
    python "$BASE_DIR/src/scripts/evaluation/task_accuracy.py" \
        --input "$RESULTS_DIR/responses_*_sysprompt_*.jsonl" \
        --workers 64
done

echo ""
echo "============================================================"
echo "ABLATION EXPERIMENT COMPLETE"
echo "============================================================"
echo ""
echo "Results saved in:"
for model in "${MODELS[@]}"; do
    echo "  - results/$model/responses_*_sysprompt_*.jsonl"
    echo "  - results/$model/language_eval_*_sysprompt_*.jsonl"
    echo "  - results/$model/evaluated_*_sysprompt_*.jsonl"
done
