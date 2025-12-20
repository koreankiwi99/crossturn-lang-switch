#!/bin/bash
# Cross-Lingual Transfer Experiment (X→Y)
# Tests switching between non-English languages

WORKERS=32
BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_DIR="$BASE_DIR/data/experiments"

MODELS=("gpt-5" "claude-opus-4.5")
DATASETS=(
    "zh_to_de"
    "de_to_zh"
    "es_to_ar"
    "ar_to_es"
)

echo "============================================================"
echo "CROSS-LINGUAL TRANSFER EXPERIMENT (X→Y)"
echo "============================================================"
echo "Models: ${MODELS[*]}"
echo "Datasets: ${DATASETS[*]}"
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
        echo "--- Running $model on $dataset ---"
        python "$BASE_DIR/src/scripts/run_experiment.py" \
            --model "$model" \
            --data "$DATA_DIR/${dataset}.jsonl" \
            --workers "$WORKERS"

        if [ $? -ne 0 ]; then
            echo "ERROR: Failed on $model / $dataset"
            continue
        fi
    done

    echo ""
    echo "--- Running Layer 1 (Language Fidelity) evaluation for $model ---"
    for dataset in "${DATASETS[@]}"; do
        python "$BASE_DIR/src/scripts/evaluation/language_fidelity.py" \
            --input "$RESULTS_DIR/responses_${dataset}_*.jsonl"
    done

    echo ""
    echo "--- Running Layer 2 (Task Accuracy) evaluation for $model ---"
    for dataset in "${DATASETS[@]}"; do
        python "$BASE_DIR/src/scripts/evaluation/task_accuracy.py" \
            --input "$RESULTS_DIR/responses_${dataset}_*.jsonl" \
            --workers 64
    done
done

echo ""
echo "============================================================"
echo "CROSS-LINGUAL EXPERIMENT COMPLETE"
echo "============================================================"
