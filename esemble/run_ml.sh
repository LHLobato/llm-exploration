#!/usr/bin/env bash
set -euo pipefail
TRAIN_PATH="../data/acme/fixed_train.csv"
VAL_PATH="../data/acme/fixed_val.csv"
TEST_PATH="../data/acme/fixed_test.csv"

MODELS=("xgb" "rf" "lr" "svm" "knn")
SEEDS=(0)

FEATURESELECT=false   


for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        exp_name="${model}_seed${seed}"
        echo ">>> Rodando ${exp_name}"

        cmd=(python machine_learning.py
            --pre-split false
            --csv_path "../data/PhiUSIIL/cleaned_phiusiil.csv"
            --model "$model"
            --random_state "$seed"
            --exp "$exp_name" 
            --label_col "label"
            )
            
        if [ "$FEATURESELECT" = true ]; then
            cmd+=(--featureselect)
        fi

        "${cmd[@]}"
    done
done

echo "Experimentos finalizados. Modelos em ./joblib, logs em ../logs."