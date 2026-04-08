#!/bin/bash

# Definição das arrays sem espaços extras
MODELS=("sentence-transformers/all-distilroberta-v1" "sentence-transformers/all-mpnet-base-v2" "BAAI/bge-base-en-v1.5")
CLASSIFIERS=("XGB" "RF" "SVM" "LR" "MLP")

# Loop principal
for model in "${MODELS[@]}"; do
    for cls in "${CLASSIFIERS[@]}"; do
        echo "=========================================================="
        echo "🚀 TESTANDO: $model | $cls"
        echo "=========================================================="
        
        python sentence-bert.py --classifier "$cls" --model "$model" --random_state 0 --fitsentencemodel
        

        echo "Finalizado: $model com $cls em $(date)" >> progresso.log
    done
done
