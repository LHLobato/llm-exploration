#!/bin/bash


PATHS=("microsoft/deberta-v3-base")
MODELS=("DEBERTa")


echo "----------------------------------"
echo "        INICIANDO TESTES"
echo "----------------------------------"

for i in "${!MODELS[@]}"; do
    path=${PATHS[$i]}
    model=${MODELS[$i]}

    echo "Rodando: $model (Caminho: $path)"
    
    python bert_training.py \
        --model "$model" \
        --modelpath "$path" \
        --dataset domain-enriched \
        --strategy "tokenized" \
        --num_epochs 5
    
    echo "Finalizado: $model"
    echo "----------------------------------"
done

echo "Todos os testes foram concluídos com sucesso!"
