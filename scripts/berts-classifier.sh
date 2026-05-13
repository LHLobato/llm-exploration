#!/bin/bash


PATHS=("google/gemma-3-4b-it")
MODELS=("Gemma-2B")


echo "----------------------------------"
echo "        INICIANDO TESTES"
echo "----------------------------------"

for i in "${!MODELS[@]}"; do
    path=${PATHS[$i]}
    model=${MODELS[$i]}

    echo "Rodando: $model (Caminho: $path)"
    
    python ../training/bert_training.py \
        --model "$model" \
        --modelpath "$path" \
        --dataset domain-enriched \
        --strategy "only-names" \
        --num_epochs 5
    
    echo "Finalizado: $model"
    echo "----------------------------------"
done

echo "Todos os testes foram concluídos com sucesso!"
