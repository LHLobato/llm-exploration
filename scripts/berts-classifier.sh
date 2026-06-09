#!/bin/bash
PATHS=("meta-llama/Llama-3.2-1B-Instruct" "google/gemma-3-1b-it" "Qwen/Qwen2.5-1.5B-Instruct")
MODELS=("Llama-3" "Gemma-2B" "Qwen")

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
        --dataset phiusiil \
        --strategy "only-names" \
        --num_epochs 3 \
        --dora
        
    echo "Finalizado: $model"
    echo "----------------------------------"
done

echo "Todos os testes foram concluídos com sucesso!"
