#!/bin/bash
PATHS=("meta-llama/Llama-3.2-1B-Instruct" "google/gemma-3-1b-it" "Qwen/Qwen2.5-1.5B-Instruct")
#MODELS=("Llama-3" "Gemma-2B" "Qwen")
#PATHS=("distilbert/distilbert-base-uncased" "google-bert/bert-base-uncased")
MODELS=("Llama-3" "Gemma" "Qwen")

#PATHS=("answerdotai/ModernBERT-Base" "microsoft/deberta-v3-base")
#MODELS=("ModernBERT" "DEBERTa" )

DATASETS=("10ksubset-phiusiil")
echo "----------------------------------"
echo "        INICIANDO TESTES"
echo "----------------------------------"

for DATA in "${!DATASETS[@]}"; do
        for i in "${!MODELS[@]}"; do
        path=${PATHS[$i]}
        model=${MODELS[$i]}
        dataset=${DATASETS[$DATA]}
        echo "Rodando: $model (Caminho: $path)"

        python ../training/bert_training.py \
                --model "$model" \
                --modelpath "$path" \
                --dataset "$dataset" \
                --batch 16 \
                --strategy tokenized-prompt \
                --train_mode "zero"

        echo "Finalizado: $model"
        echo "----------------------------------"
        done
done 
echo "Todos os testes foram concluídos com sucesso!"
