#!/bin/bash

set -e

echo "========================================================="
echo "A iniciar bateria de treinos noturnos - Qwen & TinyLlama "
echo "========================================================="




echo "[1/2] A iniciar Qwen Instruct..."
python ../training/optimized_training.py \
  --model Qwen \
  --modelpath Qwen/Qwen3.5-0.8B \
  --dataset domain-enriched \
  --parameters 0.8B \
  --strategy tokenized \
  --use_unsloth \
  --num_epochs 3 \
  --dora

echo "---------------------------------------------------------"
echo "Qwen 0.8B concluído! A preparar o TinyLlama..."
echo "---------------------------------------------------------"

#
#echo "[2/2] A iniciar TinyLlama (1.1B) Chat..."
#python bert_training.py \
#  --model TinyLlama \
 # --modelpath TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
 # --dataset domain-enriched \
  #--parameters 1.1B \
  #--strategy tokenized \
 # --num_epochs 3 \
 # --dora

echo "========================================================="
echo "Treinos noturnos finalizados com sucesso!"
echo "========================================================="
