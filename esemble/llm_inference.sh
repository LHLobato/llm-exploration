#!/usr/bin/env bash
set -euo pipefail

# Roda a partir da mesma pasta do evaluate_llm.py
# (usa paths relativos: ../logs, ../models, ../data)

# ---------------------------------------------------------------------------
# Configuração - ajuste aqui
# ---------------------------------------------------------------------------
DATASET="domain-enriched"
STRATEGY="tokenized"
TRAIN_MODE="tuning"      # zero | few | tuning
NUM_SHOTS=2               # só usado se TRAIN_MODE=few
DORA=true                 # true se os checkpoints foram ajustados com DoRA/LoRA

N_RUNS=1
WARMUP=1
SINGLE_CALL_N=300
BATCH=32

# model_key : "modelpath|parameters"
# modelpath = checkpoint local fine-tuned (TRAIN_MODE=tuning) OU id do HF Hub
# (TRAIN_MODE=zero/few). Os paths locais abaixo batem com os diretórios reais:
# ../models/{model}/{dataset}-{strategy}-{parameters}  (SEM subpasta /best --
# ajuste pra ".../best" se seu train_model_llm.py salvar aninhado assim)
declare -A MODELS=(
    ["Gemma"]="../models/Gemma/${DATASET}-${STRATEGY}-1B/best|1B"
    ["Llama-3"]="../models/Llama-3/${DATASET}-${STRATEGY}-1B/best|1B"
    ["Qwen"]="../models/Qwen/${DATASET}-${STRATEGY}-1.5B/best|1.5B"
)

# Pra avaliar zero-shot/few-shot direto dos checkpoints base do HF Hub (sem
# fine-tuning local), troque os valores acima por algo como:
#   ["Gemma"]="google/gemma-3-1b-it|1B"
#   ["Llama-3"]="meta-llama/Llama-3.2-1B-Instruct|1B"
#   ["Qwen"]="Qwen/Qwen2.5-1.5B-Instruct|1.5B"
# e ajuste TRAIN_MODE para "zero" ou "few" (e DORA=false).

# Se o checkpoint fine-tuned salvou só o adaptador PEFT (não mergeado),
# defina o modelo base correspondente aqui -- deixe "" se o checkpoint já
# está mergeado/completo.
declare -A BASE_MODELS=(
    ["Gemma"]=""
    ["Llama-3"]=""
    ["Qwen"]=""
)

# ---------------------------------------------------------------------------
for model_key in "${!MODELS[@]}"; do
    IFS="|" read -r model_path parameters <<< "${MODELS[$model_key]}"
    exp_name="${model_key}-${parameters}-${DATASET}-${STRATEGY}-${TRAIN_MODE}"

    echo ">>> Testando tempo de inferência: ${exp_name}"

    cmd=(python llm_inference.py
        --model "$model_key"
        --modelpath "$model_path"
        --dataset "$DATASET"
        --strategy "$STRATEGY"
        --parameters "$parameters"
        --train_mode "$TRAIN_MODE"
        --num_shots "$NUM_SHOTS"
        --batch "$BATCH"
        --exp "$exp_name"
        --n_runs "$N_RUNS"
        --warmup "$WARMUP"
        --single_call_n "$SINGLE_CALL_N")

    if [ "$DORA" = true ]; then
        cmd+=(--dora)
        base_model_path="${BASE_MODELS[$model_key]}"
        if [ -n "$base_model_path" ]; then
            cmd+=(--base_modelpath "$base_model_path")
        fi
    fi

    "${cmd[@]}"
    echo ""
done

echo "Testes de tempo de inferência (LLM) finalizados."
echo "Métricas: ../logs/llm_inference_results.csv"