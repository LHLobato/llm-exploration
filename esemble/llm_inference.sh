#!/usr/bin/env bash
set -euo pipefail

# Roda a partir da mesma pasta do script python de avaliação
# (usa caminhos relativos: ../logs, ../models, ../data)

SCRIPT_NAME="llm_inference.py"

# ---------------------------------------------------------------------------
# Configuração global
# ---------------------------------------------------------------------------
DATASET="domain-enriched"
STRATEGY="tokenized"
TRAIN_MODE="tuning"      # zero | few | tuning
NUM_SHOTS=2              # Usado apenas se TRAIN_MODE=few
BATCH=32
N_RUNS=1
WARMUP=1
SINGLE_CALL_N=1000       # 1000 amostras individuais para média de latência

# ---------------------------------------------------------------------------
# Definição de todos os modelos suportados
# Formato: ["NomeModelo"]="caminho_ou_hf_id|tamanho_ou_tag|usa_dora"
# ---------------------------------------------------------------------------
declare -A MODELS=(
    # --- Encoders / Sequence Classifiers ---
    ["BERT-Base"]="../models/BERT-Base/${DATASET}-${STRATEGY}/best|base|false"
    ["distilBERT"]="../models/distilBERT/${DATASET}-${STRATEGY}/best|base|false"
    ["DEBERTa"]="../models/DEBERTa/${DATASET}-${STRATEGY}/best|v3-base|false"
    ["ModernBERT"]="../models/ModernBERT/${DATASET}-${STRATEGY}/best|base|false"

    # --- Causal LLMs ---
    ["TinyLlama"]="../models/TinyLlama/${DATASET}-${STRATEGY}-1.1B/best|1.1B|true"
    ["Llama-3"]="../models/Llama-3/${DATASET}-${STRATEGY}-1B/best|1B|true"
    ["Gemma"]="../models/Gemma/${DATASET}-${STRATEGY}-1B/best|1B|true"
    ["Qwen"]="../models/Qwen/${DATASET}-${STRATEGY}-1.5B/best|1.5B|true"
    ["DeepSeek"]="../models/DeepSeek/${DATASET}-${STRATEGY}-1.5B/best|1.5B|true"
)

# Se o checkpoint fine-tuned de DoRA/LoRA salvou apenas o adaptador PEFT
# (não mergeado), indique aqui o base_modelpath de cada um.
# Deixe "" se o modelo já estiver totalmente mergeado, ou se for encoder fine-tunado.
declare -A BASE_MODELS=(
    ["BERT-Base"]=""
    ["distilBERT"]=""
    ["DEBERTa"]=""
    ["ModernBERT"]=""
    ["TinyLlama"]="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ["Llama-3"]="meta-llama/Llama-3.2-1B-Instruct"
    ["Gemma"]="google/gemma-3-1b-it"
    ["Qwen"]="Qwen/Qwen2.5-1.5B-Instruct"
    ["DeepSeek"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)

# ---------------------------------------------------------------------------
# Execução dos Benchmarks
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "Iniciando avaliação de inferência e VRAM para o dataset: ${DATASET}"
echo "Modo: ${TRAIN_MODE} | Amostras single-call: ${SINGLE_CALL_N} | Batch: ${BATCH}"
echo "======================================================================"

for model_key in "${!MODELS[@]}"; do
    IFS="|" read -r model_path parameters use_dora <<< "${MODELS[$model_key]}"
    exp_name="${model_key}-${parameters}-${DATASET}-${STRATEGY}-${TRAIN_MODE}"

    echo "------------------------------------------------------------------"
    echo ">>> Testando modelo: [${model_key}] | Exp: ${exp_name}"
    echo "    Checkpoint: ${model_path}"
    echo "------------------------------------------------------------------"

    # Valida se o diretório local existe caso não seja um repositório remoto do HF Hub
    if [[ ! "$model_path" =~ "/" ]] || [[ -d "$model_path" ]]; then
        if [[ ! -e "$model_path" ]]; then
            echo "[AVISO] Caminho local '${model_path}' não encontrado. Pulando este modelo."
            continue
        fi
    fi

    cmd=(python "$SCRIPT_NAME"
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
        --single_call_n "$SINGLE_CALL_N"
        --log_csv "../logs/llm_inference_results.csv")

    # Tratamento para adaptadores PEFT/DoRA
    if [ "$use_dora" = "true" ]; then
        cmd+=(--dora)
        base_model_path="${BASE_MODELS[$model_key]:-}"
        if [ -n "$base_model_path" ]; then
            cmd+=(--base_modelpath "$base_model_path")
        fi
    fi

    # Executa a avaliação
    "${cmd[@]}"

    echo ""
done

echo "======================================================================"
echo "Avaliações finalizadas com sucesso."
echo "Métricas salvas e anexadas em: ../logs/llm_inference_results.csv"
echo "======================================================================"
