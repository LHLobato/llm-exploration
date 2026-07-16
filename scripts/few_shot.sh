#!/bin/bash
set -e

source ~/ic_venv/bin/activate

# ---------------- Tier SMALL (~1-2B) ----------------
MODELS_SMALL=("Gemma" "Qwen" "DeepSeek")
PATHS_SMALL=(
    "google/gemma-3-1b-it"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)
SIZE_TAG_SMALL="1b"

# ---------------- Tier LARGE (Llama-8B removido temporariamente) ----------------
# DeepSeek aparece 2x: R1-Distill-Qwen-7B (raciocínio destilado) e
# deepseek-llm-7b-chat (arquitetura nativa DeepSeek, instruct direto).
MODELS_LARGE=("Qwen" "DeepSeek" "DeepSeek")
PATHS_LARGE=(
    "Qwen/Qwen2.5-7B-Instruct"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "deepseek-ai/deepseek-llm-7b-chat"
)
SIZE_TAG_LARGE="large"

DATASETS=("15kphiusiil_subset")

BATCH_ZERO=8
BATCH_FEW=4   # prompts maiores com few-shot -> batch menor pra caber na 5060 8GB

NUM_SHOTS_LIST=(2 3)

run_tier () {
    local size_tag=$1
    shift
    local -n models_ref=$1
    shift
    local -n paths_ref=$1

    for i in "${!models_ref[@]}"; do
        model=${models_ref[$i]}
        path=${paths_ref[$i]}

        for DATASET in "${DATASETS[@]}"; do

            # ----------------- ZERO-SHOT -----------------
            echo "Rodando ZERO-SHOT: $model [$size_tag] (Caminho: $path)"

            python ../training/bert_training.py \
                --model "$model" \
                --modelpath "$path" \
                --dataset "$DATASET" \
                --batch "$BATCH_ZERO" \
                --strategy tokenized-prompt \
                --train_mode "zero" \
                --parameters "${size_tag}-zero"

            echo "Finalizado ZERO-SHOT: $model [$size_tag]"
            echo "----------------------------------"

            # ----------------- FEW-SHOT (2 a 5) -----------------
            for shots in "${NUM_SHOTS_LIST[@]}"; do
                echo "Rodando FEW-SHOT ($shots shots): $model [$size_tag] (Caminho: $path)"

                python ../training/bert_training.py \
                    --model "$model" \
                    --modelpath "$path" \
                    --dataset "$DATASET" \
                    --batch "$BATCH_FEW" \
                    --strategy tokenized-prompt \
                    --train_mode "few" \
                    --num_shots "$shots" \
                    --parameters "${size_tag}-few${shots}"

                echo "Finalizado FEW-SHOT ($shots shots): $model [$size_tag]"
                echo "----------------------------------"
            done

        done
    done
}

echo "----------------------------------"
echo "     INICIANDO TESTES ZERO/FEW"
echo "----------------------------------"

#run_tier "$SIZE_TAG_SMALL" MODELS_SMALL PATHS_SMALL
run_tier "$SIZE_TAG_LARGE" MODELS_LARGE PATHS_LARGE

echo "Todos os testes zero/few-shot foram concluídos com sucesso!"
