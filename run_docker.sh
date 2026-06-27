#!/bin/bash
# =============================================================
#  run_docker.sh  —  helper para buildar e rodar o container
# =============================================================

PROJECT_DIR="."
IMAGE_NAME="llm-exploration"

# ---------- build (só precisa rodar uma vez) ----------
build() {
    echo ">>> Buildando imagem $IMAGE_NAME..."
    docker build -t $IMAGE_NAME "$PROJECT_DIR"
    echo ">>> Build concluído!"
}

# ---------- shell interativo ----------
shell() {
    echo ">>> Abrindo shell no container..."
    docker run --gpus all -it --rm \
        -v "$PROJECT_DIR":/workspace \
        -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
        --shm-size=16g \
        $IMAGE_NAME bash
}

# ---------- roda script diretamente ----------
# Uso: ./run_docker.sh run scripts/train.py --model Llama-3 --dataset domain-enriched ...
run() {
    SCRIPT=$1
    shift
    echo ">>> Rodando $SCRIPT dentro do container..."
    docker run --gpus all -it --rm \
        -v "$PROJECT_DIR":/workspace \
        -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
        --shm-size=16g \
        $IMAGE_NAME python3 $SCRIPT "$@"
}

# ---------- dispatcher ----------
case "$1" in
    build) build ;;
    shell) shell ;;
    run)   shift; run "$@" ;;
    *)
        echo "Uso: $0 {build|shell|run <script> [args...]}"
        echo ""
        echo "Exemplos:"
        echo "  $0 build"
        echo "  $0 shell"
        echo "  $0 run scripts/berts_classifier.sh"
        ;;
esac
