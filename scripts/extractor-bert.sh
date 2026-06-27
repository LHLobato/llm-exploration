#!/bin/bash
VENV_DIR="~/ic_venv"
REQUIREMENTS="requirements.txt"
if [ "$VIRTUAL_ENV" != "$(pwd)/$VENV_DIR" ]; then
    echo "O ambiente '$VENV_DIR' não está ativado no momento."
    if [ -d "$VENV_DIR" ]; then
        echo "Diretório '$VENV_DIR' encontrado. Ativando o ambiente..."
        source "$VENV_DIR/bin/activate"
    else
        echo "Ambiente '$VENV_DIR' não existe. Criando agora..."
        python3 -m venv "$VENV_DIR"
        echo "Ativando o ambiente recém-criado..."
        source "$VENV_DIR/bin/activate"
        pip install --upgrade pip > /dev/null 2>&1
        if [ -f "$REQUIREMENTS" ]; then
            echo "Arquivo $REQUIREMENTS encontrado. Instalando dependências..."
            pip install -r "$REQUIREMENTS"
        else
            echo "Aviso: Nenhum arquivo '$REQUIREMENTS' encontrado. Nenhuma dependência extra foi instalada."
        fi
    fi
    echo "Pronto! Ambiente ativado e configurado."
else
    echo "Tudo certo! O ambiente '$VENV_DIR' já está ativado e pronto para uso."
fi


declare -A BERT_MODEL_PATHS
BERT_MODEL_PATHS["BERT-Base"]="../models/BERT-Base/domain-enriched-only-names-None/best"
BERT_MODEL_PATHS["distilBERT"]="../models/distilBERT/domain-enriched-only-names-None/best"
BERT_MODEL_PATHS["DEBERTa"]="../models/DEBERTa/domain-enriched-only-names-None/best"
BERT_MODEL_PATHS["ModernBERT"]="../models/ModernBERT/domain-enriched-only-names-None/best"

declare -A DATASET_CSV
DATASET_CSV["domain"]="../data/acme/dataset.csv"

declare -A DATASET_TEXT_COL
DATASET_TEXT_COL["domain"]="name"

declare -A DATASET_LABEL_COL
DATASET_LABEL_COL["domain"]="malicious"

FEATURES_DIR="../training/features"
RESULTS_CSV="../training/results/bert_ensemble_results.csv"

echo "========================================================="
echo " Iniciando bateria de experimentos - BERT Ensemble Heads"
echo "========================================================="

for DATASET in "${!DATASET_CSV[@]}"; do
    echo "---------------------------------------------------------"
    echo " Processando Dataset: $DATASET"
    echo "---------------------------------------------------------"
    for MODEL in "${!BERT_MODEL_PATHS[@]}"; do
        echo " -> Modelo: $MODEL | Dataset: $DATASET"
        python ../training/bert_ensemble_experiment.py \
            --results_csv  "$RESULTS_CSV" \
            --model_name   "$MODEL" \
            --model_path   "${BERT_MODEL_PATHS[$MODEL]}" \
            --dataset      "$DATASET" \
            --data_csv     "${DATASET_CSV[$DATASET]}" \
            --text_col     "${DATASET_TEXT_COL[$DATASET]}" \
            --label_col    "${DATASET_LABEL_COL[$DATASET]}" \
            --features_dir "$FEATURES_DIR" \
            --batch_size   64 \
            --num_workers  8 \
            --max_length   200
        echo " -> [OK] $MODEL | $DATASET finalizado!"
        echo ""
    done
done

echo "========================================================="
echo " Todos os experimentos BERT ensemble foram concluídos!"
echo "========================================================="