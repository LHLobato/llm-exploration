#!/bin/bash
VENV_DIR="/home/lhslobato/ic_venv"
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

# =========================================================================
# Configuração dos Modelos e Datasets HTTP
# =========================================================================

# Lista de arquiteturas base (os nomes das pastas dentro de ../models/)
MODELS=("BERT-Base" "distilBERT" "DEBERTa" "ModernBERT")

declare -A DATASET_CSV
DATASET_CSV["csic"]="../data/http/csic.csv"
DATASET_CSV["fwaf"]="../data/http/fwaf.csv"
DATASET_CSV["httpparams"]="../data/http/httpparams.csv"

declare -A DATASET_TEXT_COL
DATASET_TEXT_COL["csic"]="text"
DATASET_TEXT_COL["fwaf"]="text"
DATASET_TEXT_COL["httpparams"]="text"

declare -A DATASET_LABEL_COL
DATASET_LABEL_COL["csic"]="label"
DATASET_LABEL_COL["fwaf"]="label"
DATASET_LABEL_COL["httpparams"]="label"

declare -A DATASET_DISPLAY_NAME
DATASET_DISPLAY_NAME["csic"]="CSIC-2010"
DATASET_DISPLAY_NAME["fwaf"]="FWAF"
DATASET_DISPLAY_NAME["httpparams"]="HTTP-PARAMS"

FEATURES_DIR="../training/features"
RESULTS_CSV="../training/results/bert_ensemble_results.csv"

# =========================================================================
# Preparação dos datasets HTTP (csic2010, fwaf, http-params)
# =========================================================================
CSIC_CSV="../data/http/csic.csv"
FWAF_CSV="../data/http/fwaf.csv"
HTTPPARAMS_CSV="../data/http/httpparams.csv"

if [ ! -f "$CSIC_CSV" ] || [ ! -f "$FWAF_CSV" ] || [ ! -f "$HTTPPARAMS_CSV" ]; then
    echo "Gerando CSVs padronizados dos datasets HTTP (csic/fwaf/http-params)..."
    python3 - <<EOF
import os
import urllib.parse
import pandas as pd

def load_txt(path):
    with open(path, "r", encoding="utf8") as f:
        return [l.strip() for l in f if l.strip()]

# --- CSIC-2010 ---
if not os.path.exists("$CSIC_CSV"):
    bad = load_txt("../data/http/PreProcessedAnomalous.txt")
    good = load_txt("../data/http/PreprocessedNormalTraining.txt")
    texts = bad + good
    labels = [1] * len(bad) + [0] * len(good)
    pd.DataFrame({"text": texts, "label": labels}).to_csv("$CSIC_CSV", index=False)
    print(f"[OK] $CSIC_CSV gerado ({len(texts)} amostras)")

# --- FWAF ---
if not os.path.exists("$FWAF_CSV"):
    def load_fwaf_file(path, limit=120000):
        with open(path, "r") as f:
            data = list(set(f.readlines()))
        result = []
        for d in data:
            result.append(urllib.parse.unquote(d))
            if len(result) >= limit:
                break
        return result

    bad = list(set(load_fwaf_file("../data/http/badqueries.txt")))
    good = list(set(load_fwaf_file("../data/http/goodqueries.txt")))
    texts = bad + good
    labels = [1] * len(bad) + [0] * len(good)
    pd.DataFrame({"text": texts, "label": labels}).to_csv("$FWAF_CSV", index=False)
    print(f"[OK] $FWAF_CSV gerado ({len(texts)} amostras)")

# --- HTTP-PARAMS ---
if not os.path.exists("$HTTPPARAMS_CSV"):
    df = pd.read_csv("../data/http/payload_full.csv")
    df.dropna(inplace=True)
    df.loc[df["label"] == "norm", "label"] = 0
    df.loc[df["label"] == "anom", "label"] = 1
    df["label"] = df["label"].astype(int)
    df = df.rename(columns={"payload": "text"})[["text", "label"]]
    df.to_csv("$HTTPPARAMS_CSV", index=False)
    print(f"[OK] $HTTPPARAMS_CSV gerado ({len(df)} amostras)")
EOF
else
    echo "CSVs dos datasets HTTP já existem, pulando preparação."
fi

echo "========================================================="
echo " Iniciando bateria de experimentos HTTP - BERT Ensemble "
echo "========================================================="

for DATASET in "${!DATASET_CSV[@]}"; do
    echo "---------------------------------------------------------"
    echo " Processando Dataset: ${DATASET_DISPLAY_NAME[$DATASET]}"
    echo "---------------------------------------------------------"

    for MODEL in "${MODELS[@]}"; do
        
        # Monta o caminho dinâmico para os pesos pré-treinados
        # Ex: ../models/BERT-Base/csic-None-None
        MODEL_PATH="../models/${MODEL}/${DATASET}-None-None"
        
        # Verifica se existe uma subpasta "best" (comum nos checkpoints fine-tuned)
        if [ -d "${MODEL_PATH}/best" ]; then
            MODEL_PATH="${MODEL_PATH}/best"
        fi

        # Valida se o diretório do modelo realmente existe antes de rodar
        if [ ! -d "$MODEL_PATH" ]; then
            echo " [AVISO] Modelo não encontrado em: $MODEL_PATH. Pulando..."
            continue
        fi

        echo " -> Modelo: $MODEL | Pesos: $MODEL_PATH"
        
        python ../training/pipeline-extractor.py \
            --results_csv  "$RESULTS_CSV" \
            --model_name   "$MODEL" \
            --model_path   "$MODEL_PATH" \
            --dataset      "${DATASET_DISPLAY_NAME[$DATASET]}" \
            --data_csv     "${DATASET_CSV[$DATASET]}" \
            --text_col     "${DATASET_TEXT_COL[$DATASET]}" \
            --label_col    "${DATASET_LABEL_COL[$DATASET]}" \
            --features_dir "$FEATURES_DIR" \
            --batch_size   32 \
            --num_workers  8 \
            --max_length   200
            
        echo " -> [OK] $MODEL | ${DATASET_DISPLAY_NAME[$DATASET]} finalizado!"
        echo ""
    done
done

echo "========================================================="
echo " Todos os experimentos HTTP para BERT ensemble concluídos!"
echo "========================================================="