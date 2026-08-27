#!/usr/bin/env bash
set -euo pipefail

# Roda a partir da mesma pasta do evaluate_model.py
# (usa paths relativos: ../logs, ./joblib)

# ---------------------------------------------------------------------------
# Configuração - ajuste aqui
# ---------------------------------------------------------------------------
#TEST_PATH="../data/acme/fixed_test.csv"
#LABEL_COL="malicious"
#LEX=false     
#MODELS_DIR="./joblib"

#N_RUNS=5
#WARMUP=1
#SINGLE_CALL_N=300

# ---------------------------------------------------------------------------
#for model_file in "$MODELS_DIR"/*.joblib; do
#    # pula eventuais arquivos de scaler, caso voltem a ser salvos junto
 #   [[ "$model_file" == *scaler* ]] && continue

  #  exp_name="$(basename "$model_file" .joblib)"
  #  echo ">>> Testando tempo de inferência: ${exp_name}"

 #   cmd=(python inference_measure.py
   #     --test_samples_path "$TEST_PATH"
  #      --model_path "$model_file"
   #     --label_col "$LABEL_COL"
    #    --exp "$exp_name"
    #    --n_runs "$N_RUNS"
    #    --warmup "$WARMUP"
    #    --single_call_n "$SINGLE_CALL_N")
#
    #if [ "$LEX" = true ]; then
   #     cmd+=(--lex)
    #fi

    #"${cmd[@]}"
   # echo ""
#done

#echo "Testes de tempo de inferência finalizados."
#echo "Métricas: ../logs/inference_results.csv"
#echo "Matrizes de confusão: ../logs/confusion_matrices/"

#!/usr/bin/env bash
set -euo pipefail

# Roda a partir da mesma pasta do evaluate_model.py
# (usa paths relativos: ../logs, ./joblib)

# ---------------------------------------------------------------------------
# Configuração - ajuste aqui
# ---------------------------------------------------------------------------
CSV_PATH="../data/PhiUSIIL/cleaned_phiusiil.csv"   # CSV único, MESMO usado no treino
RANDOM_STATE=0                                     # MESMO seed usado no treino
LABEL_COL="label"
LEX=false
MODELS_DIR="./joblib"

N_RUNS=5
WARMUP=1
SINGLE_CALL_N=300

# ---------------------------------------------------------------------------
for model_file in "$MODELS_DIR"/*-phiusiil.joblib; do
    # pula eventuais arquivos de scaler, caso voltem a ser salvos junto
    [[ "$model_file" == *scaler* ]] && continue

    exp_name="$(basename "$model_file" .joblib)"
    echo ">>> Testando tempo de inferência: ${exp_name}"

    cmd=(python inference_measure.py
        --pre-split false
        --csv_path "$CSV_PATH"
        --random_state "$RANDOM_STATE"
        --model_path "$model_file"
        --label_col "$LABEL_COL"
        --exp "$exp_name"
        --n_runs "$N_RUNS"
        --warmup "$WARMUP"
        --single_call_n "$SINGLE_CALL_N")

    if [ "$LEX" = true ]; then
        cmd+=(--lex)
    fi

    "${cmd[@]}"
    echo ""
done

echo "Testes de tempo de inferência finalizados."
echo "Métricas: ../logs/phisuiil_inference_results.csv"