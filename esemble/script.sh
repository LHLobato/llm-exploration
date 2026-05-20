#!/bin/bash

GREEN='\033[1;32m'
BLUE='\033[1;34m'
YELLOW='\033[1;33m'
CYAN='\033[1;36m'
RED='\033[1;31m'
NC='\033[0m'

clear
echo -e "${BLUE}======================================================${NC}"
echo -e "${CYAN}     Domain Ensemble - Pipeline de Avaliação    ${NC}"
echo -e "${BLUE}======================================================${NC}\n"


echo -e "${YELLOW}[INFO] Carregando caminhos dos modelos e datasets...${NC}"

CNN_PATH="/home/lhslobato/novo-reshaping/saved_models/CustomCNN/CustomCNN0-GADF-42_30.pth"
MLP_PATH="/home/lhslobato/llm-exploration/mlp-test/training_results/models/best_mlp.pt"
LLM_PATH="/home/lhslobato/llm-exploration/models/Llama/best"
TRAIN_PATH="/home/lhslobato/novo-reshaping/images/GADF/val"
TEST_PATH="/home/lhslobato/novo-reshaping/images/GADF/test"

echo -e "  🔹 ${GREEN}Vision Model (CNN):${NC} $CNN_PATH"
echo -e "  🔹 ${GREEN}Tabular Model (MLP):${NC} $MLP_PATH"
echo -e "  🔹 ${GREEN}Text Model (LLM):${NC}   $LLM_PATH"
echo -e "  🔹 ${GREEN}Imagens (Validação):${NC}  $TRAIN_PATH"
echo -e "  🔹 ${GREEN}Imagens (Teste):${NC}      $TEST_PATH\n"

echo -e "${YELLOW}[INFO] Iniciando a execução do script Python...${NC}\n"

python test.py \
    --train_img_path "$TRAIN_PATH" \
    --test_img_path "$TEST_PATH" \
    --llm_path "$LLM_PATH" \
    --cnn_path "$CNN_PATH" \
    --tab_path "$MLP_PATH"

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}[SUCESSO] Teste concluído sem erros!${NC}"
else
    echo -e "\n${RED}[ERRO] Ocorreu um problema durante a execução do script. Verifique os logs acima. ❌${NC}"
fi
