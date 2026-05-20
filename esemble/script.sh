CNN_PATH="/home/lhslobato/novo-reshaping/saved_models/CustomCNN/CustomCNN0-GADF-42_30.pth"
MLP_PATH="/home/lhslobato/llm-exploration/mlp-test/training_results/models/best_mlp.pt"
LLM_PATH="/home/lhslobato/llm-exploration/models/Llama/best"
TRAIN_PATH="/home/lhslobato/novo-reshaping/images/GADF/val"
TEST_PATH="/home/lhslobato/novo-reshaping/images/GADF/test"

python test.py --train_img_path "$TRAIN_PATH" \ 
		--test_img_path "$TEST_PATH" \
               --llm_path "$LLM_PATH" \ 
               --cnn_path "$CNN_PATH" \
               --tab_path "$MLP_PATH" \
