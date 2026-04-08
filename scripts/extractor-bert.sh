#!/bin/bash

MODELS=("BERT-Base" "DEBERTa" "distilBERT") 
CLASSIFIERS=("ENSEMBLE")

for model in "${MODELS[@]}"; do 
    for cls in "${CLASSIFIERS[@]}"; do 
        
        python bert-extractor.py --model "$model" \
                                 --modelpath "models/$model/best" \
				 --ensemblepath "ensemble.txt" \
				 --ensemblemode "stacking" \
                                 --classifier "$cls"
    done 
done
