import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

# 1. Definindo os Modelos e Condições
modelos = [
    "DistilBERT",
    "BERT",
    "ModernBERT",
    "DeBERTa-v3",
    "Qwen 2.5",
    "Gemma 3",
    "LLaMA 3.2",
]
condicoes = ["Custom_Raw", "Custom_Enriched", "PhiUSIIL_Raw", "PhiUSIIL_Enriched"]

# 2. Matriz de F1-Scores
data_f1 = np.array(
    [
        [0.8275, 0.9069, 0.8840, 0.9998],  # DistilBERT
        [0.8305, 0.9206, 0.8835, 0.9999],  # BERT
        [0.8315, 0.9191, 0.8831, 0.9999],  # ModernBERT
        [0.8431, 0.9143, 0.8871, 0.9999],  # DeBERTa-v3
        [0.8400, 0.9199, 0.8800, 0.9999],  # Qwen 2.5
        [0.8350, 0.9214, 0.8799, 0.9999],  # Gemma 3
        [0.8410, 0.9216, 0.8809, 0.9999],  # LLaMA 3.2
    ]
)

df = pd.DataFrame(data_f1, index=modelos, columns=condicoes)
print(df)
# 3. Teste Global (Friedman)
stat, p_value = friedmanchisquare(*df.to_numpy().T)
print(f"Teste de Friedman (Global): χ² = {stat:.4f}, p = {p_value:.4e}\n")

# 4. Testes Pareados (Wilcoxon) para as hipóteses principais
print("--- Testes Pareados (Wilcoxon Signed-Rank) ---")

# Comparação 1: Custom_Raw vs Custom_Enriched
stat_c, p_c = wilcoxon(df["Custom_Raw"], df["Custom_Enriched"])
print(f"Custom Dataset   (Raw vs Enriched): p-value = {p_c:.4f}")

# Comparação 2: PhiUSIIL_Raw vs PhiUSIIL_Enriched
stat_p, p_p = wilcoxon(df["PhiUSIIL_Raw"], df["PhiUSIIL_Enriched"])
print(f"PhiUSIIL Dataset (Raw vs Enriched): p-value = {p_p:.4f}")
