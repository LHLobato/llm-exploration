import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def enrich_prompt(row):
    prompt = (
       f"[NAME] {row['Domain']} \n"
       f"[SIMIL] {row['URLSimilarityIndex']} \n"
        f"[CODE] {row['LineOfCode']} \n"
        f"[EXREF] {row['NoOfExternalRef']} \n"
        f"[IMG] {row['NoOfImage']} \n"
        f"[INREF] {row['NoOfSelfRef']} \n"
    )
    return prompt

phi = pd.read_csv("PhiUSIIL/PhiUSIIL.csv", index_col=False)

phi['prompt'] = phi.apply(enrich_prompt, axis=1)
phi.to_csv("PhiUSIIL/phiusiil-filtered.csv", index=False)

print(phi['prompt'].iloc[0])

"""

df_raw = pd.read_csv("less-is-more/BTCP.csv", index_col=False)

name_col     = df_raw.columns[0]
feature_idxs = [1,2,3,4,5,6,7,8,9,10,11,13,15]
label_col    = df_raw.columns[17]
feature_cols = [df_raw.columns[i] for i in feature_idxs]

print("Name col   :", name_col)
print("Feature cols:", feature_cols)
print("Label col  :", label_col)

def enrich_prompt(row):
    fields = "\n".join(
        f"[{col.upper()}] {round(row[col], 3)}"
        for col in feature_cols
    )
    return f"[NAME] {row[name_col]} \n{fields} \n"

df = df_raw[[name_col] + feature_cols].copy()
df['prompt'] = df.apply(enrich_prompt, axis=1)
df['label']  = df_raw[label_col].values

df.to_csv("less-is-more/BTCP.csv", index=False)

print(f"Total: {len(df)} amostras")
print("\nExemplo de prompt:")
print(df['prompt'].iloc[0])
"""