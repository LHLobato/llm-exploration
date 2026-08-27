import pandas as pd 

columns = ['Domain', 'URLSimilarityIndex', 'LineOfCode', 'NoOfExternalRef', 'NoOfImage', 'NoOfSelfRef', 'label']

df = pd.read_csv("phiusiil-filtered.csv", index_col=False)
df = df[columns]
df.to_csv("cleaned_phiusiil.csv", index=False)