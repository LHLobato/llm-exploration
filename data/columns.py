import pandas as pd 

df = pd.read_csv("dataset.csv", index_col=False)

print(df.columns)
