import pandas as pd 

arquivos = ['benign_5345_features.csv', 'malicious_1356_features.csv']

# Abre o arquivo de texto UMA única vez ("w" para sobrescrever e criar um novo limpo a cada execução)
with open("relatory.txt", "w") as file:
    
    for nome in arquivos:
        # Lê o CSV
        df = pd.read_csv(f"data/less-is-more/{nome}", index_col=False)


        file.write(f"\n{'='*50}\n")
        file.write(f"ARQUIVO: {nome}.csv\n")
        file.write(f"{'='*50}\n\n")
        

        file.write(">>> COLUNAS:\n")
        file.write(str(df.columns.tolist()) + "\n\n")
        
        file.write(">>> HEAD (5 primeiras linhas):\n")
        file.write(df.head().to_string() + "\n\n")
        
        # Escreve o describe
        file.write(">>> DESCRIBE (Estatísticas):\n")
        file.write(df.describe().to_string() + "\n\n")

print("Relatório gerado com sucesso em 'relatory.txt'!")