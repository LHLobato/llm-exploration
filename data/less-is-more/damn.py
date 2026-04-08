import pandas as pd
import re

def normalize_domain(d):
    d = str(d).strip().lower()
    d = re.sub(r'^https?://', '', d)
    d = re.sub(r'^www\.', '', d)
    d = d.split('?')[0].split('#')[0].rstrip('/')
    return d

# Ajuste os caminhos se necessário
benign = pd.read_csv("malicious_1356_features.csv", index_col=False)
median = pd.read_csv("BTCP.csv", index_col=False)

# Identifica coluna de nome em cada arquivo
benign_name_col = benign.columns[0]   # provavelmente 'name'
median_name_col = median.columns[0]   # provavelmente também 'name'

print(f"Coluna benign: '{benign_name_col}' | Coluna median: '{median_name_col}'")
print(f"Total benign_5345: {len(benign)}")
print(f"Total median (benign+malicious): {len(median)}")

benign_domains = set(benign[benign_name_col].apply(normalize_domain))
median_domains = set(median[median_name_col].apply(normalize_domain))

# Interseção
em_ambos      = benign_domains & median_domains
so_no_benign  = benign_domains - median_domains
so_no_median  = median_domains - benign_domains

print(f"\nDomínios de benign_5345 presentes no median : {len(em_ambos)}")
print(f"Domínios de benign_5345 AUSENTES no median  : {len(so_no_benign)}")
print(f"Domínios no median não presentes no benign  : {len(so_no_median)}")

if so_no_benign:
    print("\nExemplos ausentes no median:")
    for d in list(so_no_benign)[:10]:
        print(" ", d)