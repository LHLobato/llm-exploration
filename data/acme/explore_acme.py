#!/usr/bin/env python3
"""Análise exploratória completa do dataset acme"""

import pandas as pd
import numpy as np
from collections import defaultdict

print("=" * 80)
print("ANÁLISE EXPLORATÓRIA COMPLETA - DATASET ACME")
print("=" * 80)

# Carregar dados (amostra grande para representatividade)
print("\n📦 Carregando dados...")
df_train = pd.read_csv("data/acme/train.csv", nrows=500000)
df_val = pd.read_csv("data/acme/val.csv", nrows=100000)
df_test = pd.read_csv("data/acme/test.csv", nrows=100000)

print(f"  Train: {len(df_train)} samples")
print(f"  Val:   {len(df_val)} samples")
print(f"  Test:  {len(df_test)} samples")

# Combinar para análise
df = pd.concat([df_train, df_val, df_test], ignore_index=True)
print(f"  Total: {len(df)} samples")

# ============================================================================
# 1. ANÁLISE BÁSICA
# ============================================================================
print("\n" + "=" * 80)
print("1. DISTRIBUIÇÃO DE CLASSES")
print("=" * 80)

label_counts = df['malicious'].value_counts()
label_pct = df['malicious'].value_counts(normalize=True) * 100
print(f"\n  Benign (0):    {label_counts.get(0, 0):,} ({label_pct.get(0, 0):.2f}%)")
print(f"  Malicious (1): {label_counts.get(1, 0):,} ({label_pct.get(1, 0):.2f}%)")
print(f"  Ratio:         {label_counts.get(1, 0) / max(label_counts.get(0, 0), 1):.3f}")

# Por split
for name, split_df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
    n_mal = split_df['malicious'].sum()
    n_total = len(split_df)
    print(f"  {name}: {n_mal:,} malicious / {n_total:,} total ({n_mal/n_total*100:.2f}%)")

# ============================================================================
# 2. ANÁLISE DE FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("2. FEATURES DISPONÍVEIS")
print("=" * 80)

feature_cols = [c for c in df.columns if c not in ['prompt', 'malicious', 'name']]
print(f"\n  Total features: {len(feature_cols)}")
print(f"  Features:\n  {feature_cols}")

# Categorizar features
lex_features = [c for c in feature_cols if c.startswith('lex_')]
dns_features = [c for c in feature_cols if 'Count' in c or 'Missing' in c or c in ['A Count', 'IP Count', 'MX Count', 'NS Count', 'AAAA Count', 'CNAME Count', 'ASN Count']]
ttl_features = [c for c in feature_cols if 'TTL' in c or 'med' in c.lower() or 'std' in c.lower() or 'distinct' in c.lower()]
soa_features = [c for c in feature_cols if c.startswith('SOA ') or 'soa' in c.lower() or c in ['SOA Retry', 'SOA Minimum', 'SOA Missing', 'SOA Refresh', 'SOA Expire Length', 'SOA Serial Length', 'SOA Medium TTL', 'SOA TTL Count', 'SOA TTL Standard Deviation']]
whois_features = [c for c in feature_cols if 'whois' in c.lower()]
risk_features = [c for c in feature_cols if c in ['CCR', 'CCA', 'has_country']]
time_features = [c for c in feature_cols if c in ['lifetime', 'active_time']]

print(f"\n  Lexicais:      {len(lex_features)} -> {lex_features}")
print(f"  DNS:           {len(dns_features)} -> {dns_features}")
print(f"  TTL:           {len(ttl_features)} -> {ttl_features}")
print(f"  SOA:           {len(soa_features)} -> {soa_features}")
print(f"  WHOIS:         {len(whois_features)} -> {whois_features}")
print(f"  Risk (CCR/CCA):{len(risk_features)} -> {risk_features}")
print(f"  Time:          {len(time_features)} -> {time_features}")

# ============================================================================
# 3. ANÁLISE POR GRUPO DE FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("3. ESTATÍSTICAS DESCRITIVAS POR GRUPO")
print("=" * 80)

for group_name, features in [
    ("Lexicais", lex_features[:6]),
    ("DNS Records", dns_features[:6]),
    ("TTL", [c for c in ttl_features if 'med' in c.lower()][:4]),
    ("Risk", risk_features),
    ("WHOIS/Time", whois_features + time_features),
]:
    if not features:
        continue
    print(f"\n  --- {group_name} ---")
    stats = df[features].describe().T
    print(stats[['mean', 'std', 'min', '50%', 'max']].to_string())

# ============================================================================
# 4. CORRELAÇÃO COM LABEL
# ============================================================================
print("\n" + "=" * 80)
print("4. CORRELAÇÃO FEATURE × LABEL (Point-Biserial)")
print("=" * 80)

correlations = {}
for col in feature_cols:
    if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
        corr = df[col].corr(df['malicious'])
        correlations[col] = abs(corr)

sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

print(f"\n  Top 20 features por correlação absoluta com 'malicious':")
for i, (col, corr_val) in enumerate(sorted_corr[:20], 1):
    bar = "█" * int(corr_val * 50)
    print(f"  {i:2d}. {col:40s} |{bar:50s}| {corr_val:.4f}")

# ============================================================================
# 5. VALORES NULOS / MISSING
# ============================================================================
print("\n" + "=" * 80)
print("5. VALORES NULOS POR FEATURE")
print("=" * 80)

null_counts = df[feature_cols].isnull().sum()
null_pct = (null_counts / len(df)) * 100
null_df = pd.DataFrame({'nulls': null_counts, 'pct': null_pct})
null_df = null_df[null_df['nulls'] > 0].sort_values('nulls', ascending=False)

if len(null_df) > 0:
    for col, row in null_df.iterrows():
        print(f"  {col:40s}: {row['nulls']:>8,} ({row['pct']:.2f}%)")
else:
    print("  Nenhum valor nulo encontrado!")

# ============================================================================
# 6. FEATURE IMPORTANCE COM RANDOM FOREST (amostra)
# ============================================================================
print("\n" + "=" * 80)
print("6. FEATURE IMPORTANCE (Random Forest - 100k amostras)")
print("=" * 80)

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Preparar dados
sample = df.sample(n=min(100000, len(df)), random_state=42)
X = sample[feature_cols].copy()
y = sample['malicious']

# Fill NaN com mediana
for col in X.columns:
    if X[col].isnull().any():
        X[col] = X[col].fillna(X[col].median())

rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

print(f"\n  Top 25 features por importância:")
for i, (col, imp) in enumerate(importances.head(25).items(), 1):
    bar = "█" * int(imp * 80)
    print(f"  {i:2d}. {col:40s} |{bar:40s}| {imp:.4f}")

# Cumulative importance
cum_imp = importances.cumsum()
print(f"\n  Features necessárias para 80% da importância:")
n_80 = (cum_imp >= 0.80).idxmax()
n_80_idx = list(importances.index).index(n_80) + 1
print(f"  → Top {n_80_idx} features (até '{n_80}'): {cum_imp[n_80]*100:.1f}%")

print(f"\n  Features necessárias para 95% da importância:")
n_95 = (cum_imp >= 0.95).idxmax()
n_95_idx = list(importances.index).index(n_95) + 1
print(f"  → Top {n_95_idx} features (até '{n_95}'): {cum_imp[n_95]*100:.1f}%")

# ============================================================================
# 7. REDUNDÂNCIA / MULTICOLINEARIDADE
# ============================================================================
print("\n" + "=" * 80)
print("7. REDUNDÂNCIA ENTRE FEATURES (correlação > 0.8)")
print("=" * 80)

corr_matrix = df[feature_cols].corr()
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        val = abs(corr_matrix.iloc[i, j])
        if val > 0.8:
            high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], val))

high_corr.sort(key=lambda x: x[2], reverse=True)

if high_corr:
    print(f"\n  {len(high_corr)} pares com correlação > 0.8:")
    for c1, c2, val in high_corr[:20]:
        print(f"  {c1:40s} ↔ {c2:40s} = {val:.4f}")
else:
    print("  Nenhuma correlação alta encontrada!")

# ============================================================================
# 8. ANÁLISE DO PROMPT
# ============================================================================
print("\n" + "=" * 80)
print("8. ANÁLISE DO PROMPT")
print("=" * 80)

prompt_sample = df['prompt'].iloc[0]
print(f"\n  Exemplo de prompt:")
for line in prompt_sample.split('\n'):
    print(f"    {line}")

# Contar seções do prompt
sections = defaultdict(int)
for prompt in df['prompt'].dropna().head(10000):
    for line in prompt.split('\n'):
        if line.startswith('['):
            section = line.split(']')[0] + ']'
            sections[section] += 1

print(f"\n  Seções encontradas nos prompts:")
for section, count in sorted(sections.items(), key=lambda x: x[1], reverse=True):
    print(f"    {section:20s}: {count/10000*100:.1f}% dos prompts")

# ============================================================================
# 9. ANÁLISE DE SUBGRUPOS - O QUE CARACTERIZA MALICIOSOS?
# ============================================================================
print("\n" + "=" * 80)
print("9. DIFERENÇAS BENIGNO vs MALICIOSO")
print("=" * 80)

benign = df[df['malicious'] == 0][feature_cols].describe()
malicious = df[df['malicious'] == 1][feature_cols].describe()

# Features com maior diferença relativa
diffs = []
for col in feature_cols:
    if col in benign.index and col in malicious.index:
        mean_b = benign.loc[col, 'mean']
        mean_m = malicious.loc[col, 'mean']
        if mean_b != 0:
            diff = abs(mean_m - mean_b) / abs(mean_b)
        else:
            diff = abs(mean_m - mean_b)
        diffs.append((col, mean_b, mean_m, diff))

diffs.sort(key=lambda x: x[3], reverse=True)

print(f"\n  Top 15 features com maior diferença entre benigno e malicioso:")
print(f"  {'Feature':40s} {'Benign':>10s} {'Malicious':>10s} {'Diff Rel':>10s}")
print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*10}")
for col, mean_b, mean_m, diff in diffs[:15]:
    print(f"  {col:40s} {mean_b:10.4f} {mean_m:10.4f} {diff:10.4f}")

# ============================================================================
# 10. CORRELAÇÃO ENTRE FEATURES DO PROMPT E PERFORMANCE ESPERADA
# ============================================================================
print("\n" + "=" * 80)
print("10. FEATURES COM BAIXA IMPORTANCE (< 0.001)")
print("=" * 80)

low_imp = importances[importances < 0.001]
if len(low_imp) > 0:
    print(f"\n  {len(low_imp)} features com importância < 0.001:")
    for col, imp in low_imp.items():
        print(f"    {col:40s}: {imp:.6f}")
    print(f"\n  → Poderiam ser removidas do prompt sem perda significativa!")
else:
    print("\n  Todas as features têm importância >= 0.001")

# ============================================================================
# 11. RECOMENDAÇÕES
# ============================================================================
print("\n" + "=" * 80)
print("11. RECOMENDAÇÕES PARA MELHORAR PERFORMANCE")
print("=" * 80)

print("\n  ✅ BASELINE ATUAL:")
print(f"     DeBERTa 460k: 91.44% accuracy, 96.89% AUC")
print(f"     ML (features): 90.7%")
print(f"     ML (só DNS+lex): 85%")

print("\n  🔍 INSIGHTS DA ANÁLISE:")

# Top 5 features mais importantes
top5 = list(importances.head(5).index)
print(f"\n     Top 5 features mais importantes: {top5}")

# Features redundantes
if high_corr:
    print(f"     {len(high_corr)} pares redundantes detectados")

# Features inúteis
if len(low_imp) > 0:
    print(f"     {len(low_imp)} features com importância desprezível")

print("\n  💡 SUGESTÕES:")
print("     1. Remover features com importance < 0.001 do prompt")
print("     2. Adicionar features derivadas (ex: whois + CCR combinados)")
print("     3. Testar prompts com ordem diferente (mais importantes primeiro)")
print("     4. Testar ensembling: DeBERTa + LLM (cada um capta padrões diferentes)")
print("     5. Adicionar feature 'domain age' se whois disponível")
print("     6. Testar prompt mais conciso (só top 10 features)")
print("     7. Testar modelo maior (Llama 3 8B, Qwen 3B+)")

print("\n" + "=" * 80)
print("ANÁLISE CONCLUÍDA")
print("=" * 80)
