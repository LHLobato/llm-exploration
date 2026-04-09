#!/usr/bin/env python3
"""Análise avançada: redundância, engenharia de features, e impacto no prompt"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

print("=" * 80)
print("ANÁLISE AVANÇADA - REDUNDÂNCIA E FEATURE ENGINEERING")
print("=" * 80)

# Carregar amostra
df = pd.read_csv("data/acme/train.csv", nrows=300000)
feature_cols = [c for c in df.columns if c not in ['prompt', 'malicious', 'name']]

# ============================================================================
# 1. ANALISAR REDUNDÂNCIAS DETECTADAS
# ============================================================================
print("\n" + "=" * 80)
print("1. REDUNDÂNCIAS DETECTADAS (o que pode ser removido)")
print("=" * 80)

redundant_pairs = [
    ("lex_vowel_ratio", "lex_consonant_ratio", 1.0, "Soma = 1.0, redundância perfeita"),
    ("SOA Missing", "SOA TTL Count", 1.0, "Idênticos"),
    ("SOA Medium TTL", "SOA TTL Standard Deviation", 1.0, "Idênticos"),
    ("A Count", "A TTL Count", 0.99, "Quase idênticos"),
    ("IP Count", "ASN Count", 0.92, "Alta correlação"),
    ("lifetime", "active_time", 0.89, "Alta correlação temporal"),
    ("CCR", "CCA", 0.84, "Métricas de risco similares"),
]

for f1, f2, corr, note in redundant_pairs:
    print(f"\n  {f1} ↔ {f2} (r={corr:.2f})")
    print(f"    → {note}")
    
    # Qual manter?
    imp1 = df[f1].corr(df['malicious'])
    imp2 = df[f2].corr(df['malicious'])
    print(f"    → Corr com label: {f1}={abs(imp1):.4f}, {f2}={abs(imp2):.4f}")
    keeper = f1 if abs(imp1) >= abs(imp2) else f2
    print(f"    → ✅ Manter: {keeper}")

# ============================================================================
# 2. TESTAR SUBCONJUNTOS DE FEATURES COM RF
# ============================================================================
print("\n" + "=" * 80)
print("2. PERFORMANCE RF COM DIFERENTES SUBCONJUNTOS DE FEATURES")
print("=" * 80)

def test_features(name, features, df_sample):
    X = df_sample[features].copy()
    y = df_sample['malicious']
    for c in X.columns:
        if X[c].isnull().any():
            X[c] = X[c].fillna(X[c].median())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    auc_rf = rf.predict_proba(X_test)[:, 1]
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, auc_rf)
    print(f"  {name:40s}: {len(features):3d} features → Acc={acc:.4f}, AUC={auc:.4f}")
    return acc, auc

sample = df.sample(n=100000, random_state=42)

# Todas features
test_features("Todas features", feature_cols, sample)

# Só lexica
lex = [c for c in feature_cols if c.startswith('lex_')]
test_features("Só lexica", lex, sample)

# Só DNS
dns = [c for c in feature_cols if 'Count' in c or 'Missing' in c or 'Medium' in c or 'Standard' in c 
       or c in ['SOA Retry', 'SOA Minimum', 'SOA Refresh', 'SOA Expire Length', 'SOA Serial Length']]
test_features("Só DNS+TTL+SOA", dns, sample)

# Só risk+whois
risk = ['CCR', 'CCA', 'has_country', 'has_whois', 'lifetime', 'active_time']
test_features("Só risk+whois+time", risk, sample)

# Top 15 por importance
top15 = ['CCA', 'CCR', 'lifetime', 'has_whois', 'has_country', 'active_time', 
         'SOA Medium TTL', 'SOA TTL Standard Deviation', 'SOA Minimum', 'SOA Retry',
         'SOA Refresh', 'NS Count', 'lex_length', 'SOA Expire Length', 'SOA Serial Length']
test_features("Top 15 importance", top15, sample)

# Top 10 (só as críticas)
top10 = ['CCA', 'CCR', 'lifetime', 'has_whois', 'has_country', 'active_time', 
         'SOA Medium TTL', 'SOA Minimum', 'NS Count', 'lex_length']
test_features("Top 10 importance", top10, sample)

# Top 7 (ultra-conciso)
top7 = ['CCA', 'CCR', 'lifetime', 'has_whois', 'has_country', 'active_time', 'SOA Medium TTL']
test_features("Top 7 importance", top7, sample)

# Sem redundantes
non_redundant = [c for c in feature_cols if c not in [
    'lex_vowel_ratio',  # redundante com consonant_ratio
    'SOA TTL Count',    # idêntico a SOA Missing
    'SOA TTL Standard Deviation',  # idêntico a SOA Medium TTL
    'A TTL Count',      # 0.99 com A Count
    'ASN Count',        # 0.92 com IP Count
    'active_time',      # 0.89 com lifetime
    'CCA',              # 0.84 com CCR
    'SOA Expire Length', # 0.97 com SOA Missing
    'SOA Serial Length', # 0.83 com SOA Missing
    'AAAA Count',       # correlacionado com IP Count
    'lex_max_digit_seq', # correlacionado com digit_ratio
    'lex_digit_count',   # correlacionado com digit_ratio
]]
test_features("Sem redundantes", non_redundant, sample)

# ============================================================================
# 3. FEATURES DERIVADAS PROMISSORAS
# ============================================================================
print("\n" + "=" * 80)
print("3. FEATURES DERIVADAS SUGERIDAS")
print("=" * 80)

# Testar features derivadas
df_test = df.copy()
sample_test = sample.copy()

# 1. Domain age proxy (lifetime + whois)
df_test['whois_age_interaction'] = df_test['has_whois'] * df_test['lifetime']
sample_test['whois_age_interaction'] = sample_test['has_whois'] * sample_test['lifetime']
test_features("+ whois*lifetime", feature_cols + ['whois_age_interaction'], sample_test)

# 2. Risk composite
df_test['risk_composite'] = df_test['CCR'] * df_test['CCA'] * df_test['has_country']
sample_test['risk_composite'] = sample_test['CCR'] * sample_test['CCA'] * sample_test['has_country']
test_features("+ risk_composite", feature_cols + ['risk_composite'], sample_test)

# 3. DNS diversity
df_test['dns_diversity'] = (df_test['A Count'] > 0).astype(int) + \
                           (df_test['AAAA Count'] > 0).astype(int) + \
                           (df_test['MX Count'] > 0).astype(int) + \
                           (df_test['NS Count'] > 0).astype(int) + \
                           (df_test['CNAME Count'] > 0).astype(int)
sample_test['dns_diversity'] = (sample_test['A Count'] > 0).astype(int) + \
                               (sample_test['AAAA Count'] > 0).astype(int) + \
                               (sample_test['MX Count'] > 0).astype(int) + \
                               (sample_test['NS Count'] > 0).astype(int) + \
                               (sample_test['CNAME Count'] > 0).astype(int)
test_features("+ dns_diversity", feature_cols + ['dns_diversity'], sample_test)

# 4. Entropy/length ratio (complexidade por caractere)
df_test['entropy_per_char'] = df_test['lex_entropy'] / df_test['lex_length'].clip(lower=1)
sample_test['entropy_per_char'] = sample_test['lex_entropy'] / sample_test['lex_length'].clip(lower=1)
test_features("+ entropy_per_char", feature_cols + ['entropy_per_char'], sample_test)

# 5. TTL consistency
df_test['ttl_consistency'] = df_test['SOA Medium TTL'] / (df_test['SOA TTL Standard Deviation'] + 1)
sample_test['ttl_consistency'] = sample_test['SOA Medium TTL'] / (sample_test['SOA TTL Standard Deviation'] + 1)
test_features("+ ttl_consistency", feature_cols + ['ttl_consistency'], sample_test)

# ============================================================================
# 4. ANÁLISE DO PROMPT ATUAL
# ============================================================================
print("\n" + "=" * 80)
print("4. ANÁLISE DO PROMPT ATUAL")
print("=" * 80)

prompt = df['prompt'].iloc[0]
print(f"\n  Prompt atual:")
for line in prompt.split('\n'):
    print(f"    {line}")

# Contar tokens aproximados
sections = ['name', 'entropy', 'whois', 'risk', 'dns', 'ttl', 'soa']
print(f"\n  Seções no prompt: {len(sections)}")
print(f"  Features por seção:")
print(f"    [name]:     1 (domain name)")
print(f"    [entropy]:  1 (lex_entropy)")
print(f"    [whois]:    1 (present=yes/no)")
print(f"    [risk]:     3 (CCR, CCA, country)")
print(f"    [dns]:      7 (A, AAAA, MX, NS, CNAME, IPs, ASNs, a_missing, mx_missing)")
print(f"    [ttl]:      5 (a_med, a_std, a_distinct, mx_med, aaaa_med)")
print(f"    [soa]:      8 (missing, refresh, retry, min, expire_len, serial_len, med, std, distinct)")
print(f"  Total: ~26 features no prompt")

# ============================================================================
# 5. PROMPT OTIMIZADO SUGERIDO
# ============================================================================
print("\n" + "=" * 80)
print("5. PROMPT OTIMIZADO SUGERIDO")
print("=" * 80)

print(f"\n  📌 ORDEM ATUAL (por aparecimento):")
print(f"     name → entropy → whois → risk → dns → ttl → soa")

print(f"\n  📌 ORDEM POR IMPORTÂNCIA (RF feature importance):")
top_features_order = ['risk (CCR, CCA)', 'lifetime', 'whois', 'country', 'soa', 'NS Count', 'lex_length', 'entropy']
for i, f in enumerate(top_features_order, 1):
    print(f"     {i}. {f}")

print(f"\n  📌 PROMPT REORGANIZADO SUGERIDO:")
print(f"     [name] + [risk] + [whois] + [time] + [soa] + [dns] + [ttl] + [entropy]")
print(f"     → Features mais importantes vêm primeiro (atenção do modelo)")

print(f"\n  📌 FEATURES QUE PODEM SER ADICIONADAS AO PROMPT:")
print(f"     - lifetime/active_time (não estão no prompt atual!)")
print(f"     - has_country (não está explicitamente no prompt!)")

# ============================================================================
# 6. TESTAR IMPACTO DAS FEATURES FALTANTES NO PROMPT
# ============================================================================
print("\n" + "=" * 80)
print("6. FEATURES PRESENTES NOS DADOS MAS AUSENTES NO PROMPT")
print("=" * 80)

# Features do prompt atual (pelo formato)
prompt_features = {
    'lex_entropy', 'has_whois', 'CCR', 'CCA', 'has_country',
    'A Count', 'AAAA Count', 'MX Count', 'NS Count', 'CNAME Count', 'IP Count', 'ASN Count',
    'A Missing', 'MX Missing',
    'A Medium TTL', 'A TTL Standard Deviation', 'A TTL Count',
    'MX Medium TTL', 'AAAA Medium TTL',
    'SOA Missing', 'SOA Refresh', 'SOA Retry', 'SOA Minimum',
    'SOA Expire Length', 'SOA Serial Length', 'SOA Medium TTL', 
    'SOA TTL Standard Deviation', 'SOA TTL Count'
}

# Features ausentes
missing_from_prompt = set(feature_cols) - prompt_features
missing_important = [(f, abs(df[f].corr(df['malicious']))) for f in missing_from_prompt]
missing_important.sort(key=lambda x: x[1], reverse=True)

print(f"\n  Features NÃO usadas no prompt ({len(missing_from_prompt)}):")
for f, corr in missing_important:
    bar = "█" * int(corr * 50)
    print(f"    {f:40s} |{bar:50s}| {corr:.4f}")

print(f"\n  ⚠️ CRÍTICAS FALTANDO: lifetime, active_time (corr=0.26-0.28)")

# ============================================================================
# 7. RESUMO FINAL
# ============================================================================
print("\n" + "=" * 80)
print("7. RESUMO FINAL - O QUE TESTAR PARA MAXIMIZAR PERFORMANCE")
print("=" * 80)

print("""
  🔥 MÁXIMA PRIORIDADE (maior impacto esperado):
  
  1. ADICIONAR lifetime/active_time ao prompt
     → Features com corr=0.26-0.28, importance=0.065-0.075
     → Atualmente NÃO estão no prompt!
     → Impacto esperado: +1-2% accuracy
  
  2. REORDENAR prompt por importância
     → [name] → [risk] → [time] → [whois] → [soa] → [dns] → [ttl]
     → LLMs atendem mais ao início do prompt
     → Impacto esperado: +0.5-1%
  
  3. REMOVER features redundantes do prompt
     → lex_vowel_ratio (redundante com consonant)
     → SOA TTL Count (idêntico a SOA Missing)
     → Impacto: prompt mais limpo, mesmo resultado
  
  ⚡ MÉDIA PRIORIDADE:
  
  4. Adicionar features derivadas:
     → risk_composite = CCR * CCA * has_country
     → dns_diversity = count de tipos de records presentes
     → Impacto esperado: +0.5-1%
  
  5. Usar modelo maior (Qwen 1.5B/3B, Llama 3 8B)
     → Mais capacidade de captar interações complexas
     → Impacto esperado: +1-3%
  
  📊 GANHO TOTAL ESTIMADO:
  
     DeBERTa 460k atual:          91.44%
     + lifetime/time:             +1-2% → ~92.5-93.5%
     + reordenação:               +0.5-1% → ~93-94%
     + features derivadas:        +0.5% → ~93.5-94.5%
     + modelo maior (8B):         +1-2% → ~94.5-96%
  
     POTENCIAL MÁXIMO: ~95-96% accuracy
""")

print("=" * 80)
print("ANÁLISE CONCLUÍDA")
print("=" * 80)
