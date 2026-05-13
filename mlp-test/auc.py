import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset, DataLoader
from MLP import MultiLayerPerceptron, ResNetTabular
import torch.nn as nn
import os
import csv

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
MODELS_DIR = "training_results/models"

# ── Carrega e escala os dados ──────────────────────────────────────────────────
train_df = pd.read_csv("../data/acme/fixed_train.csv", index_col=False)
val_df   = pd.read_csv("../data/acme/fixed_val.csv",   index_col=False)
test_df  = pd.read_csv("../data/acme/fixed_test.csv",  index_col=False)

feature_cols = [c for c in train_df.columns if c not in ('malicious', 'name', 'prompt')]
scaler = MinMaxScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
val_df[feature_cols]   = scaler.transform(val_df[feature_cols])
test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

n_features = len(feature_cols)

X_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32)
y_test = torch.tensor(test_df['malicious'].values,  dtype=torch.long)
test_ds     = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=8, pin_memory=True)


# ── Coleta probabilidades do modelo ───────────────────────────────────────────
def get_probs(model: torch.nn.Module) -> tuple[np.ndarray, np.ndarray]:
    """Retorna (y_true, y_prob_classe1) para o test set."""
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            logits  = model(X_batch)
            probs   = torch.softmax(logits, dim=1)[:, 1]  # prob da classe malicioso
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y_batch.numpy())

    return np.concatenate(all_labels), np.concatenate(all_probs)


# ── Modelos a avaliar ──────────────────────────────────────────────────────────
checkpoints = [
    {
        'name':       'MLP',
        'path':       os.path.join(MODELS_DIR, 'best_mlp.pt'),
        'model':      MultiLayerPerceptron([n_features, 256, 128, 64, 2], 0.0, nn.ReLU),
    },
    {
        'name':       'ResNet',
        'path':       os.path.join(MODELS_DIR, 'best_resnet.pt'),
        'model':      ResNetTabular(input_dim=n_features, hidden_dim=128,
                                    num_blocks=2, num_classes=2, dropout=0.2),
    },
]

# ── Avalia e salva ─────────────────────────────────────────────────────────────
auc_csv = os.path.join("training_results", "auc_results.csv")
rows    = []

print(f"\n{'='*50}")
for ckpt in checkpoints:
    if not os.path.exists(ckpt['path']):
        print(f"[{ckpt['name']}] checkpoint não encontrado: {ckpt['path']}")
        continue

    model = ckpt['model'].to(DEVICE)
    model.load_state_dict(torch.load(ckpt['path'], map_location=DEVICE))

    y_true, y_prob = get_probs(model)
    auc = roc_auc_score(y_true, y_prob)

    print(f"[{ckpt['name']}]  AUC = {auc:.4f}")
    rows.append({'model': ckpt['name'], 'auc': round(auc, 6), 'checkpoint': ckpt['path']})

print(f"{'='*50}\n")

with open(auc_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['model', 'auc', 'checkpoint'])
    writer.writeheader()
    writer.writerows(rows)

print(f"AUC salva em: {auc_csv}")