import torch
from torch.utils.data import TensorDataset, DataLoader
from MLP import MultiLayerPerceptron, train, test, ResNetTabular
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import os
import csv
from datetime import datetime

BATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "training_results"
MODELS_DIR  = os.path.join(RESULTS_DIR, "models")
RESULTS_CSV = os.path.join(RESULTS_DIR, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

os.makedirs(MODELS_DIR, exist_ok=True)

def prepare_dataset(df: pd.DataFrame, n_samples: int | None = None,
                    random_state: int = 42) -> TensorDataset:
    if n_samples is not None and n_samples < len(df):
        df = (
            df.groupby('malicious', group_keys=False)
              .apply(lambda g: g.sample(
                  min(len(g), round(n_samples * len(g) / len(df))),
                  random_state=random_state))
        )
    X = torch.tensor(
        df.drop(columns=['malicious', 'name', 'prompt'], errors='ignore').values,
        dtype=torch.float32
    )
    y = torch.tensor(df['malicious'].values, dtype=torch.long)
    return TensorDataset(X, y)

def make_loader(ds: TensorDataset, shuffle: bool = False) -> DataLoader:
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=8, pin_memory=True)

def log_result(writer, row: dict):
    writer.writerow(row)

CSV_FIELDS = [
    'model_type', 'architecture', 'hidden_dim', 'num_blocks',
    'activation', 'regularization', 'lr', 'weight_decay', 'optimizer',
    'epochs', 'val_accuracy',
    'test_accuracy', 'test_loss', 'test_precision', 'test_recall', 'test_f1',
]

if __name__ == "__main__":  
    train_df = pd.read_csv("../data/acme/fixed_train.csv", index_col=False)
    val_df   = pd.read_csv("../data/acme/fixed_val.csv",   index_col=False)
    test_df  = pd.read_csv("../data/acme/fixed_test.csv",  index_col=False)

    feature_cols = [c for c in train_df.columns if c not in ('malicious', 'name', 'prompt')]
    scaler = MinMaxScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols]   = scaler.transform(val_df[feature_cols])
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

    n_features = len(feature_cols)

    train_ds = prepare_dataset(train_df, len(train_df))
    val_ds   = prepare_dataset(val_df,   len(val_df))
    test_ds  = prepare_dataset(test_df,  len(test_df))

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader   = make_loader(val_ds)
    test_loader  = make_loader(test_ds)

    EPOCHS       = 100
    LR           = 1e-3
    WEIGHT_DECAY = 1e-4
    OPTIMIZER    = "Adam"

    runs = [
        {
            'model_type':     'MLP',
            'architecture':   [n_features, 256, 128, 64, 2],
            'activation':     nn.ReLU,
            'regularization': 0.0,
            'model_kwargs':   {},          
        },
        {
            'model_type':     'ResNet',
            'architecture':   None,
            'hidden_dim':     128,
            'num_blocks':     2,
            'activation':     None,
            'regularization': 0.2,
            'model_kwargs':   {
                'input_dim':  n_features,
                'hidden_dim': 128,
                'num_blocks': 2,
                'num_classes': 2,
                'dropout':    0.2,
            },
        },
    ]

    with open(RESULTS_CSV, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS, extrasaction='ignore')
        writer.writeheader()

        for run in runs:
            mtype = run['model_type']
            print(f"\n{'='*60}")
            print(f"Treinando {mtype}")
            print(f"{'='*60}")

            if mtype == 'MLP':
                model = MultiLayerPerceptron(
                    run['architecture'],
                    run['regularization'],
                    run['activation'],
                ).to(DEVICE)
            else:
                model = ResNetTabular(**run['model_kwargs']).to(DEVICE)

            model, val_acc, _, _ = train(
                model, EPOCHS, LR,
                train_loader, val_loader,
                WEIGHT_DECAY, DEVICE, OPTIMIZER,
            )


            ckpt_path = os.path.join(MODELS_DIR, f"best_{mtype.lower()}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint salvo em: {ckpt_path}")


            loss, acc, prec, rec, f1 = test(model, test_loader, DEVICE)

            print(f"\n[{mtype}] Resultados no test set:")
            print(f"  accuracy  = {acc:.4f}")
            print(f"  f1        = {f1:.4f}")
            print(f"  precision = {prec:.4f}")
            print(f"  recall    = {rec:.4f}")
            print(f"  loss      = {loss:.4f}")

            # ── Log CSV ────────────────────────────────────────────────────────
            row = {
                'model_type':     mtype,
                'architecture':   str(run.get('architecture') or run.get('model_kwargs')),
                'hidden_dim':     run.get('model_kwargs', {}).get('hidden_dim', ''),
                'num_blocks':     run.get('model_kwargs', {}).get('num_blocks', ''),
                'activation':     run['activation'].__name__ if run['activation'] else '',
                'regularization': run['regularization'],
                'lr':             LR,
                'weight_decay':   WEIGHT_DECAY,
                'optimizer':      OPTIMIZER,
                'epochs':         EPOCHS,
                'val_accuracy':   round(val_acc, 6),
                'test_accuracy':  round(acc,     6),
                'test_loss':      round(loss,    6),
                'test_precision': round(prec,    6),
                'test_recall':    round(rec,     6),
                'test_f1':        round(f1,      6),
            }
            log_result(writer, row)
            csvfile.flush()   

    print(f"\nResultados salvos em: {RESULTS_CSV}")
    print(f"Modelos salvos em:    {MODELS_DIR}/")