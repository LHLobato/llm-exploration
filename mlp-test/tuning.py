import torch
from torch.utils.data import TensorDataset, DataLoader
from MLP import MultiLayerPerceptron, train, test, ResNetTabular
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import os
import csv
from datetime import datetime

# ── Constantes ────────────────────────────────────────────────────────────────
DEVICE         = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS         = 15
BATCH_SIZE     = 64
N_TRAIN        = 10_000   # reduzido de 25k
N_VAL          = 3_000    # reduzido de 5k
N_TEST         = 3_000    # reduzido de 5k
RESULTS_DIR    = "grid_search_results"
MODELS_DIR     = os.path.join(RESULTS_DIR, "models")
RESULTS_CSV    = os.path.join(RESULTS_DIR, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

os.makedirs(MODELS_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def prepare_dataset(df: pd.DataFrame, n_samples: int | None = None,
                    random_state: int = 42) -> TensorDataset:
    if n_samples is not None and n_samples < len(df):
        # Amostragem estratificada por classe para manter proporção malicioso/benigno
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


# ── Grid Search ───────────────────────────────────────────────────────────────
def grid_search(hyperparams: dict,
                train_df: pd.DataFrame,
                val_df:   pd.DataFrame,
                test_df:  pd.DataFrame):

    train_ds = prepare_dataset(train_df, N_TRAIN)
    val_ds   = prepare_dataset(val_df,   N_VAL)
    test_ds  = prepare_dataset(test_df,  N_TEST)

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader   = make_loader(val_ds)
    test_loader  = make_loader(test_ds)

    # Melhores por categoria
    best = {
        'MLP':    {'val_acc': float('-inf'), 'params': {}, 'model': None},
        'ResNet': {'val_acc': float('-inf'), 'params': {}, 'model': None},
    }

    # Cabeçalho do CSV
    csv_fields = [
        'model_type', 'lr', 'weight_decay', 'optimizer', 'regularization',
        'architecture', 'activation',        # MLP-specific
        'hidden_dim', 'num_blocks',           # ResNet-specific
        'val_accuracy', 'test_accuracy',
        'test_loss', 'test_precision', 'test_recall', 'test_f1',
    ]

    with open(RESULTS_CSV, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()

        total_combinations = (
            len(hyperparams['optimizers']) *
            len(hyperparams['weight_decay']) *
            len(hyperparams['lr'])
        )
        run_idx = 0

        for p_o in hyperparams['optimizers']:
            for p_wd in hyperparams['weight_decay']:
                for p_lr in hyperparams['lr']:

                    # ── MLP ──────────────────────────────────────────
                    for p_f in hyperparams['functions']:
                        for p_r in hyperparams['regularization']:
                            for p_a in hyperparams['mlp_architectures']:
                                run_idx += 1
                                tag = (f"[MLP {run_idx}] opt={p_o} lr={p_lr} wd={p_wd} "
                                       f"act={p_f.__name__} drop={p_r} arch={p_a}")
                                print(tag)

                                model = MultiLayerPerceptron(p_a, p_r, p_f).to(DEVICE)
                                model, val_acc, _, _ = train(
                                    model, EPOCHS, p_lr,
                                    train_loader, val_loader,
                                    p_wd, DEVICE, p_o
                                )

                                row = {
                                    'model_type':     'MLP',
                                    'lr':             p_lr,
                                    'weight_decay':   p_wd,
                                    'optimizer':      p_o,
                                    'regularization': p_r,
                                    'architecture':   str(p_a),
                                    'activation':     p_f.__name__,
                                    'val_accuracy':   round(val_acc, 6),
                                }

                                if val_acc > best['MLP']['val_acc']:
                                    best['MLP']['val_acc'] = val_acc
                                    best['MLP']['params']  = row.copy()
                                    best['MLP']['model']   = model
                                    # Salva checkpoint imediatamente (sobrescreve anterior)
                                    torch.save(model.state_dict(),
                                               os.path.join(MODELS_DIR, 'best_mlp.pt'))
                                    print(f"  ✔ Novo melhor MLP: val_acc={val_acc:.4f}")

                                # Métricas de teste só para o melhor ao final;
                                # por ora preenche como None para não desperdiçar GPU
                                row.update({k: None for k in
                                            ['test_accuracy','test_loss',
                                             'test_precision','test_recall','test_f1']})
                                log_result(writer, row)
                                csvfile.flush()

                    # ── ResNet Tabular ────────────────────────────────
                    for p_r in hyperparams['regularization']:
                        for p_a in hyperparams['resnet_architectures']:
                            run_idx += 1
                            tag = (f"[ResNet {run_idx}] opt={p_o} lr={p_lr} wd={p_wd} "
                                   f"drop={p_r} arch={p_a}")
                            print(tag)

                            model = ResNetTabular(
                                input_dim  = p_a['input_dim'],
                                hidden_dim = p_a['hidden_dim'],
                                num_blocks = p_a['num_blocks'],
                                num_classes= 2,
                                dropout    = p_r,
                            ).to(DEVICE)
                            model, val_acc, _, _ = train(
                                model, EPOCHS, p_lr,
                                train_loader, val_loader,
                                p_wd, DEVICE, p_o
                            )

                            row = {
                                'model_type':     'ResNet',
                                'lr':             p_lr,
                                'weight_decay':   p_wd,
                                'optimizer':      p_o,
                                'regularization': p_r,
                                'hidden_dim':     p_a['hidden_dim'],
                                'num_blocks':     p_a['num_blocks'],
                                'architecture':   str(p_a),
                                'val_accuracy':   round(val_acc, 6),
                            }

                            if val_acc > best['ResNet']['val_acc']:
                                best['ResNet']['val_acc'] = val_acc
                                best['ResNet']['params']  = row.copy()
                                best['ResNet']['model']   = model
                                torch.save(model.state_dict(),
                                           os.path.join(MODELS_DIR, 'best_resnet.pt'))
                                print(f"  ✔ Novo melhor ResNet: val_acc={val_acc:.4f}")

                            row.update({k: None for k in
                                        ['test_accuracy','test_loss',
                                         'test_precision','test_recall','test_f1']})
                            log_result(writer, row)
                            csvfile.flush()

        # ── Avaliação final dos melhores no test set ──────────────────────────
        print("\n" + "="*60)
        print("Avaliação final no test set")
        print("="*60)

        summary_rows = []
        for category, info in best.items():
            if info['model'] is None:
                continue
            loss, acc, prec, rec, f1 = test(info['model'], test_loader, DEVICE)
            print(f"\n[{category}] test_acc={acc:.4f}  f1={f1:.4f}  "
                  f"prec={prec:.4f}  rec={rec:.4f}")
            final_row = {
                **info['params'],
                'test_accuracy':   round(acc,  6),
                'test_loss':       round(loss, 6),
                'test_precision':  round(prec, 6),
                'test_recall':     round(rec,  6),
                'test_f1':         round(f1,   6),
            }
            log_result(writer, final_row)
            summary_rows.append(final_row)

    print(f"\nResultados salvos em: {RESULTS_CSV}")
    print(f"Modelos salvos em:    {MODELS_DIR}/")
    return summary_rows


# ── Setup ─────────────────────────────────────────────────────────────────────
train_df = pd.read_csv("../data/acme/fixed_train.csv", index_col=False)
val_df   = pd.read_csv("../data/acme/fixed_val.csv",   index_col=False)
test_df  = pd.read_csv("../data/acme/fixed_test.csv",  index_col=False)

feature_cols = [c for c in train_df.columns if c not in ('malicious', 'name', 'prompt')]
scaler = MinMaxScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
val_df[feature_cols]   = scaler.transform(val_df[feature_cols])
test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

n_features = len(feature_cols)

hyperparams = {
    'lr':           [1e-3, 1e-4],
    'weight_decay': [0.0, 1e-4],
    'optimizers':   ['Adam', 'AdamW'],

    'regularization': [0.0, 0.2],

    # MLP
    'functions': [nn.ReLU, nn.GELU],
    'mlp_architectures': [
        [n_features, 128, 64, 2],
        [n_features, 256, 128, 2],
        [n_features, 256, 128, 64, 2],
    ],

    # ResNet Tabular
    'resnet_architectures': [
        {'input_dim': n_features, 'hidden_dim': 128, 'num_blocks': 2},
        {'input_dim': n_features, 'hidden_dim': 256, 'num_blocks': 4},
        {'input_dim': n_features, 'hidden_dim': 512, 'num_blocks': 4},
    ],
}

results = grid_search(hyperparams, train_df, val_df, test_df)
print("\nResumo final:")
for r in results:
    print(r)