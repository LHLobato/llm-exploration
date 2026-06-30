# bert_ensemble_experiment.py

import os
import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from extractor import BertFeatureExtractor, registry

BERT_MODELS = ["BERT-Base", "distilBERT", "DEBERTa", "ModernBERT"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv",  type=str, required=True,
                        help="Caminho do CSV de resultados (ex: results/bert_ensemble.csv)")
    parser.add_argument("--model_name",   type=str, required=True, choices=BERT_MODELS)
    parser.add_argument("--model_path",   type=str, required=True,
                        help="Caminho HuggingFace ou local para o modelo BERT")
    parser.add_argument("--dataset",      type=str, required=True,
                        help="Nome do dataset (ex: domain-enriched)")
    parser.add_argument("--features_dir", type=str, default="features",
                        help="Diretório para salvar/carregar features")
    parser.add_argument("--data_csv",     type=str, required=True,
                        help="CSV com colunas 'text' e 'label' (dados já processados)")
    parser.add_argument("--text_col",     type=str, default="name",
                        help="Nome da coluna de texto no CSV")
    parser.add_argument("--label_col",    type=str, default="malicious",
                        help="Nome da coluna de label no CSV")
    parser.add_argument("--batch_size",   type=int, default=64)
    parser.add_argument("--num_workers",  type=int, default=4)
    parser.add_argument("--max_length",   type=int, default=200)
    return parser.parse_args()


def _append_csv(path: str, row: dict):
    df = pd.DataFrame([row])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def _already_evaluated(results_csv: str, model_name: str, dataset_name: str, clf_name: str) -> bool:
    if not os.path.exists(results_csv):
        return False
    df = pd.read_csv(results_csv)
    return (
        (df["Model"]   == model_name)  &
        (df["Dataset"] == dataset_name) &
        (df["Clf"]     == clf_name)
    ).any()


def _features_exist(features_dir: str, model_name: str, dataset_name: str) -> bool:
    for split in ("train", "val", "test"):
        path = os.path.join(features_dir, f"{split}-{model_name}-{dataset_name}.npz")
        if not os.path.exists(path):
            return False
    return True


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"Modelo: {args.model_name} | Dataset: {args.dataset}")
    print(f"Model path: {args.model_path}")
    print(f"{'='*60}")

    if not _features_exist(args.features_dir, args.model_name, args.dataset):
        df = pd.read_csv(args.data_csv, index_col=False)
        texts  = df[args.text_col].values
        labels = df[args.label_col].values

        X_train, X_temp, y_train, y_temp = train_test_split(
            texts, labels, test_size=0.30, random_state=0, stratify=labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=0, stratify=y_temp
        )

        extractor = BertFeatureExtractor(
            device=device,
            model_name=args.model_name,
            model_path=args.model_path,
            max_length=args.max_length,
        )
        extractor.set_datasets(
            X_train, y_train,
            X_val,   y_val,
            X_test,  y_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        extractor.extract_and_save(args.features_dir, args.dataset)
        del extractor
        torch.cuda.empty_cache()
    else:
        print("[SKIP] Features já existem para os três splits.")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = BertFeatureExtractor.load_features(
        args.features_dir, args.model_name, args.dataset
    )
    print(f"Features — train: {X_train.shape} | val: {X_val.shape} | test: {X_test.shape}")

    X_train = X_train.astype(np.float32)
    X_val   = X_val.astype(np.float32)
    X_test  = X_test.astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_val   = y_val.astype(np.int64)
    y_test  = y_test.astype(np.int64)

    X_fit = X_val 
    y_fit = y_val
    for clf_name, clf_factory in registry.items():
        if _already_evaluated(args.results_csv, args.model_name, args.dataset, clf_name):
            print(f"[SKIP] {clf_name} já avaliado.")
            continue

        print(f"  → {clf_name}...", end=" ", flush=True)
        try:
            clf = clf_factory()
            clf.fit(X_fit, y_fit)
            y_pred = clf.predict(X_test)

            try:
                if hasattr(clf, "predict_proba"):
                    y_prob = clf.predict_proba(X_test)[:, 1]
                elif hasattr(clf, "decision_function"):
                    y_prob = clf.decision_function(X_test)
                else:
                    y_prob = None
            except Exception:
                y_prob = None

            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec  = recall_score(y_test, y_pred, zero_division=0)
            f1   = f1_score(y_test, y_pred, zero_division=0)
            auc  = roc_auc_score(y_test, y_prob) if y_prob is not None else None

            print(f"acc={acc:.4f}" + (f" | auc={auc:.4f}" if auc else ""))

            _append_csv(args.results_csv, {
                "Model":     args.model_name,
                "Dataset":   args.dataset,
                "Clf":       clf_name,
                "Accuracy":  acc,
                "Precision": prec,
                "Recall":    rec,
                "F1":        f1,
                "ROC_AUC":   auc,
            })

        except Exception as e:
            print(f"ERRO: {e}")
            _append_csv(args.results_csv, {
                "Model": args.model_name, "Dataset": args.dataset,
                "Clf": clf_name,
                "Accuracy": None, "Precision": None,
                "Recall": None, "F1": None, "ROC_AUC": None,
            })


if __name__ == "__main__":
    main(get_args())