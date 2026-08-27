"""
Avalia um modelo salvo (.joblib) em um conjunto de teste: métricas de
classificação e métricas de desempenho de inferência (latência, throughput,
tamanho, memória).

Uso:
    Teste já separado (--pre-split true):
        python evaluate_model.py --pre-split true --test_path test.csv \
            --model_path ./joblib/exp-xgb.joblib --label_col malicious --exp exp-xgb

    CSV único, mesmo split do treino (--pre-split false):
        python evaluate_model.py --pre-split false --csv_path dataset.csv --random_state 42 \
            --model_path ./joblib/exp-xgb.joblib --label_col malicious --exp exp-xgb

    No modo --pre-split false, o script refaz o split 70/15/15 (train_test_split
    com stratify, MESMA lógica do train_model.py) e usa só a fatia de teste (15%).
    Para isso funcionar, --csv_path e --random_state precisam ser IDÊNTICOS aos
    usados no treino, senão as amostras de teste vazam pro treino (ou vice-versa)
    e as métricas ficam infladas/erradas.

Se o modelo foi treinado com --lex, passe --lex aqui também (a seleção de
features precisa ser IDÊNTICA à do treino, senão o predict quebra ou fica
silenciosamente errado).

ATENÇÃO: modelos lr/svm/knn/svd são treinados com scaler, mas hoje o
train_model.py só salva o classificador (não o scaler). Rodar este script
pra esses modelos vai dar previsão errada sem nenhum erro visível, porque
os dados de teste não são reescalados. Só é seguro hoje para xgb/rf.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from statistics import mean, median, pstdev

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    import resource

    def peak_memory_mb():
        # Linux: ru_maxrss vem em KB. (No macOS viria em bytes, não tratado aqui.)
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
except ImportError:  # Windows não tem o módulo `resource`
    def peak_memory_mb():
        return None


# Modelos treinados com scaler no train_model.py (ver SCALERS lá) -> sem o
# scaler salvo, a inferência aqui fica incorreta pra eles.
MODELS_THAT_NEED_SCALING = {"lr", "svm", "knn", "svd"}


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Valor booleano esperado (true/false).")


# ---------------------------------------------------------------------------
# Features: precisa ser EXATAMENTE a mesma lógica do train_model.py
# ---------------------------------------------------------------------------
def get_numeric_features(df: pd.DataFrame, label_col: str, lex: bool) -> pd.DataFrame:
    no_label = df.drop(columns=[label_col])
    numeric_cols = no_label.select_dtypes(include=["number"]).columns.tolist()
    if not lex:
        numeric_cols = [c for c in numeric_cols if "lex" not in c or "entropy" in c]
    return no_label[numeric_cols]


def align_features(model, samples: pd.DataFrame) -> pd.DataFrame:
    """Se o modelo foi ajustado (fit) direto num DataFrame, ele guarda os nomes
    das colunas em feature_names_in_. Usamos isso pra validar/reordenar as
    features de teste. Modelos escalonados (fit em array puro) não têm esse
    atributo, então não dá pra validar automaticamente."""
    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        return samples

    expected = list(expected)
    got = list(samples.columns)

    if set(expected) != set(got):
        missing = sorted(set(expected) - set(got))
        extra = sorted(set(got) - set(expected))
        raise ValueError(
            "Features do CSV de teste não batem com as do treino. "
            f"Faltando: {missing}. A mais: {extra}. "
            "Confira se --lex está igual ao usado no treino."
        )

    if expected != got:
        samples = samples[expected]

    return samples


# ---------------------------------------------------------------------------
# Carregamento / split
# ---------------------------------------------------------------------------
def load_test_dataframe(args) -> pd.DataFrame:
    """Retorna o DataFrame de teste, respeitando --pre-split.

    --pre-split true : lê --test_path diretamente (já é o conjunto de teste).
    --pre-split false: lê --csv_path e refaz o split 70/15/15 com o mesmo
                        random_state do treino, devolvendo só a fatia de teste.
    """
    if args.pre_split:
        if not os.path.isfile(args.test_path):
            raise FileNotFoundError(f"CSV de teste não encontrado: {args.test_path}")
        df_test = pd.read_csv(args.test_path, index_col=False)
        return df_test

    if not os.path.isfile(args.csv_path):
        raise FileNotFoundError(f"CSV não encontrado: {args.csv_path}")

    df = pd.read_csv(args.csv_path, index_col=False)
    if args.label_col not in df.columns:
        raise ValueError(f"Coluna de label '{args.label_col}' não existe em {args.csv_path}")

    labels_full = df[args.label_col].values

    # Mesma lógica (e mesma ordem de chamadas) do train_model.py: primeiro
    # separa 70% treino / 30% resto, depois divide o resto 50/50 em val/test.
    # Só assim o conjunto de teste sai idêntico ao usado (ou reservado) no treino.
    df_train, df_rest, _, y_rest = train_test_split(
        df, labels_full, test_size=0.30, stratify=labels_full, random_state=args.random_state
    )
    df_val, df_test, _, _ = train_test_split(
        df_rest, y_rest, test_size=0.50, stratify=y_rest, random_state=args.random_state
    )
    return df_test


def load_samples(args, label_col: str, lex: bool) -> tuple:
    df_test = load_test_dataframe(args)

    if label_col not in df_test.columns:
        raise ValueError(f"Coluna de label '{label_col}' não existe no conjunto de teste")

    samples = get_numeric_features(df_test, label_col, lex)
    labels = df_test[label_col].values
    return samples, labels


def load_model(model_path: str) -> object:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    return joblib.load(model_path)


def _percentile(sorted_data, p):
    idx = min(len(sorted_data) - 1, int(round(p / 100 * (len(sorted_data) - 1))))
    return sorted_data[idx]


def benchmark_batch(model, samples: pd.DataFrame, n_runs: int, warmup: int):
    """Mede o predict() no lote inteiro, repetido n_runs vezes (com warm-up)."""
    n_samples = len(samples)

    for _ in range(warmup):
        model.predict(samples)

    batch_times = []
    preds = None
    for _ in range(n_runs):
        ini = time.perf_counter()
        preds = model.predict(samples)
        end = time.perf_counter()
        batch_times.append(end - ini)

    per_sample_s = [t / n_samples for t in batch_times]

    metrics = {
        "batch_n_samples": n_samples,
        "batch_n_runs": n_runs,
        "batch_time_mean_s": round(mean(batch_times), 4),
        "batch_time_std_s": round(pstdev(batch_times), 4) if n_runs > 1 else 0.0,
        "batch_per_sample_mean_ms": round(mean(per_sample_s) * 1000, 5),
        "throughput_samples_per_s": round(n_samples / mean(batch_times), 1),
    }
    return metrics, preds


def benchmark_single_call(model, samples: pd.DataFrame, n_calls: int):
    """Mede predict() amostra a amostra (cenário de serving em tempo real,
    ex.: classificar um domínio por vez numa requisição de DNS)."""
    n_calls = min(n_calls, len(samples))
    subset = samples.iloc[:n_calls]

    model.predict(subset.iloc[[0]])  # warm-up

    times = []
    for i in range(n_calls):
        row = subset.iloc[[i]]
        ini = time.perf_counter()
        model.predict(row)
        end = time.perf_counter()
        times.append(end - ini)

    times_sorted = sorted(times)
    return {
        "single_call_n": n_calls,
        "single_call_mean_ms": round(mean(times) * 1000, 4),
        "single_call_median_ms": round(median(times) * 1000, 4),
        "single_call_p95_ms": round(_percentile(times_sorted, 95) * 1000, 4),
        "single_call_p99_ms": round(_percentile(times_sorted, 99) * 1000, 4),
        "single_call_max_ms": round(max(times) * 1000, 4),
    }


def model_size_mb(model_path: str) -> float:
    return round(os.path.getsize(model_path) / (1024 * 1024), 3)


def classification_metrics(labels, preds, model, samples) -> tuple:
    report = classification_report(labels, preds, output_dict=True)
    metrics = {
        "accuracy": report["accuracy"],
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
    }

    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(samples)[:, 1]
            metrics["auc_roc"] = roc_auc_score(labels, probs)
        except Exception:
            metrics["auc_roc"] = None
    else:
        metrics["auc_roc"] = None

    cm = confusion_matrix(labels, preds).tolist()
    return metrics, cm, report


def log_metrics_csv(row: dict, log_path: str = "../logs/phisuiil_inference_results.csv") -> str:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    df_row = pd.DataFrame([row])

    if not os.path.isfile(log_path):
        df_row.to_csv(log_path, index=False)
    else:
        df_row.to_csv(log_path, mode="a", header=False, index=False)

    return log_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-split", dest="pre_split", type=str2bool, default=True,
                         help="true: usa --test_path (CSV de teste já separado). "
                              "false: usa --csv_path e refaz o split 70/15/15 (mesmo random_state "
                              "do train_model.py) pra extrair a mesma fatia de teste.")
    parser.add_argument("--test_path", type=str, default=None,
                         help="CSV de teste (requerido se --pre-split true)")
    parser.add_argument("--csv_path", type=str, default=None,
                         help="CSV único, igual ao usado no treino (requerido se --pre-split false)")
    parser.add_argument("--random_state", type=int, default=42,
                         help="Seed do split. Precisa ser IGUAL à usada no train_model.py pra "
                              "reproduzir o mesmo conjunto de teste (só importa se --pre-split false)")
    parser.add_argument("--model_path", type=str, required=True, help="Caminho para o .joblib do modelo")
    parser.add_argument("--label_col", type=str, required=True, help="Nome da coluna de label")
    parser.add_argument("--lex", action="store_true", help="Use IGUAL ao valor usado no treino desse modelo")
    parser.add_argument("--exp", type=str, default=None, help="Nome pra identificar a linha no CSV de log")
    parser.add_argument("--n_runs", type=int, default=5, help="Repetições do benchmark em lote")
    parser.add_argument("--warmup", type=int, default=1, help="Rodadas de warm-up antes de medir (não contam)")
    parser.add_argument("--single_call_n", type=int, default=300,
                         help="Nº de amostras usadas no benchmark de latência amostra-a-amostra")
    parser.add_argument("--output_json", type=str, default=None, help="Salva todas as métricas nesse .json")
    parser.add_argument("--log_csv", type=str, default="../logs/inference_results.csv",
                         help="CSV onde as métricas são anexadas")
    args = parser.parse_args()

    if args.pre_split and not args.test_path:
        parser.error("--pre-split true requer --test_path")
    if not args.pre_split and not args.csv_path:
        parser.error("--pre-split false requer --csv_path")

    return args


def main():
    args = get_args()
    exp_name = args.exp or os.path.splitext(os.path.basename(args.model_path))[0]

    print(f"Carregando modelo de {args.model_path}...")
    model = load_model(args.model_path)

    model_hint = exp_name.split("-")[-1] if "-" in exp_name else ""
    if model_hint in MODELS_THAT_NEED_SCALING:
        print(
            f"[AVISO] '{model_hint}' costuma precisar de scaler no treino, e ele não é salvo hoje. "
            "As previsões abaixo podem estar erradas se este modelo foi treinado com scaling."
        )

    if args.pre_split:
        print(f"Carregando amostras de teste (pre-split) de {args.test_path}...")
    else:
        print(
            f"Carregando '{args.csv_path}' e refazendo o split 70/15/15 "
            f"(seed={args.random_state}) para extrair o conjunto de teste..."
        )
    samples, labels = load_samples(args, args.label_col, args.lex)
    samples = align_features(model, samples)
    print(f"{len(samples)} amostras, {samples.shape[1]} features.")

    print("Rodando benchmark em lote...")
    perf_metrics, preds = benchmark_batch(model, samples, args.n_runs, args.warmup)

    print("Rodando benchmark amostra-a-amostra...")
    perf_metrics.update(benchmark_single_call(model, samples, args.single_call_n))

    perf_metrics["model_size_mb"] = model_size_mb(args.model_path)
    mem = peak_memory_mb()
    perf_metrics["process_peak_memory_mb"] = round(mem, 2) if mem is not None else None

    cls_metrics, confusion, report = classification_metrics(labels, preds, model, samples)

    print("\n=== Métricas de classificação ===")
    print(classification_report(labels, preds))
    if cls_metrics.get("auc_roc") is not None:
        print(f"AUC-ROC: {cls_metrics['auc_roc']:.4f}")
    print(f"Matriz de confusão: {confusion}")

    print("\n=== Métricas de desempenho (inferência) ===")
    for k, v in perf_metrics.items():
        print(f"{k}: {v}")

    row = {"exp": exp_name, "timestamp": datetime.now().isoformat(timespec="seconds")}
    row.update(cls_metrics)
    row.update(perf_metrics)
    log_path = log_metrics_csv(row, args.log_csv)
    print(f"\nMétricas anexadas em: {log_path}")

    if args.output_json:
        payload = {
            "exp": exp_name,
            "classification": {**cls_metrics, "confusion_matrix": confusion},
            "performance": perf_metrics,
        }
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Métricas completas salvas em: {args.output_json}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERRO] {e}", file=sys.stderr)
        sys.exit(1)