"""
Treino de classificadores para detecção de domínios maliciosos.

Uso:
    Dados já divididos em train/val/test:
        python train_model.py --pre-split true --train_path train.csv --val_path val.csv \
            --test_path test.csv --model xgb --exp meu_exp --label_col malicious

    CSV único (o script faz o split 70/15/15):
        python train_model.py --pre-split false --csv_path dataset.csv --random_state 42 \
            --model xgb --exp meu_exp --label_col malicious
"""

import argparse
import logging
import os

import joblib
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBClassifier


def get_numeric_features(df: pd.DataFrame, label_col: str, lex: bool, logger) -> pd.DataFrame:
    no_label = df.drop(columns=[label_col])
    numeric_cols = no_label.select_dtypes(include=["number"]).columns.tolist()

    if not lex:
        numeric_cols = [col for col in numeric_cols if 'lex' not in col or 'entropy' in col]
    logger.info(f"Features usadas: {numeric_cols}")
    return no_label[numeric_cols]


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Valor booleano esperado (true/false).")


def setup_logger(exp_name: str, log_dir: str = "../logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not logger.handlers:
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"{exp_name}.log"), encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def evaluate(clf, X, y, split_name, exp_name, model_name, logger):
    y_prob = clf.predict_proba(X)[:, 1]
    y_pred = clf.predict(X)

    report = classification_report(y, y_pred, output_dict=True)
    auc_score = roc_auc_score(y, y_prob)

    logger.info(
        f"[{split_name}] acc={report['accuracy']:.4f} "
        f"precision={report['macro avg']['precision']:.4f} "
        f"recall={report['macro avg']['recall']:.4f} "
        f"f1={report['macro avg']['f1-score']:.4f} "
        f"auc={auc_score:.4f}"
    )

    return {
        "exp": exp_name,
        "model": model_name,
        "split": split_name,
        "accuracy": report["accuracy"],
        "avg_precision": report["macro avg"]["precision"],
        "avg_recall": report["macro avg"]["recall"],
        "avg_f1-score": report["macro avg"]["f1-score"],
        "auc_roc": auc_score,
    }


def log_metrics_csv(rows, log_path="../logs/phiusiil_results.csv"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    df_new_log = pd.DataFrame(rows)

    if not os.path.isfile(log_path):
        df_new_log.to_csv(log_path, index=False)
    else:
        df_new_log.to_csv(log_path, mode="a", header=False, index=False)

    return log_path


parser = argparse.ArgumentParser()
parser.add_argument("--pre-split", dest="pre_split", type=str2bool, default=False,
                    help="true: usa --train_path/--val_path/--test_path. "
                         "false: usa --csv_path com split automático 70/15/15.")
parser.add_argument("--train_path", type=str, help="CSV de treino (requerido se --pre-split true)")
parser.add_argument("--val_path", type=str, help="CSV de validação (requerido se --pre-split true)")
parser.add_argument("--test_path", type=str, help="CSV de teste (requerido se --pre-split true)")
parser.add_argument("--csv_path", type=str, help="CSV único (requerido se --pre-split false)")
parser.add_argument("--exp", type=str, default="exp", help="Nome do experimento")
parser.add_argument("--random_state", type=int, default=42, help="Seed do split e dos modelos")
parser.add_argument("--featureselect", action="store_true", help="Feature selection")
parser.add_argument("--label_col", type=str, help="Label column")
parser.add_argument("--lex", action="store_true")
parser.add_argument("--model", type=str, default="xgb",
                    choices=["xgb", "rf", "lr", "svm", "knn", "svd"],
                    help="Modelo: xgb, rf, lr, svm, knn ou svd")
args = parser.parse_args()

if args.pre_split:
    if not (args.train_path and args.val_path and args.test_path):
        parser.error("--pre-split true requer --train_path, --val_path e --test_path")
else:
    if not args.csv_path:
        parser.error("--pre-split false requer --csv_path")

logger = setup_logger(args.exp)

label_column = args.label_col

if args.pre_split:
    logger.info("Carregando dados já divididos em train/val/test...")
    df_train = pd.read_csv(args.train_path, index_col=False)
    df_val = pd.read_csv(args.val_path, index_col=False)
    df_test = pd.read_csv(args.test_path, index_col=False)
else:
    logger.info(f"Carregando '{args.csv_path}' e dividindo em 70/15/15 (seed={args.random_state})...")
    df = pd.read_csv(args.csv_path, index_col=False)
    labels = df[label_column].values

    df_train, df_rest, y_train, y_rest = train_test_split(
        df, labels, test_size=0.30, stratify=labels, random_state=args.random_state
    )
    df_val, df_test, y_val, y_test = train_test_split(
        df_rest, y_rest, test_size=0.50, stratify=y_rest, random_state=args.random_state
    )

y_train = df_train[label_column].values
y_val = df_val[label_column].values
y_test = df_test[label_column].values

logger.info(f"Tamanhos -> train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")

logger.info("Extraindo atributos numéricos (colunas de texto descartadas)...")
X_train_final = get_numeric_features(df_train, label_column, args.lex, logger)
X_val_final = get_numeric_features(df_val, label_column, args.lex, logger)
X_test_final = get_numeric_features(df_test, label_column, args.lex, logger)
X_train_num, X_test_num = X_train_final, X_test_final  # cópia "crua" p/ feature selection
n_features = X_train_final.shape[1]
logger.info(f"Número de features: {n_features}")
os.makedirs("./joblib/", exist_ok=True)

models = {
    "xgb": XGBClassifier(
        subsample=0.7, reg_lambda=1, reg_alpha=0.01, n_estimators=1000, min_child_weight=3,
        max_depth=10, learning_rate=0.01, gamma=0.5, colsample_bytree=0.8,
        tree_method="hist", eval_metric="auc", random_state=args.random_state, n_jobs=-1
    ),
    "rf": RandomForestClassifier(
        n_estimators=500, max_depth=15, min_samples_split=5,
        class_weight="balanced", random_state=args.random_state, n_jobs=-1
    ),
    "lr": SGDClassifier(
        loss="log_loss", penalty="l2", max_iter=2000,
        class_weight="balanced", random_state=args.random_state, n_jobs=-1
    ),
    "svm": SGDClassifier(
        loss="modified_huber", penalty="l2", max_iter=2000,
        class_weight="balanced", random_state=args.random_state, n_jobs=-1
    ),
    "knn": KNeighborsClassifier(
        n_neighbors=5, weights="distance", algorithm="auto", leaf_size=100, n_jobs=-1
    ),
    "svd": Pipeline([
        ("svd", TruncatedSVD(n_components=n_features, random_state=args.random_state)),
        ("clf", SGDClassifier(
            loss="modified_huber", class_weight="balanced",
            random_state=args.random_state, n_jobs=-1
        ))
    ]),
}

SCALERS = {
    "xgb": None,
    "rf": None,
    "knn": MinMaxScaler,
    "lr": StandardScaler,
    "svm": StandardScaler,
    "svd": StandardScaler,
}

selected_classifier = models[args.model]
scaler_cls = SCALERS[args.model]

if scaler_cls is not None:
    logger.info(f"Aplicando {scaler_cls.__name__} para {args.model.upper()}...")
    scaler = scaler_cls()
    X_train_final = scaler.fit_transform(X_train_final)
    X_val_final = scaler.transform(X_val_final)
    X_test_final = scaler.transform(X_test_final)
else:
    logger.info(f"Sem normalização para {args.model.upper()} (modelo baseado em árvores).")


logger.info(f"Treinando o modelo: {args.model.upper()} com todos os atributos numéricos...")
selected_classifier.fit(X_train_final, y_train)

val_metrics = evaluate(selected_classifier, X_val_final, y_val, "val", args.exp, args.model, logger)
test_metrics = evaluate(selected_classifier, X_test_final, y_test, "test", args.exp, args.model, logger)

log_path = log_metrics_csv([val_metrics, test_metrics])
logger.info(f"Métricas salvas em: {log_path}")

if args.featureselect:
    logger.info("Rodando feature selection (SelectKBest / mutual_info_classif)...")
    for k in [5, 10, 15, 20]:
        selector = SelectKBest(mutual_info_classif, k=k)
        X_train_sel = selector.fit_transform(X_train_num, y_train)
        X_test_sel = selector.transform(X_test_num)

        selected_classifier.fit(X_train_sel, y_train)
        y_pred = selected_classifier.predict(X_test_sel)
        y_proba = selected_classifier.predict_proba(X_test_sel)[:, 1]

        acc = round(accuracy_score(y_test, y_pred), 4)
        auc = round(roc_auc_score(y_test, y_proba), 4)
        f1 = round(f1_score(y_test, y_pred, average="macro"), 4)

        logger.info(f"  k={k} model={args.model}: acc={acc} auc={auc} f1={f1}")

    selected_classifier.fit(X_train_final, y_train)


model_path = f"./joblib/{args.exp}-{args.model}-phiusiil.joblib"
joblib.dump(selected_classifier, model_path)
logger.info(f"Modelo salvo em {model_path}")