"""
Avalia um checkpoint LLM/encoder (BERT-Base, distilBERT, DEBERTa, ModernBERT,
Llama-3, TinyLlama, Qwen, Gemma, DeepSeek) em métricas de classificação e
métricas de desempenho de inferência (latência, throughput, tamanho, memória
CPU/GPU) -- espelha o que o evaluate_model.py faz para xgb/rf, só que
reaproveitando o pipeline de dados e de formatação de prompt do script de
treino (bert_training.py), pra garantir que o conjunto avaliado e a forma de
montar o prompt sejam EXATAMENTE os mesmos usados no treino/zero-shot.

Uso:
    Modelo fine-tuned (classifier head), avaliado no split de teste (15%):
        python evaluate_llm.py --model Qwen --modelpath ../models/Qwen/phiusiil-tokenized-prompt-7B/best \
            --dataset phiusiil --strategy tokenized-prompt --train_mode tuning --exp qwen-phiusiil

    Zero-shot (sem treino, avalia no pool de 90% após reservar exemplos):
        python evaluate_llm.py --model Gemma --modelpath google/gemma-3-4b-it \
            --dataset phiusiil --strategy tokenized-prompt --train_mode zero --exp gemma-zeroshot

    Few-shot:
        python evaluate_llm.py --model Llama-3 --modelpath meta-llama/Llama-3.2-3B-Instruct \
            --dataset phiusiil --strategy tokenized-prompt --train_mode few --num_shots 2 --exp llama-fewshot

    DoRA/LoRA salvo como adaptador (não mergeado) -- passe a base separada:
        python evaluate_llm.py --model Qwen --modelpath ../models/Qwen/.../best \
            --base_modelpath Qwen/Qwen2.5-7B-Instruct --dora --train_mode tuning ...

ATENÇÃO:
- --dataset/--strategy/--parameters/--sample precisam bater com o que foi
  usado no treino, senão o split de teste (tuning) ou o pool de avaliação
  (zero/few) sai diferente do que o modelo realmente nunca viu.
- Em --train_mode zero/few não há split de teste "oficial": o script
  reproduz a MESMA lógica do treino (pool de 10% para exemplos few-shot +
  90% de avaliação, dedupe, seed=0), então o conjunto avaliado é idêntico
  independente da config de shots.
- --dora + adaptador não mergeado exige --base_modelpath; se o adaptador foi
  treinado sobre um modelo em 4-bit, o merge_and_unload() é pulado (não dá
  pra mergear LoRA num modelo quantizado sem desquantizar) e o benchmark
  roda com o PEFT wrapper por cima -- isso é o cenário real de produção
  nesse caso, então não é um bug.
"""

import argparse
import json
import os
import sys
import time
import urllib
from datetime import datetime
from statistics import mean, median, pstdev

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from evaluate import load as load_metric
from huggingface_hub import login
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

load_dotenv()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
login(token=os.getenv("HF_TOKEN"))

try:
    import resource

    def peak_memory_mb():
        # Linux: ru_maxrss vem em KB. (No macOS viria em bytes, não tratado aqui.)
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
except ImportError:  # Windows não tem o módulo `resource`
    def peak_memory_mb():
        return None


def gpu_peak_memory_mb():
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)


# ---------------------------------------------------------------------------
# Constantes -- copiadas do script de treino pra manter prompt/formatação idênticos
# ---------------------------------------------------------------------------
CHAT_TEMPLATE_MODELS = ["Llama-3", "Qwen", "Gemma", "TinyLlama", "DeepSeek"]
LLM_MODELS = ["Llama-3", "TinyLlama", "Qwen", "Gemma", "DeepSeek"]

SYSTEM_INSTRUCTION = (
    "You are a cybersecurity classifier. "
    "Analyze the domain record below and classify it as benign or malicious."
)

# Mesmo prefill do treino: força o próximo token gerado a ser o rótulo em si.
LABEL_PREFILL = "Answer: "

accuracy = load_metric("accuracy")
auc_score = load_metric("roc_auc")
f1 = load_metric("f1")


# ---------------------------------------------------------------------------
# Dataset helpers -- copiados do script de treino (split fixo seed=0)
# ---------------------------------------------------------------------------
def split_70_15_15(X, y, random_state: int = 0):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def dedupe_arrays(X, y):
    _, idx = np.unique(X, return_index=True)
    idx = np.sort(idx)
    return X[idx], y[idx]


def split_example_eval(X, y, example_frac: float = 0.10, random_state: int = 0):
    X_example, X_eval, y_example, y_eval = train_test_split(
        X, y, test_size=1 - example_frac, random_state=random_state, stratify=y
    )
    return X_example, y_example, X_eval, y_eval


def build_few_shot_pool(X_pool, y_pool, k, seed=0):
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y_pool == 1)[0]
    neg_idx = np.where(y_pool == 0)[0]
    chosen = np.concatenate(
        [rng.choice(pos_idx, size=k, replace=False), rng.choice(neg_idx, size=k, replace=False)]
    )
    rng.shuffle(chosen)
    return [(X_pool[i], int(y_pool[i])) for i in chosen]


def remove_www_prefix(domains):
    processed = [str(d).removeprefix("https://") for d in domains]
    processed = [str(d).removeprefix("www.") for d in processed]
    return np.array(processed)


def load_csic():
    def loadData(file):
        with open(file, "r", encoding="utf8") as f:
            return [d.strip() for d in f.readlines() if len(d.strip()) > 0]

    bad_requests = loadData("../data/http/PreProcessedAnomalous.txt")
    good_requests = loadData("../data/http/PreprocessedNormalTraining.txt")
    all_requests = bad_requests + good_requests
    labels = [1] * len(bad_requests) + [0] * len(good_requests)
    return all_requests, labels


def load_fwaf():
    def loadFile(name):
        num_samples = 0
        filepath = os.path.join(str(os.getcwd()), name)
        with open(filepath, "r") as f:
            data = f.readlines()
        data = list(set(data))
        result = []
        for d in data:
            result.append(str(urllib.parse.unquote(d)))
            num_samples += 1
            if num_samples >= 120000:
                return result
        return result

    badQueries = list(set(loadFile("../data/http/badqueries.txt")))
    validQueries = list(set(loadFile("../data/http/goodqueries.txt")))
    queries = badQueries + validQueries
    y = [1] * len(badQueries) + [0] * len(validQueries)
    return queries, y


def load_httpparams():
    df = pd.read_csv("../data/http/payload_full.csv")
    df.dropna(inplace=True)
    df.loc[df["label"] == "norm", "label"] = 0
    df.loc[df["label"] == "anom", "label"] = 1
    df["label"] = df["label"].astype(int)
    return df["payload"].values, df["label"].values


def load_eval_split(args):
    """Reproduz exatamente a lógica de carregamento/split do script de
    treino e devolve (X_eval, y_eval, few_shot_examples):
    - train_mode tuning: X_eval/y_eval = split de teste (15%), o mesmo
      conjunto nunca visto durante o fine-tuning.
    - train_mode zero/few: X_eval/y_eval = pool de 90% (após dedupe e
      reserva do pool de exemplos), few_shot_examples = k exemplares por
      classe amostrados do pool de 10% (só preenchido se train_mode=='few')."""
    if args.dataset == "domain-enriched":
        strategy = "prompt" if args.strategy == "tokenized" else "name"
        df_train = pd.read_csv("../data/acme/fixed_train.csv", index_col=False)
        df_val = pd.read_csv("../data/acme/fixed_val.csv", index_col=False)
        df_test = pd.read_csv("../data/acme/fixed_test.csv", index_col=False)
        X_train, X_val, X_test = df_train[strategy].values, df_val[strategy].values, df_test[strategy].values
        y_train = df_train["malicious"].values
        y_val = df_val["malicious"].values
        y_test = df_test["malicious"].values

        if args.sample:
            X_train, _, y_train, _ = train_test_split(
                X_train, y_train, train_size=150000, random_state=0, stratify=y_train
            )
            X_val, _, y_val, _ = train_test_split(
                X_val, y_val, train_size=35000, random_state=0, stratify=y_val
            )
            X_test, _, y_test, _ = train_test_split(
                X_test, y_test, train_size=35000, random_state=0, stratify=y_test
            )

    elif args.dataset == "phiusiil":
        df = pd.read_csv("../data/PhiUSIIL/phiusiil-filtered.csv", index_col=False)
        prompts = (
            df["prompt"].values
            if args.strategy == "tokenized-prompt"
            else remove_www_prefix(df["Domain"].values)
        )
        labels = df["label"].values
        if args.sample:
            N_SAMPLES = min(36000, len(prompts))
            _, prompts, _, labels = train_test_split(
                prompts, labels, test_size=N_SAMPLES, random_state=0, stratify=labels
            )
        X_train, X_val, X_test, y_train, y_val, y_test = split_70_15_15(prompts, labels)

    elif "subset" in args.dataset:
        df = pd.read_csv(f"../data/{args.dataset}.csv", index_col=False)
        labels = df["label"].values if "phiusiil" in args.dataset else df["malicious"].values
        if args.strategy == "tokenized-prompt":
            names = df["prompt"].values
        else:
            names = df["Domain"].values if "phiusiil" in args.dataset else df["name"].values
        names = remove_www_prefix(names)
        X_train, X_val, X_test, y_train, y_val, y_test = split_70_15_15(names, labels)

    elif args.dataset == "lim":
        col_to_get = "prompt" if args.strategy == "tokenized-prompt" else "0"
        df = pd.read_csv("../data/less-is-more/BTCP.csv", index_col=False)
        prompts = df[col_to_get].values
        labels = df["label"].values
        X_train, X_val, X_test, y_train, y_val, y_test = split_70_15_15(prompts, labels)

    elif args.dataset == "domain":
        df = pd.read_csv("../dns-feature-enrichment/csvs/dataset.csv")
        labels = df["malicious"].values
        prompts = df["name"].values
        X_train, X_val, X_test, y_train, y_val, y_test = split_70_15_15(prompts, labels)

    elif args.dataset in ("csic", "fwaf", "httpparams"):
        loaders = {"csic": load_csic, "fwaf": load_fwaf, "httpparams": load_httpparams}
        prompts, labels = loaders[args.dataset]()
        X_train, X_val, X_test, y_train, y_val, y_test = split_70_15_15(prompts, labels)

    else:
        raise ValueError(f"Dataset desconhecido: {args.dataset}")

    few_shot_examples = None

    if args.train_mode in ("zero", "few"):
        X_all = np.concatenate([X_train, X_val, X_test])
        y_all = np.concatenate([y_train, y_val, y_test])
        X_all, y_all = dedupe_arrays(X_all, y_all)

        X_example_pool, y_example_pool, X_eval, y_eval = split_example_eval(
            X_all, y_all, example_frac=0.10, random_state=0
        )

        if args.train_mode == "few":
            few_shot_examples = build_few_shot_pool(X_example_pool, y_example_pool, k=args.num_shots)

        return X_eval, y_eval, few_shot_examples

    # train_mode == "tuning": avalia no split de teste (15%), o mesmo
    # conjunto reservado pelo train_model.py e nunca visto no fine-tuning.
    return X_test, y_test, few_shot_examples


# ---------------------------------------------------------------------------
# Formatação de prompt -- copiada do script de treino
# ---------------------------------------------------------------------------
def format_eval_texts(args, tokenizer, X, few_shot_examples, is_causal):
    def format_chat_prompt(prompt_text):
        messages = [{"role": "user", "content": f"{SYSTEM_INSTRUCTION}\n\n{prompt_text}"}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def format_zero_shot(prompt_text):
        return format_chat_prompt(prompt_text) + LABEL_PREFILL

    def format_few_shot(prompt_text, examples):
        messages = []
        for i, (ex_text, ex_label) in enumerate(examples):
            content = f"{SYSTEM_INSTRUCTION}\n\n{ex_text}" if i == 0 else ex_text
            messages.append({"role": "user", "content": content})
            messages.append({"role": "assistant", "content": f"{LABEL_PREFILL}{ex_label}"})
        messages.append({"role": "user", "content": prompt_text})
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return formatted + LABEL_PREFILL

    if is_causal:
        if args.model in CHAT_TEMPLATE_MODELS:
            if args.train_mode == "zero":
                return [format_zero_shot(p) for p in X]
            return [format_few_shot(p, few_shot_examples) for p in X]
        return [f"{SYSTEM_INSTRUCTION}\n\n{p}{LABEL_PREFILL}" for p in X]

    # train_mode == "tuning": classifier head -> sem prefill, sem geração
    if args.model in CHAT_TEMPLATE_MODELS:
        return [format_chat_prompt(p) for p in X]
    return [f"{SYSTEM_INSTRUCTION}\n\n{p}" for p in X]


def get_label_token_ids(tokenizer, prefix_text: str) -> tuple:
    base_len = len(tokenizer(prefix_text, add_special_tokens=False).input_ids)
    id0 = tokenizer(prefix_text + "0", add_special_tokens=False).input_ids[base_len]
    id1 = tokenizer(prefix_text + "1", add_special_tokens=False).input_ids[base_len]
    return id0, id1


# ---------------------------------------------------------------------------
# Carregamento do modelo -- copiado do script de treino
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(args):
    is_causal = args.train_mode in ("zero", "few")

    tokenizer = AutoTokenizer.from_pretrained(args.modelpath)

    if args.model in LLM_MODELS:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right" if args.model == "Gemma" else "left"

    id2label = {0: "Benign", 1: "Malicious"}
    label2id = {"Benign": 0, "Malicious": 1}

    bnb_config = None
    if args.model in ["Qwen", "Gemma", "Llama-3", "TinyLlama", "DeepSeek"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    if is_causal:
        model = AutoModelForCausalLM.from_pretrained(
            args.modelpath,
            quantization_config=bnb_config,
            device_map="auto" if bnb_config else None,
        )
    else:
        load_path = args.base_modelpath if (args.dora and args.base_modelpath) else args.modelpath
        model = AutoModelForSequenceClassification.from_pretrained(
            load_path,
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
            quantization_config=bnb_config,
            device_map="auto" if bnb_config else None,
        )
        if args.dora and args.base_modelpath:
            from peft import PeftModel

            print(f"Anexando adaptador PEFT de {args.modelpath} sobre a base {args.base_modelpath}...")
            model = PeftModel.from_pretrained(model, args.modelpath)
            if not bnb_config:
                model = model.merge_and_unload()

    if args.model in LLM_MODELS:
        model.config.pad_token_id = tokenizer.pad_token_id
    if args.model == "DEBERTa":
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id

    model.eval()
    return model, tokenizer, is_causal


# ---------------------------------------------------------------------------
# Forward pass em lote -- unifica o caso causal (logit restrito 0/1) e o
# caso classifier head (logits do próprio classificador)
# ---------------------------------------------------------------------------
def run_forward(model, tokenizer, texts, batch_size, max_length, is_causal, token_ids=None):
    all_logits = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True, padding=True, max_length=max_length
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        if is_causal:
            token_0, token_1 = token_ids
            attn = inputs["attention_mask"]
            seq_len = attn.size(1)
            # Índice do último token real por sequência -- funciona com left
            # padding (Llama/Qwen/TinyLlama/DeepSeek) e right padding (Gemma).
            last_real_idx = seq_len - 1 - attn.flip(dims=[1]).argmax(dim=1)
            last_token_logits = outputs.logits[
                torch.arange(outputs.logits.size(0)), last_real_idx, :
            ]
            batch_logits = last_token_logits[:, [token_0, token_1]]
        else:
            batch_logits = outputs.logits

        all_logits.append(batch_logits.cpu().float())

    return torch.cat(all_logits, dim=0).numpy()


def _percentile(sorted_data, p):
    idx = min(len(sorted_data) - 1, int(round(p / 100 * (len(sorted_data) - 1))))
    return sorted_data[idx]


def benchmark_batch(model, tokenizer, texts, is_causal, batch_size, n_runs, warmup, max_length, token_ids=None):
    n_samples = len(texts)

    for _ in range(warmup):
        run_forward(model, tokenizer, texts, batch_size, max_length, is_causal, token_ids)

    batch_times = []
    logits = None
    for _ in range(n_runs):
        ini = time.perf_counter()
        logits = run_forward(model, tokenizer, texts, batch_size, max_length, is_causal, token_ids)
        end = time.perf_counter()
        batch_times.append(end - ini)

    per_sample_s = [t / n_samples for t in batch_times]
    metrics = {
        "batch_size": batch_size,
        "batch_n_samples": n_samples,
        "batch_n_runs": n_runs,
        "batch_time_mean_s": round(mean(batch_times), 4),
        "batch_time_std_s": round(pstdev(batch_times), 4) if n_runs > 1 else 0.0,
        "batch_per_sample_mean_ms": round(mean(per_sample_s) * 1000, 5),
        "throughput_samples_per_s": round(n_samples / mean(batch_times), 1),
    }
    return metrics, logits


def benchmark_single_call(model, tokenizer, texts, is_causal, n_calls, max_length, token_ids=None):
    """Mede predict() amostra a amostra (batch_size=1) -- cenário de serving
    em tempo real, ex.: classificar um domínio por vez numa requisição."""
    n_calls = min(n_calls, len(texts))
    subset = texts[:n_calls]

    run_forward(model, tokenizer, subset[:1], 1, max_length, is_causal, token_ids)  # warm-up

    times = []
    for t in subset:
        ini = time.perf_counter()
        run_forward(model, tokenizer, [t], 1, max_length, is_causal, token_ids)
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


def get_model_size_mb(model, model_path) -> float:
    """Tamanho em disco do checkpoint local (comparável ao model_size_mb do
    evaluate_model.py). Se model_path for um id do HF Hub (não um diretório
    local, ex.: zero-shot direto do hub), cai pro footprint em memória dos
    parâmetros/buffers já carregados."""
    if model_path and os.path.isdir(model_path):
        total = 0
        for root, _, files in os.walk(model_path):
            for f in files:
                fp = os.path.join(root, f)
                if os.path.isfile(fp):
                    total += os.path.getsize(fp)
        if total > 0:
            return round(total / (1024 * 1024), 3)

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return round((param_bytes + buffer_bytes) / (1024 * 1024), 3)


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    probs = softmax(logits, axis=-1)
    positive_class_probs = probs[:, 1]

    auc = np.round(
        auc_score.compute(prediction_scores=positive_class_probs, references=labels)["roc_auc"], 4
    )
    predicted_classes = np.argmax(logits, axis=1)
    acc = np.round(
        accuracy.compute(predictions=predicted_classes, references=labels)["accuracy"], 4
    )
    f1_sc = np.round(
        f1.compute(predictions=predicted_classes, references=labels, average="macro")["f1"], 4
    )
    return {"Accuracy": float(acc), "AUC": float(auc), "F1-Score": float(f1_sc)}


def log_metrics_csv(row: dict, log_path: str) -> str:
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model", type=str, default="BERT-Base",
        choices=["BERT-Base", "distilBERT", "DEBERTa", "Llama-3", "TinyLlama", "Qwen", "Gemma", "ModernBERT", "DeepSeek"],
    )
    parser.add_argument("--modelpath", type=str, required=True, help="Checkpoint a avaliar (local ou HF Hub)")
    parser.add_argument(
        "--base_modelpath", type=str, default=None,
        help="Modelo base -- só necessário com --dora se --modelpath for um adaptador PEFT não mergeado",
    )
    parser.add_argument("--num_shots", type=int, default=2, help="Usado só se --train_mode few")
    parser.add_argument(
        "--dataset", type=str, default="domain",
        choices=["domain", "domain-enriched", "csic", "fwaf", "httpparams", "lim", "phiusiil",
                 "15kphiusiil_subset", "30kcustom_subset"],
    )
    parser.add_argument("--train_mode", type=str, required=True, choices=["zero", "few", "tuning"])
    parser.add_argument("--dora", action="store_true", help="Modelo foi ajustado com DoRA/LoRA")
    parser.add_argument("--parameters", type=str, default="", help="Só usado pro nome do experimento")
    parser.add_argument("--strategy", type=str, default="", help="domain-only | tokenized-prompt | ...")
    parser.add_argument("--batch", type=int, default=32, help="Batch size do benchmark em lote")
    parser.add_argument("--sample", action="store_true", help="Usar o mesmo subsample do treino")
    parser.add_argument(
        "--max_eval_samples", type=int, default=None,
        help="Limita o nº de amostras do benchmark em lote (amostragem aleatória, seed=0)",
    )
    parser.add_argument("--exp", type=str, default=None, help="Nome pra identificar a linha no CSV de log")
    parser.add_argument("--n_runs", type=int, default=5, help="Repetições do benchmark em lote")
    parser.add_argument("--warmup", type=int, default=1, help="Rodadas de warm-up antes de medir (não contam)")
    parser.add_argument("--single_call_n", type=int, default=300, help="Nº de amostras no benchmark amostra-a-amostra")
    parser.add_argument("--output_json", type=str, default=None, help="Salva todas as métricas nesse .json")
    parser.add_argument("--log_csv", type=str, default="../logs/llm_inference_results.csv",
                         help="CSV onde as métricas são anexadas")

    args = parser.parse_args()

    if args.dora and args.base_modelpath is None:
        print(
            "[AVISO] --dora sem --base_modelpath: assumindo que --modelpath já é um checkpoint "
            "mergeado/completo (não um adaptador PEFT solto)."
        )

    return args


def main():
    args = get_args()
    exp_name = args.exp or f"{args.model}-{args.dataset}-{args.strategy}-{args.parameters}-{args.train_mode}"

    print(f"[1/4] Carregando dataset '{args.dataset}' (train_mode={args.train_mode})...")
    eval_texts_raw, eval_labels, few_shot_examples = load_eval_split(args)
    print(f"Amostras de avaliação: {len(eval_texts_raw)}")

    if args.max_eval_samples and len(eval_texts_raw) > args.max_eval_samples:
        rng = np.random.RandomState(0)
        idx = np.sort(rng.choice(len(eval_texts_raw), size=args.max_eval_samples, replace=False))
        eval_texts_raw = eval_texts_raw[idx]
        eval_labels = eval_labels[idx]
        print(f"Limitado a --max_eval_samples={args.max_eval_samples} (amostragem aleatória, seed=0).")

    print(f"[2/4] Carregando modelo/tokenizer de {args.modelpath}...")
    model, tokenizer, is_causal = load_model_and_tokenizer(args)

    print("[3/4] Formatando textos de entrada...")
    formatted_texts = format_eval_texts(args, tokenizer, eval_texts_raw, few_shot_examples, is_causal)

    if is_causal:
        max_length = max(len(tokenizer(t, add_special_tokens=False).input_ids) for t in formatted_texts) + 10
        token_ids = get_label_token_ids(tokenizer, formatted_texts[0])
        print(f"max_length (dinâmico, zero/few): {max_length}")
    else:
        max_length = 200  # mesmo valor fixo usado no preprocess_function do treino
        token_ids = None

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("[4/4] Rodando benchmarks de inferência...")
    print("Benchmark em lote...")
    batch_metrics, logits = benchmark_batch(
        model, tokenizer, formatted_texts, is_causal,
        batch_size=args.batch, n_runs=args.n_runs, warmup=args.warmup,
        max_length=max_length, token_ids=token_ids,
    )

    print("Benchmark amostra-a-amostra...")
    single_metrics = benchmark_single_call(
        model, tokenizer, formatted_texts, is_causal,
        n_calls=args.single_call_n, max_length=max_length, token_ids=token_ids,
    )

    perf_metrics = {**batch_metrics, **single_metrics}
    perf_metrics["model_size_mb"] = get_model_size_mb(model, args.modelpath)
    mem = peak_memory_mb()
    perf_metrics["process_peak_memory_mb"] = round(mem, 2) if mem is not None else None
    perf_metrics["gpu_peak_memory_mb"] = gpu_peak_memory_mb()

    cls_metrics = compute_metrics((logits, eval_labels))

    print("\n=== Métricas de classificação ===")
    print(cls_metrics)

    print("\n=== Métricas de desempenho (inferência) ===")
    for k, v in perf_metrics.items():
        print(f"{k}: {v}")

    row = {
        "exp": exp_name,
        "model": args.model,
        "dataset": args.dataset,
        "strategy": args.strategy,
        "train_mode": args.train_mode,
        "num_shots": args.num_shots if args.train_mode == "few" else None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "accuracy": cls_metrics["Accuracy"],
        "auc": cls_metrics["AUC"],
        "f1_macro": cls_metrics["F1-Score"],
    }
    row.update(perf_metrics)
    log_path = log_metrics_csv(row, args.log_csv)
    print(f"\nMétricas anexadas em: {log_path}")

    if args.output_json:
        payload = {"exp": exp_name, "classification": cls_metrics, "performance": perf_metrics}
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Métricas completas salvas em: {args.output_json}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERRO] {e}", file=sys.stderr)
        sys.exit(1)