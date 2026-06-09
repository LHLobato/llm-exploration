import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from huggingface_hub import login

login(token="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")

import argparse

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="BERT-Base",
    choices=[
        "BERT-Base",
        "distilBERT",
        "DEBERTa",
        "Llama-3",
        "TinyLlama",
        "Qwen",
        "Gemma-2B",
        "ModernBERT"
    ],
)
parser.add_argument(
    "--modelpath", type=str, help="Caminho para o modelo no HuggingFace ou local"
)
parser.add_argument(
    "--dataset",
    type=str,
    default="domain",
    choices=["domain", "domain-enriched", "csic", "fwaf", "httpparams", "lim", "phiusiil"],
)
parser.add_argument("--dora", action="store_true", help="Usar a técnica DoRA")
parser.add_argument("--parameters", type=str, help="Número de parâmetros")
parser.add_argument("--strategy", type=str, help="Strategy adopted")
parser.add_argument("--num_epochs", type=int, help="Número de épocas")
parser.add_argument("--check", action="store_true", help="Resume from checkpoint")
parser.add_argument("--optuna", action="store_true")
parser.add_argument("--sample", action="store_true")
args = parser.parse_args()


CHAT_TEMPLATE_MODELS = ["Llama-3", "Qwen", "Gemma-2B", "TinyLlama"]


LLM_MODELS = ["Llama-3", "TinyLlama", "Qwen", "Gemma-2B"]

# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
accuracy = evaluate.load("accuracy")
auc_score = evaluate.load("roc_auc")
f1 = evaluate.load("f1")

def remove_www_prefix(domains):
    """
    Recebe um vetor de domínios e remove o prefixo 'www.' do início de cada um.
    Retorna um array do NumPy processado.
    """
    processed_domains = [str(domain).removeprefix("https://") for domain in domains]
    processed_domains = [str(domain).removeprefix("www.") for domain in processed_domains]
    return np.array(processed_domains)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = softmax(logits, axis=-1)
    positive_class_probs = probs[:, 1]

    auc = np.round(
        auc_score.compute(prediction_scores=positive_class_probs, references=labels)[
            "roc_auc"
        ],
        4,
    )
    predicted_classes = np.argmax(logits, axis=1)
    acc = np.round(
        accuracy.compute(predictions=predicted_classes, references=labels)["accuracy"],
        4,
    )
    f1_sc = np.round(
        f1.compute(predictions=predicted_classes, references=labels, average="macro")[
            "f1"
        ],
        4,
    )

    return {"Accuracy": acc, "AUC": auc, "F1-Score": f1_sc}


if args.dataset == "lim":

    col_to_get = "prompt" if args.strategy == "tokenized-prompt" else "0"
    df = pd.read_csv("../data/less-is-more/BTCP.csv", index_col=False)
    prompts = df[col_to_get].values
    labels = df['label'].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        prompts, labels, test_size=0.25, random_state=0, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=0, stratify=y_temp
    )

elif args.dataset == "phiusiil":
    if args.strategy == "tokenized-prompt":
        df = pd.read_csv("../data/PhiUSIIL/phiusiil-filtered.csv", index_col=False)
        names = df['prompt'].values

    else:
        df = pd.read_csv("../data/PhiUSIIL/phiusiil-filtered.csv", index_col=False)
        names = df['Domain'].values
    labels = df['label'].values

    """
    N_SAMPLES = 50000

        if N_SAMPLES and N_SAMPLES < len(names):
            print(f"Modo de Teste: Extraindo uma amostra estratificada de {N_SAMPLES} domínios...")
            _, names, _, labels = train_test_split(
                names, labels,
                test_size=N_SAMPLES,
                random_state=0,
                stratify=labels
        )
    """
    names = remove_www_prefix(names)

    X_train, X_temp, y_train, y_temp = train_test_split(
        names, labels, test_size=0.30, random_state=0, stratify=labels
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=0, stratify=y_temp
    )

if args.dataset == "domain":
    df = pd.read_csv("../dns-feature-enrichment/csvs/dataset.csv")
    labels = df["malicious"].values
    prompts = df["name"].values
    df = None

    X_train, X_temp, y_train, y_temp = train_test_split(
        prompts, labels, test_size=0.30, random_state=0, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=0, stratify=y_temp
    )

elif args.dataset == "domain-enriched":

    strategy = "prompt" if args.strategy == "tokenized" else "name"

    df_train = pd.read_csv(f"../data/acme/fixed_train.csv", index_col=False)
    df_val = pd.read_csv(f"../data/acme/fixed_val.csv", index_col=False)
    df_test = pd.read_csv(f"../data/acme/fixed_test.csv", index_col=False)
    X_train = df_train[strategy].values
    X_val = df_val[strategy].values
    X_test = df_test[strategy].values
    y_train = df_train["malicious"].values
    y_val = df_val["malicious"].values
    y_test = df_test["malicious"].values

    if args.sample:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train,
            train_size=150000,
            random_state=0,
            stratify=y_train
        )

        X_val, _, y_val, _ = train_test_split(
            X_val, y_val,
            train_size=35000,
            random_state=0,
            stratify=y_val
        )

        X_test, _, y_test, _ = train_test_split(
            X_test, y_test,
            train_size=35000,
            random_state=0,
            stratify=y_test
        )



print("\n--- Tamanho dos Conjuntos ---")
print(f"Treino:    {len(X_train)} amostras")
print(f"Validação: {len(X_val)} amostras")
print(f"Teste:     {len(X_test)} amostras")
print("---------------------------\n")

train_dataset = Dataset.from_dict({"prompt": X_train, "label": y_train})
val_dataset = Dataset.from_dict({"prompt": X_val, "label": y_val})
test_dataset = Dataset.from_dict({"prompt": X_test, "label": y_test})

raw_datasets = DatasetDict(
    {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
)

# ---------------------------------------------------------------------------
# Tokenizador
# ---------------------------------------------------------------------------
model_path = args.modelpath
tokenizer = AutoTokenizer.from_pretrained(model_path)

if args.model in LLM_MODELS:
    optim = "adamw_8bit"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" if args.model == "Gemma-2B" else "left"
elif args.model in["DEBERTa", "ModernBERT"] :
    print("debertaa")
    optim = "adamw_8bit"
else:
    optim = "adamw_torch"

SYSTEM_INSTRUCTION = (
    "You are a cybersecurity classifier. "
    "Analyze the domain record below and classify it as benign or malicious."
)


def format_chat_prompt(prompt_text: str) -> str:
    """
    Envolve o prompt enriquecido na estrutura role/content do chat template
    nativo do modelo (ex: Gemma, Llama, Qwen).

    Resultado para o Gemma 2B Instruct:
        <start_of_turn>user
        You are a cybersecurity classifier. Analyze the domain record below
        and classify it as benign or malicious.

        [name]: fr7ehgd.duckdns.org
        [entropy]: 3.6819
        [whois]: present=no
        ...
        <end_of_turn>
        <start_of_turn>model
    """
    messages = [{"role": "user", "content": f"{SYSTEM_INSTRUCTION}\n\n{prompt_text}"}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def preprocess_function(examples):
    if args.model in CHAT_TEMPLATE_MODELS:
        formatted = [format_chat_prompt(p) for p in examples["prompt"]]
    else:
        formatted = [f"{SYSTEM_INSTRUCTION}\n\n{p}" for p in examples["prompt"]]

    return tokenizer(
        formatted,
        truncation=True,
        padding=True,
        max_length=200,
    )

id2label = {0: "Benign", 1: "Malicious"}
label2id = {"Benign": 0, "Malicious": 1}


bnb_config = None
if args.model in ["Qwen", "Gemma-2B", "Llama-3", "TinyLlama"]:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
    quantization_config=bnb_config,
    device_map="auto" if bnb_config else None,
)


if args.model == "Gemma-2B":
    model.config.pad_token_id = tokenizer.pad_token_id

if args.model == "DEBERTa":
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id


if args.dora:
    if bnb_config:
        model = prepare_model_for_kbit_training(model)

    target_modules = {
        "BERT-Base": ["query", "value"],
        "distilBERT": ["q_lin", "v_lin"],
        "DEBERTa": ["query_proj", "value_proj"],
        "Llama-3": [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "TinyLlama": [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "Qwen": [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "Gemma-2B": [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    }[args.model]

    lora_config = LoraConfig(
        use_dora=True,
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        task_type=TaskType.SEQ_CLS,
        bias="none",
        lora_dropout=0.00,
    )
    model = get_peft_model(model, lora_config)


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


if args.model in ["distilBERT", "ModernBERT"]:
    batch_size = 8

elif args.model in ["BERT-Base", "DEBERTa"]:
    batch_size = 8

elif args.model in ["Llama-3", "Qwen"]:
    batch_size = 32

elif args.model == "TinyLlama":
    batch_size = 16

elif args.model == "Gemma-2B":
    batch_size = 4


lr =  2e-05


outputdir = f"../models/{args.model}/{args.dataset}-{args.strategy}-{args.parameters}/"
best = outputdir + "best/"

training_args = TrainingArguments(
    output_dir=outputdir,
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=8,
    num_train_epochs=args.num_epochs,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="AUC",
    bf16=True,
    greater_is_better=True,
    warmup_ratio=0.05,
    weight_decay= 0.05,
    optim=optim,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)


if args.optuna:
    def model_init():
        temp_model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
            quantization_config=bnb_config,
            device_map="auto" if bnb_config else None,
        )

        if args.model == "Gemma-2B":
            temp_model.config.pad_token_id = tokenizer.pad_token_id

        if args.model == "DEBERTa":
            temp_model.config.pad_token_id = tokenizer.pad_token_id
            temp_model.config.bos_token_id = tokenizer.bos_token_id
            temp_model.config.eos_token_id = tokenizer.eos_token_id

        if args.dora:
            if bnb_config:
                temp_model = prepare_model_for_kbit_training(temp_model)
            l_config = LoraConfig(
                use_dora=True,
                r=16,
                lora_alpha=32,
                target_modules=target_modules,
                task_type=TaskType.SEQ_CLS,
                bias="none",
                lora_dropout=0.05,
            )
            temp_model = get_peft_model(temp_model, l_config)

        return temp_model

    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.05, 0.15),
        }


    search_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(min(10000, len(tokenized_datasets["train"]))))
    search_val_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(min(2000, len(tokenized_datasets["validation"]))))

    trainer = Trainer(
        model=None,
        model_init=model_init,
        args=training_args,
        train_dataset=search_train_dataset,
        eval_dataset=search_val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\n--- Iniciando Grid/Bayesian Search com Optuna (10k samples) ---")
    best_run = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=hp_space,
        n_trials=6
    )

    print(f"\nMelhores Hiperparâmetros: {best_run.hyperparameters}")

    for n, v in best_run.hyperparameters.items():
        setattr(trainer.args, n, v)

    trainer.train_dataset = tokenized_datasets["train"]
    trainer.eval_dataset = tokenized_datasets["validation"]

    print(f"\nIniciando treinamento FINAL do {args.model} com parâmetros otimizados...\n")
    trainer.train(resume_from_checkpoint=args.check)
else:
    print(f"\nIniciando treinamento do {args.model}...\n")
    trainer.train(resume_from_checkpoint=args.check)

trainer.save_model(best)
tokenizer.save_pretrained(best)

print("\nAvaliação no Conjunto Final de Teste:")
predictions = trainer.predict(tokenized_datasets["test"])

logits = predictions.predictions
labels = predictions.label_ids
metrics = compute_metrics((logits, labels))

path = "../results/phiusiil-results.csv"
exists = os.path.exists(path)

pd.DataFrame(
    {
        "model": [f"{model_path}-{args.dataset}-{args.parameters}"],
        "strategy": [
           args.strategy
        ],
        "accuracy": [metrics["Accuracy"]],
        "auc": [metrics["AUC"]],
        "f1-score": [metrics["F1-Score"]],
    }
).to_csv(path, index=False, header=not exists, mode="a")

print(metrics)
print("\nTreinamento finalizado e salvo com sucesso!")
