import os
UNSLOTH_AVAILABLE = False
try:
    from unsloth import FastLanguageModel
    from unsloth.trainer import UnslothTrainer, UnslothTrainingArguments
    UNSLOTH_AVAILABLE = True
    print("Unsloth disponível!")
except ImportError:
    print("Unsloth não disponível. Usando transformers padrão.")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from huggingface_hub import login

login(token="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")

import argparse

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_from_disk
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

# Importação segura do Unsloth - sem quebrar se não estiver disponível


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="Qwen",
    choices=[
        "BERT-Base",
        "distilBERT",
        "DEBERTa",
        "Llama-3",
        "TinyLlama",
        "Qwen",
        "Gemma-2B",
    ],
)
parser.add_argument(
    "--modelpath", type=str, help="Caminho para o modelo no HuggingFace ou local"
)
parser.add_argument(
    "--dataset",
    type=str,
    default="domain-enriched",
    choices=["domain", "domain-enriched", "csic", "fwaf", "httpparams", "lim", "phiusiil"],
)
parser.add_argument("--dora", action="store_true", help="Usar a técnica DoRA")
parser.add_argument("--parameters", type=str, help="Número de parâmetros")
parser.add_argument("--strategy", type=str, help="Strategy adopted")
parser.add_argument("--num_epochs", type=int, help="Número de épocas")
parser.add_argument("--check", action="store_true", help="Resume from checkpoint")
parser.add_argument("--optuna", action="store_true")
parser.add_argument("--use_unsloth", action="store_true", help="Usar Unsloth")
parser.add_argument("--no_subsampling", action="store_true", help="Não fazer subsampling (usa todos os dados)")
parser.add_argument("--token_cache", action="store_true", help="Usar cache de tokenização")
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


# ============================================================================
# CARREGAR DATASET
# ============================================================================
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
        df = pd.read_csv("../data/PhiUSIIL/PhiUSIIL.csv", index_col=False)
        names = df['Domain'].values
    labels = df['label'].values

    names = remove_www_prefix(names)

    X_train, X_temp, y_train, y_temp = train_test_split(
        names, labels, test_size=0.30, random_state=0, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=0, stratify=y_temp
    )

elif args.dataset == "domain":
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

    df_train = pd.read_csv("../data/acme/fixed_train.csv", index_col=False)
    df_val = pd.read_csv("../data/acme/fixed_val.csv", index_col=False)
    df_test = pd.read_csv("../data/acme/fixed_test.csv", index_col=False)

    X_train = df_train[strategy].values
    X_val = df_val[strategy].values
    X_test = df_test[strategy].values

    y_train = df_train["malicious"].values
    y_val = df_val["malicious"].values
    y_test = df_test["malicious"].values

    # === SUBAMOSTRAGEM PARA 150K (se não flag --no_subsampling) ===
    if not args.no_subsampling and len(X_train) > 150000:
        print(f"\n📊 Subamostragem: {len(X_train)} → 150,000 amostras (estratificada)")
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

        print(f"✅ Train reduzido para {len(X_train)} amostras\n")

print("\n--- Tamanho dos Conjuntos ---")
print(f"Treino:    {len(X_train)} amostras")
print(f"Validação: {len(X_val)} amostras")
print(f"Teste:     {len(X_test)} amostras")
print("---------------------------\n")


# ============================================================================
# TOKENIZER E MODELO
# ============================================================================
model_path = args.modelpath
tokenizer = AutoTokenizer.from_pretrained(model_path)

if args.model in LLM_MODELS:
    optim = "adamw_8bit"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" if args.model == "Gemma-2B" else "left"
elif args.model == "DEBERTa":
    optim = "adamw_8bit"
else:
    optim = "adamw_torch"

SYSTEM_INSTRUCTION = (
    "You are a cybersecurity classifier. "
    "Analyze the domain record below and classify it as benign or malicious."
)

def format_chat_prompt(prompt_text: str) -> str:
    messages = [{"role": "user", "content": f"{SYSTEM_INSTRUCTION}\n\n{prompt_text}"}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

# max_length otimizado para prompts do acme (~170-180 tokens em média)
MAX_LENGTH = 192 if args.model in LLM_MODELS else 128

def preprocess_function(examples):
    if args.model in CHAT_TEMPLATE_MODELS:
        formatted = [format_chat_prompt(p) for p in examples["prompt"]]
        return tokenizer(formatted, truncation=True, max_length=MAX_LENGTH)
    else:
        return tokenizer(examples["prompt"], truncation=True, max_length=MAX_LENGTH)


# Cache de tokenização - evita reprocessar datasets grandes
cache_dir = f"../data/.token_cache_{args.model}_{args.dataset}_{args.strategy}"
use_cache = args.token_cache and os.path.exists(cache_dir)

if use_cache:
    print(f"\n💾 Carregando tokenização em cache de: {cache_dir}")
    tokenized_datasets = load_from_disk(cache_dir)
else:
    print("\n🔄 Tokenizando dataset...")
    train_dataset = Dataset.from_dict({"prompt": X_train, "label": y_train})
    val_dataset = Dataset.from_dict({"prompt": X_val, "label": y_val})
    test_dataset = Dataset.from_dict({"prompt": X_test, "label": y_test})

    raw_datasets = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })

    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=4,  # Paralelizar tokenização
        desc="Tokenizando"
    )

    if args.token_cache:
        print(f"\n💾 Salvando cache em: {cache_dir}")
        tokenized_datasets.save_to_disk(cache_dir)


id2label = {0: "Benign", 1: "Malicious"}
label2id = {"Benign": 0, "Malicious": 1}

# ============================================================================
# CARREGAR MODELO
# ============================================================================
bnb_config = None
if args.model in ["Qwen", "Gemma-2B", "Llama-3", "TinyLlama"]:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

# Tentar Unsloth se disponível e requisitado
use_unsloth = args.use_unsloth and UNSLOTH_AVAILABLE and args.model in ["Llama-3", "Qwen", "Gemma-2B", "TinyLlama"]

if use_unsloth:
    print("\n🚀 Carregando modelo com Unsloth...")
    try:
        # CRUCIAL: FastLanguageModel retorna (model, tokenizer)
        # O bug comum é confundir os retornos ou não configurar tokens antes do PEFT
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            num_labels=2,
            max_seq_length=MAX_LENGTH,
            dtype=None,  # Auto-detect (usa bf16 se disponível)
            load_in_4bit=True,
            sequence_classification=True
        )

        # Configurar tokens ANTES de aplicar PEFT
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.eos_token

        # Configurar modelo corretamente
        model.config.use_cache = False
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id

        # VERIFICAR se model é realmente um objeto de modelo e não string
        if isinstance(model, str):
            raise TypeError(f"Unsloth retornou string ao invés de modelo: {model}")

        print(f"Modelo carregado: {type(model)}")
        print(f"Device do modelo: {next(model.parameters()).device}")

        tokenizer.padding_side = "right"

        # Aplicar LoRA via Unsloth
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=42,
            use_dora=args.dora,
        )
        model.print_trainable_parameters()

    except Exception as e:
        print(f"\n⚠️ Unsloth falhou: {e}")
        import traceback
        traceback.print_exc()
        print("Revertendo para transformers padrão...")
        use_unsloth = False

if not use_unsloth:
    print("\n📦 Carregando modelo com transformers padrão...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        quantization_config=bnb_config,
        device_map="auto" if bnb_config else None,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.model == "Gemma-2B":
        model.config.pad_token_id = tokenizer.pad_token_id
        tokenizer.padding_side = "right"
    elif args.model == "DEBERTa":
        model.config.pad_token_id = tokenizer.pad_token_id
    elif args.model in LLM_MODELS:
        tokenizer.padding_side = "left"

    # LoRA/DoRA otimizado
    target_modules_map = {
            # Encoders: Atenção (Q, K, V) + Camadas Densas (Feed-Forward e Saídas)
            # O uso da palavra "dense" vai mapear automaticamente o attention.output.dense, 
            # intermediate.dense e output.dense.
            "BERT-Base": ["query", "key", "value", "dense"],
            "distilBERT": ["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"],
            "DEBERTa": ["query_proj", "key_proj", "value_proj", "dense"],
            
            # Decoders / LLMs: Atenção Total (Q, K, V, O) + Perceptron Multicamadas (Gate, Up, Down)
            "Llama-3": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "TinyLlama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "Qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "Gemma-2B": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        }
    lora_config = LoraConfig(
        use_dora=args.dora,
        r=16,  # Reduzido de 16 para 8
        lora_alpha=32,  # Proporcional
        target_modules=target_modules_map.get(args.model, ["query", "value"]),
        task_type=TaskType.SEQ_CLS,
        bias="none",
        lora_dropout=0.0,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# ============================================================================
# BATCH SIZE OTIMIZADO POR MODELO
# ============================================================================
batch_sizes = {
    "distilBERT": 64,
    "BERT-Base": 48,
    "DEBERTa": 48,
    "Llama-3": 64,
    "Qwen": 32,
    "TinyLlama": 48,
    "Gemma-2B": 32,
}
batch_size = batch_sizes.get(args.model, 32)

lr = 3e-5

outputdir = f"models/{args.model}/{args.dataset}-{args.strategy}-{args.parameters}/"
best = outputdir + "best/"

# ============================================================================
# TRAINING ARGS OTIMIZADOS
# ============================================================================
# Gradient accumulation maior = menos overhead de comunicação
# Mais workers no dataloader = GPU menos ociosa
grad_accumulation = 4 if args.model in LLM_MODELS else 2

if use_unsloth:
    training_args = UnslothTrainingArguments(
        output_dir=outputdir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accumulation,
        num_train_epochs=args.num_epochs,
        logging_strategy="steps",
        logging_steps=100,  # Log a cada 100 steps (mais granular)
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="AUC",
        bf16=True,
        greater_is_better=True,
        warmup_ratio=0.05,
        weight_decay=0.08,
        optim="adamw_8bit",
        gradient_checkpointing=True,
        dataloader_num_workers=8,  # Aumentado de 4 para 8
        dataloader_prefetch_factor=4,  # Aumentado de 2 para 4
    )
else:
    training_args = TrainingArguments(
        output_dir=outputdir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accumulation,
        num_train_epochs=args.num_epochs,
        logging_strategy="steps",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="AUC",
        bf16=True,
        greater_is_better=True,
        warmup_ratio=0.05,
        weight_decay=0.08,
        optim=optim,
        gradient_checkpointing=True,
        dataloader_num_workers=8,
        dataloader_prefetch_factor=4,
    )

# ============================================================================
# TRAINER
# ============================================================================
if use_unsloth:
    trainer = UnslothTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Reduzido de 5 para 3
    )
else:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )


# ============================================================================
# OPTUNA (se ativado)
# ============================================================================
if args.optuna:
    def model_init():
        if use_unsloth:
            temp_model, _ = FastLanguageModel.from_pretrained(
                model_name=model_path,
                num_labels=2,
                max_seq_length=MAX_LENGTH,
                dtype=None,
                load_in_4bit=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.bos_token is None:
                tokenizer.bos_token = tokenizer.eos_token
            temp_model.config.use_cache = False
            temp_model.config.pad_token_id = tokenizer.pad_token_id
            temp_model.config.bos_token_id = tokenizer.bos_token_id
            temp_model.config.eos_token_id = tokenizer.eos_token_id

            temp_model = FastLanguageModel.get_peft_model(
                temp_model,
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.0,
                bias="none",
                use_gradient_checkpointing=True,
                random_state=42,
                use_dora=args.dora,
            )
            return temp_model
        else:
            temp_model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=2,
                id2label=id2label,
                label2id=label2id,
                quantization_config=bnb_config,
                device_map="auto" if bnb_config else None,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if args.model == "Gemma-2B":
                temp_model.config.pad_token_id = tokenizer.pad_token_id
            elif args.model == "DEBERTa":
                temp_model.config.pad_token_id = tokenizer.pad_token_id

            l_config = LoraConfig(
                use_dora=args.dora,
                r=8,
                lora_alpha=16,
                target_modules=target_modules_map.get(args.model, ["query", "value"]),
                task_type=TaskType.SEQ_CLS,
                bias="none",
                lora_dropout=0.05,
            )
            return get_peft_model(temp_model, l_config)

    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.03, 0.10),
        }

    search_train = tokenized_datasets["train"].shuffle(seed=42).select(range(min(10000, len(tokenized_datasets["train"]))))
    search_val = tokenized_datasets["validation"].shuffle(seed=42).select(range(min(2000, len(tokenized_datasets["validation"]))))

    trainer = Trainer(
        model=None,
        model_init=model_init,
        args=training_args,
        train_dataset=search_train,
        eval_dataset=search_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\n🔍 Iniciando busca de hiperparâmetros com Optuna (10k samples)...")
    best_run = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=hp_space,
        n_trials=6
    )

    print(f"\n✅ Melhores Hiperparâmetros: {best_run.hyperparameters}")

    for n, v in best_run.hyperparameters.items():
        setattr(trainer.args, n, v)

    trainer.train_dataset = tokenized_datasets["train"]
    trainer.eval_dataset = tokenized_datasets["validation"]

    print(f"\n🚀 Iniciando treinamento FINAL com parâmetros otimizados...\n")
    trainer.train(resume_from_checkpoint=args.check)
else:
    print(f"\n🚀 Iniciando treinamento do {args.model}...\n")
    trainer.train(resume_from_checkpoint=args.check)

# ============================================================================
# SALVAR E AVALIAR
# ============================================================================
trainer.save_model(best)
tokenizer.save_pretrained(best)

if use_unsloth:
    model.save_pretrained(best)

print(f"\n✅ Modelo salvo em {best}")

print("\n📊 Avaliação no conjunto de teste:")
predictions = trainer.predict(tokenized_datasets["test"])

logits = predictions.predictions
labels = predictions.label_ids
metrics = compute_metrics((logits, labels))

path = "results/optimized-bert-classifier.csv"
os.makedirs("results", exist_ok=True)
exists = os.path.exists(path)

pd.DataFrame({
    "model": [f"{model_path}-{args.dataset}-{args.parameters}"],
    "strategy": [args.strategy],
    "accuracy": [metrics["Accuracy"]],
    "auc": [metrics["AUC"]],
    "f1-score": [metrics["F1-Score"]],
}).to_csv(path, index=False, header=not exists, mode="a")

print(metrics)
print("\n✅ Treinamento final e salvo com sucesso!")
