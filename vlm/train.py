import evaluate
import pandas as pd
from scipy.special import softmax
import torch
from transformers import TrainingArguments, Trainer, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
from vlm import VisionLanguageModel
from VLMDataset import VLMDataset
import numpy as np 
accuracy = evaluate.load("accuracy")
auc_score = evaluate.load("roc_auc")
f1 = evaluate.load("f1")

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


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False 
        
def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True 


def phase1(llm_path:str, vision_path:str, num_tradutors:int,train_path:str,
           val_path:str, num_epochs:int,model_name:str, quantization:bool):
    

    model = VisionLanguageModel(llm_path, vision_path, num_tradutors)
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    train_dataset = VLMDataset(train_path,tokenizer, model_name)
    val_dataset  = VLMDataset(val_path,   tokenizer, model_name)
    freeze_model(model.cnn)
    freeze_model(model.llm)
    unfreeze_model(model.tradutor)


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.llm.config.pad_token_id = tokenizer.eos_token_id

    MODELS_LEFT_PAD  = ["Llama", "Qwen", "TinyLlama"]
    tokenizer.padding_side = "left" if any(m in llm_path for m in MODELS_LEFT_PAD) else "right"

    args = TrainingArguments(
        output_dir="checkpoints/phase1",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=1e-3,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="AUC",
        logging_strategy="epoch",
        bf16=torch.cuda.is_bf16_supported() and not quantization,
        fp16=not torch.cuda.is_bf16_supported() and not quantization,
        report_to="tensorboard",
        logging_dir="logs/phase1",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,  
    )

    trainer.train()
    trainer.save_model("checkpoints/phase1/best")
    
    return model, tokenizer

def phase2(model, tokenizer, train_path: str, val_path: str, 
           num_epochs: int, model_name: str, quantization:bool):
    
    train_dataset = VLMDataset(train_path,tokenizer, model_name)
    val_dataset  = VLMDataset(val_path,   tokenizer, model_name)
    freeze_model(model.cnn)
    unfreeze_model(model.llm)
    unfreeze_model(model.tradutor)



    args = TrainingArguments(
        output_dir="checkpoints/phase2",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="AUC",
        logging_strategy="epoch",
        bf16=torch.cuda.is_bf16_supported() and not quantization,
        fp16=not torch.cuda.is_bf16_supported() and not quantization,
        report_to="tensorboard",
        logging_dir="logs/phase2",
        warmup_ratio=0.1,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("checkpoints/phase2/best")
    
    return model, tokenizer, trainer


def evaluate_final(trainer, tokenizer, test_path, model_name):
    test_dataset = VLMDataset(test_path, tokenizer, model_name)
    predictions  = trainer.predict(test_dataset)
    metrics      = compute_metrics((predictions.predictions, predictions.label_ids))

    print("\nFinal Test Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    pd.DataFrame([metrics]).to_csv("results/final_metrics.csv", index=False)
    return metrics



def run_pipeline(llm_path, vision_path, num_tradutors, 
                 train_path, val_path, test_path, model_name, quantization):
    
    model, tokenizer = phase1(llm_path, vision_path, num_tradutors,
                               train_path, val_path, 10, model_name, quantization=False)
    
    model, tokenizer, trainer = phase2(model, tokenizer,
                               train_path, val_path, 5, model_name, quantization=quantization)
    
    evaluate_final(trainer, tokenizer, test_path, model_name)

