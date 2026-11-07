"""Carregamento e inferência do LLM (`Llama-3.2-1B-Instruct` + `PeftModel`) de `esemble/multimodal.py`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.pipeline.constants import LLM_MAX_LENGTH, SYSTEM_INSTRUCTION


def load_llm_and_tokenizer(
    llm_path: Path,
    base_model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    device: str = "cpu",
) -> tuple[Any, Any]:
    """
    Carrega o `AutoModelForSequenceClassification` (`meta-llama/Llama-3.2-1B-Instruct`)
    e aplica o adaptador `PeftModel` de `llm_path`, configurando `pad_token = eos_token`,
    exatamente como em `esemble/multimodal.py` (linhas 129-151).
    """
    import torch
    from peft import PeftModel  # type: ignore
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

    torch_device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
    )
    llm_model = PeftModel.from_pretrained(base_model, str(llm_path))
    llm_model.to(torch_device)
    llm_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(llm_path), local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    llm_model.config.pad_token_id = tokenizer.eos_token_id

    return llm_model, tokenizer


def predict_llm_probability(
    llm_model: Any,
    tokenizer: Any,
    structured_prompt: str,
    device: str = "cpu",
    max_len: int = LLM_MAX_LENGTH,
) -> float:
    """
    Formata o prompt de 7 seções usando `SYSTEM_INSTRUCTION` e `apply_chat_template`
    (`TextInferenceDataset` em `esemble/multimodal.py`, linhas 63-88) e retorna `P(malicious)`.
    """
    import torch

    torch_device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    messages = [{"role": "user", "content": f"{SYSTEM_INSTRUCTION}\n\n{structured_prompt}"}]
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoding = tokenizer(
        formatted_text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(torch_device)
    attention_mask = encoding["attention_mask"].to(torch_device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=torch_device.type == "cuda"):
            outputs = llm_model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()

    return float(probs[0, 1])
