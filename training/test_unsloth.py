#!/usr/bin/env python3
"""Teste rápido para verificar se Unsloth funciona corretamente"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 1. Testar import (A classe certa!)
try:
    from unsloth import FastLanguageModel
    print("✅ Unsloth importado com sucesso")
except ImportError as e:
    print(f"❌ Unsloth não disponível: {e}")
    exit(1)
    
import torch
from transformers import AutoTokenizer

print("=== Teste Unsloth ===\n")

# 2. Testar carregamento do modelo
model_path = "Qwen/Qwen2.5-0.5B"
print(f"\n📦 Carregando: {model_path}")

try:
    # Usando FastLanguageModel com sequence_classification=True
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        num_labels=2,
        max_seq_length=192,
        dtype=None,
        load_in_4bit=True,
        sequence_classification=True, 
    )
    print("✅ Modelo carregado")
    print(f"   Tipo: {type(model)}")
    print(f"   Device: {next(model.parameters()).device}")
except Exception as e:
    print(f"❌ Erro ao carregar modelo: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 3. Configurar tokens
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.bos_token is None:
    tokenizer.bos_token = tokenizer.eos_token
    
model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
print("✅ Tokens configurados")

# 4. Aplicar LoRA
print("\n🔧 Aplicando LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth", # Otimizado para VRAM
    random_state=42,
)

print("✅ LoRA aplicado")
model.print_trainable_parameters()

# 5. Testar forward pass
print("\n🧪 Testando forward pass...")
tokenizer.padding_side = "right"
inputs = tokenizer("test prompt here", return_tensors="pt").to(model.device)
print(f"   Input shape: {inputs['input_ids'].shape}")

with torch.no_grad():
    outputs = model(**inputs)
    print(f"   Output logits shape: {outputs.logits.shape}")

print("\n✅✅✅ Unsloth funcionando perfeitamente! ✅✅✅")