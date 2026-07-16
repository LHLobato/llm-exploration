"""
Diagnóstico: verifica se os token IDs usados pra extrair logits no zero/few-shot
(predict_zero_few_shot) realmente correspondem ao token que o modelo gera em contexto.

Uso:
    python diagnose_token_ids.py --model Llama-3 --modelpath meta-llama/Llama-3.2-1B-Instruct
    python diagnose_token_ids.py --model Qwen --modelpath Qwen/Qwen2.5-1.5B-Instruct
    python diagnose_token_ids.py --model Gemma --modelpath google/gemma-3-1b-it
"""
import argparse
import os

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

SYSTEM_INSTRUCTION = (
    "You are a cybersecurity classifier. "
    "Analyze the domain record below and classify it as benign or malicious. Use 0 for benign and 1 for malicious"
)

CHAT_TEMPLATE_MODELS = ["Llama-3", "Qwen", "Gemma", "TinyLlama"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--modelpath", type=str, required=True)
    return parser.parse_args()


def main():
    args = get_args()

    tokenizer = AutoTokenizer.from_pretrained(args.modelpath)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right" if args.model == "Gemma" else "left"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.modelpath, quantization_config=bnb_config, device_map="auto"
    )
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id

    # --- 1) IDs usados hoje no script (tokenização isolada, sem contexto) ---
    token_0_isolated = tokenizer("0", add_special_tokens=False).input_ids[-1]
    token_1_isolated = tokenizer("1", add_special_tokens=False).input_ids[-1]
    print("=== Tokenização isolada (o que o script usa hoje) ===")
    print(f"  '0' -> id={token_0_isolated}  repr={tokenizer.convert_ids_to_tokens([token_0_isolated])}")
    print(f"  '1' -> id={token_1_isolated}  repr={tokenizer.convert_ids_to_tokens([token_1_isolated])}")

    # --- 2) Prompt real formatado (mesmo formato do format_chat_prompt) ---
    example_prompt = "[name]: fr7ehgd.duckdns.org\n[entropy]: 3.6819\n[whois]: present=no"
    messages = [{"role": "user", "content": f"{SYSTEM_INSTRUCTION}\n\n{example_prompt}"}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print("\n=== Prompt formatado (final) ===")
    print(repr(formatted[-80:]))  # mostra só o final, onde a geração começa

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    # --- 3) Geração real: qual token o modelo de fato produz na próxima posição? ---
    with torch.no_grad():
        gen_out = model.generate(
            **inputs, max_new_tokens=1, do_sample=False, pad_token_id=tokenizer.pad_token_id
        )
    real_next_token_id = gen_out[0][-1].item()
    print("\n=== Token real gerado pelo modelo (greedy, 1 token) ===")
    print(f"  id={real_next_token_id}  repr={tokenizer.convert_ids_to_tokens([real_next_token_id])}")

    # --- 4) Top-5 candidatos reais na última posição do forward pass ---
    with torch.no_grad():
        out = model(**inputs)
    last_idx = inputs["attention_mask"].sum(dim=1) - 1  # posição real do último token (funciona p/ left ou right pad)
    logits_last = out.logits[torch.arange(len(last_idx)), last_idx, :]
    topk = torch.topk(logits_last, k=5, dim=-1)
    print("\n=== Top-5 tokens mais prováveis na posição de geração ===")
    for score, idx in zip(topk.values[0].tolist(), topk.indices[0].tolist()):
        marker = ""
        if idx == token_0_isolated:
            marker = "  <-- é o id usado hoje pra '0'"
        if idx == token_1_isolated:
            marker = "  <-- é o id usado hoje pra '1'"
        print(f"  id={idx:>7}  logit={score:8.3f}  repr={tokenizer.convert_ids_to_tokens([idx])}{marker}")

    # --- 5) Veredito ---
    print("\n=== Veredito ===")
    ids_in_top5 = set(topk.indices[0].tolist())
    if token_0_isolated in ids_in_top5 or token_1_isolated in ids_in_top5:
        print("  Pelo menos um dos IDs isolados aparece no top-5 real. Pode estar OK, mas confira o rank exato.")
    else:
        print("  BUG CONFIRMADO: nenhum dos IDs isolados ('0'/'1' tokenizados fora de contexto)")
        print("  aparece entre os tokens mais prováveis que o modelo realmente gera.")
        print("  predict_zero_few_shot está lendo logits de tokens essencialmente aleatórios.")


if __name__ == "__main__":
    main()
