"""
Simple utility to inspect reasoning-model generation token by token.

Shows:
  - Raw generated token IDs (no decoding)
  - Per-token string forms
  - Full decoded output with and without special-token skipping

Optional:
  - Stop generation as soon as `</think>` appears (default), useful for fast activation collection.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def pick_dtype(dtype_str: str, device: str) -> torch.dtype:
    if dtype_str == "auto":
        if device.startswith("cuda"):
            return torch.bfloat16
        if device == "mps":
            return torch.float16
        return torch.float32
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_str]


def sample_next_token(logits: torch.Tensor, do_sample: bool, temperature: float) -> int:
    if do_sample:
        probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())
    return int(torch.argmax(logits, dim=-1).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful reasoning assistant.")
    parser.add_argument("--stop_at_think_start", action="store_true")
    parser.add_argument("--stop_at_think_end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop_on_eos", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = pick_dtype(args.dtype, device)

    print(f"[info] device={device} dtype={dtype} model={args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": args.prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = args.prompt

    model_inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask", None)

    think_start_ids = tokenizer.encode("<think>", add_special_tokens=False)
    think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    generated_ids: list[int] = []

    past_key_values = None
    next_input_ids = input_ids
    next_attention_mask = attention_mask

    print("[info] stepwise generation begins")
    with torch.no_grad():
        for step in range(args.max_new_tokens):
            outputs = model(
                input_ids=next_input_ids,
                attention_mask=next_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            token_id = sample_next_token(logits, do_sample=args.do_sample, temperature=args.temperature)
            generated_ids.append(token_id)

            tok_piece = tokenizer.convert_ids_to_tokens([token_id])[0]
            tok_decoded = tokenizer.decode([token_id], skip_special_tokens=False)
            print(f"step={step:03d} id={token_id:<8} piece={repr(tok_piece)} decoded={repr(tok_decoded)}")

            if args.stop_on_eos and tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                print("[stop] EOS encountered")
                break

            if args.stop_at_think_start and think_start_ids:
                if len(generated_ids) >= len(think_start_ids) and generated_ids[-len(think_start_ids):] == think_start_ids:
                    print(f"[stop] detected <think> start token sequence: {think_start_ids}")
                    break
            if args.stop_at_think_end and think_end_ids:
                if len(generated_ids) >= len(think_end_ids) and generated_ids[-len(think_end_ids):] == think_end_ids:
                    print(f"[stop] detected </think> end token sequence: {think_end_ids}")
                    break

            past_key_values = outputs.past_key_values
            next_input_ids = torch.tensor([[token_id]], device=device)
            next_attention_mask = None

    print("\n=== RAW IDS (no decoding) ===")
    print(generated_ids)
    print(f"count={len(generated_ids)}")

    print("\n=== DECODED (skip_special_tokens=False) ===")
    print(tokenizer.decode(generated_ids, skip_special_tokens=False))

    print("\n=== DECODED (skip_special_tokens=True) ===")
    print(tokenizer.decode(generated_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
