from dataclasses import replace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reasoning.save_reasoning_acts import (
    SaveReasoningActsConfig,
    configure_hf_downloads,
    determine_middle_layer,
    detect_stale_incomplete_weights,
    extract_cot_activations,
    get_torch_dtype,
    resolve_model_load_kwargs,
    resolve_model_source,
)


def get_num_hidden_layers(model) -> int:
    if hasattr(model, "config"):
        n_layers = getattr(model.config, "num_hidden_layers", None)
        if n_layers is not None:
            return n_layers
    for attr in ["model.layers", "transformer.h", "gpt_neox.layers"]:
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return len(obj)
        except AttributeError:
            continue
    raise ValueError("Could not determine number of layers")


def resolve_demo_layer_indices(cfg: SaveReasoningActsConfig, model) -> list[int]:
    n_layers = get_num_hidden_layers(model)
    layer_idx = cfg.layer_idx

    if layer_idx is None:
        candidates = [n_layers // 4, n_layers // 2, (3 * n_layers) // 4]
        return sorted(set(idx for idx in candidates if 0 <= idx < n_layers))

    if isinstance(layer_idx, int):
        return [layer_idx]

    if isinstance(layer_idx, (list, tuple)):
        return [int(idx) for idx in layer_idx if 0 <= int(idx) < n_layers]

    if layer_idx == "all":
        return list(range(n_layers))

    if layer_idx == "every_4th":
        return list(range(0, n_layers, 4))

    raise ValueError(f"Unsupported layer_idx value for demo: {layer_idx!r}")


def layer_name_for_index(cfg: SaveReasoningActsConfig, layer_idx: int) -> str:
    return f"{cfg.layer_prefix}.{layer_idx}"


def load_demo_model(cfg: SaveReasoningActsConfig):
    dtype = get_torch_dtype(cfg.torch_dtype)
    cache_dir = configure_hf_downloads(cfg)
    model_source, load_cache_dir = resolve_model_source(cfg, cache_dir)

    print(f"model_source={model_source}")
    print(f"cache_dir={load_cache_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=True,
        cache_dir=load_cache_dir,
        local_files_only=cfg.local_files_only,
    )

    stale_cache_msg = None
    if model_source == cfg.model_name:
        stale_cache_msg = detect_stale_incomplete_weights(cache_dir, cfg.model_name)
    if stale_cache_msg:
        raise RuntimeError(stale_cache_msg)

    load_kwargs, move_device = resolve_model_load_kwargs(cfg)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=dtype,
        trust_remote_code=True,
        cache_dir=load_cache_dir,
        local_files_only=cfg.local_files_only,
        low_cpu_mem_usage=True,
        **load_kwargs,
    )

    if move_device is not None:
        model = model.to(move_device)

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    layer_indices = resolve_demo_layer_indices(cfg, model)
    if not layer_indices:
        layer_indices = [determine_middle_layer(model)]

    return tokenizer, model, layer_indices


def print_token_table(trace_result: dict, limit: int = 40) -> None:
    header = f"{'idx':>4}  {'norm':>10}  {'mean':>10}  {'std':>10}  token"
    print(header)
    print("-" * len(header))
    for idx, (token, act) in enumerate(
        zip(trace_result["generated_tokens"], trace_result["activations"])
    ):
        if idx >= limit:
            break
        print(
            f"{idx:>4}  {act.norm().item():>10.4f}  "
            f"{act.mean().item():>10.4f}  {act.std().item():>10.4f}  {token!r}"
        )


def find_matching_token_indices(trace_result: dict, patterns: list[str]) -> list[int]:
    matches = []
    for idx, token in enumerate(trace_result["generated_tokens"]):
        if any(pattern and pattern in token for pattern in patterns):
            matches.append(idx)
    return matches


def find_marker_token_indices(trace_result: dict, markers: list[str]) -> list[int]:
    joined_text = ""
    token_char_ranges = []
    for token in trace_result["generated_tokens"]:
        start = len(joined_text)
        joined_text += token
        token_char_ranges.append((start, len(joined_text)))

    if not joined_text:
        return []

    spans = []
    for marker in markers:
        start = 0
        while True:
            idx = joined_text.find(marker, start)
            if idx == -1:
                break
            spans.append((idx, idx + len(marker)))
            start = idx + 1

    if not spans:
        return []

    matches = []
    for tok_idx, (tok_start, tok_end) in enumerate(token_char_ranges):
        if any(tok_start < span_end and tok_end > span_start for span_start, span_end in spans):
            matches.append(tok_idx)
    return sorted(set(matches))


def summarize_activation(act: torch.Tensor, top_k: int = 8) -> list[dict]:
    values, indices = torch.topk(act.abs(), k=min(top_k, act.numel()))
    rows = []
    for rank, (idx, magnitude) in enumerate(
        zip(indices.tolist(), values.tolist()),
        start=1,
    ):
        rows.append(
            {
                "rank": rank,
                "dim": idx,
                "value": float(act[idx].item()),
                "abs_value": float(magnitude),
            }
        )
    return rows


def activation_deltas(trace_result: dict) -> list[dict]:
    rows = []
    acts = trace_result["activations"]
    toks = trace_result["generated_tokens"]
    for idx in range(1, len(acts)):
        diff = acts[idx] - acts[idx - 1]
        rows.append(
            {
                "from_idx": idx - 1,
                "to_idx": idx,
                "from_token": toks[idx - 1],
                "to_token": toks[idx],
                "delta_norm": float(diff.norm().item()),
            }
        )
    rows.sort(key=lambda row: row["delta_norm"], reverse=True)
    return rows


def print_multi_layer_token_summary(
    trace_results_by_layer: dict[int, dict],
    patterns: list[str],
    top_k_dims: int,
    max_matches: int = 5,
) -> None:
    first_layer = next(iter(trace_results_by_layer))
    reference_result = trace_results_by_layer[first_layer]
    match_indices = find_marker_token_indices(reference_result, ["<think>", "</think>"])
    selection_label = "thinking-marker tokens"

    if not match_indices and patterns:
        match_indices = find_matching_token_indices(reference_result, patterns)
        selection_label = "pattern-matched tokens"

    if not match_indices:
        print("\nNo thinking-marker tokens or pattern-matched tokens were found.")
        return

    print(f"\nSelected {selection_label} across layers:")
    for token_idx in match_indices[:max_matches]:
        token = reference_result["generated_tokens"][token_idx]
        print(f"\nToken index {token_idx} | token={token!r}")
        for layer_idx, result in trace_results_by_layer.items():
            if token_idx >= len(result["activations"]):
                print(f"  layer {layer_idx}: missing activation")
                continue
            act = result["activations"][token_idx]
            print(f"  layer {layer_idx}: norm={act.norm().item():.4f}")
            for row in summarize_activation(act, top_k=top_k_dims):
                print(
                    f"    top{row['rank']}: dim={row['dim']} "
                    f"value={row['value']:.4f} abs={row['abs_value']:.4f}"
                )


def run_demo(
    question: str | list[str],
    patterns: list[str],
    top_k_dims: int = 8,
    max_tokens_to_print: int = 40,
    max_samples_to_show: int = 3,
    config: SaveReasoningActsConfig | None = None,
) -> list[dict]:
    cfg = config or SaveReasoningActsConfig()
    cfg = replace(cfg, num_traces_per_prompt=1)

    questions = [question] if isinstance(question, str) else list(question)
    if not questions:
        raise ValueError("At least one question is required.")

    tokenizer, model, layer_indices = load_demo_model(cfg)
    print(f"selected_layers={layer_indices}")
    print(f"model.device={model.device}")

    sample_summaries = []
    for sample_idx, sample_question in enumerate(questions[:max_samples_to_show], start=1):
        print("\n" + "=" * 80)
        print(f"Sample {sample_idx}")
        print(f"question={sample_question!r}")

        trace_results_by_layer = {}
        for layer_idx in layer_indices:
            layer_name = layer_name_for_index(cfg, layer_idx)
            trace_results_by_layer[layer_idx] = extract_cot_activations(
                model=model,
                tokenizer=tokenizer,
                question=sample_question,
                config=cfg,
                layer_name=layer_name,
            )

        reference_result = trace_results_by_layer[layer_indices[0]]

        print("\nGenerated text:\n")
        print(reference_result["generated_text"])

        print(f"\nToken-level activation summary (layer {layer_indices[0]}):")
        print_token_table(reference_result, limit=max_tokens_to_print)
        print_multi_layer_token_summary(
            trace_results_by_layer=trace_results_by_layer,
            patterns=patterns,
            top_k_dims=top_k_dims,
        )

        delta_rows = activation_deltas(reference_result)
        if delta_rows:
            print(f"\nLargest activation jumps (layer {layer_indices[0]}):")
            for row in delta_rows[:5]:
                print(
                    f"  {row['from_idx']}->{row['to_idx']}: "
                    f"delta_norm={row['delta_norm']:.4f} "
                    f"{row['from_token']!r} -> {row['to_token']!r}"
                )

        sample_summaries.append(
            {
                "question": sample_question,
                "layer_indices": layer_indices,
                "trace_results_by_layer": trace_results_by_layer,
            }
        )

    return sample_summaries


if __name__ == "__main__":
    demo_config = SaveReasoningActsConfig(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        model_source=None,
        cache_dir="/tmp/hf-cache",
        local_files_only=False,
        torch_dtype="bfloat16",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        max_new_tokens=256,
        num_traces_per_prompt=1,
        temperature=0.7,
        use_chat_template=True,
    )

    run_demo(
        question=[
            (
                "If 17 + 28 = 45, what is 45 - 19? "
                "Show your reasoning before the final answer."
            ),
            (
                "A store has 24 apples and sells 9. Then it receives 6 more. "
                "How many apples does it have now? Show your reasoning first."
            ),
            (
                "What is 36 divided by 6, then plus 8? "
                "Explain the steps before the final answer."
            ),
        ],
        patterns=["Let", "So", "Therefore", "=", "2", "6"],
        top_k_dims=8,
        max_tokens_to_print=40,
        max_samples_to_show=3,
        config=demo_config,
    )
