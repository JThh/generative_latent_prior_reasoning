"""
Extract residual stream activations from reasoning models during
chain-of-thought (CoT) generation on math/logic datasets.

Supports:
  - Multiple traces per prompt via temperature sampling
  - Correctness tracking for correctness-conditioned GLP training
  - Reasoning phase assignment per token
  - Cognitive operation labeling per token
  - Streaming to disk via MemmapWriter
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from glp.utils_acts import MemmapWriter
from reasoning.label_cot_segments import (
    CognitiveLabeler,
    assign_all_phases,
    summarize_labels,
)
from reasoning.benchmarks import (
    load_reasoning_corpus,
    DATASET_REGISTRY,
    REASONING_CORPUS,
    format_chat_prompt,
    format_reasoning_prompt,
    extract_final_answer,
    check_correctness,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================
#   Phase index mapping
# ==============================
PHASE_TO_IDX = {
    "pre_reasoning": 0,
    "early_reasoning": 1,
    "mid_reasoning": 2,
    "late_reasoning": 3,
    "post_reasoning": 4,
}


@dataclass
class SaveReasoningActsConfig:
    # model
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    torch_dtype: str = "bfloat16"
    device: str = "cuda:0"
    # layers to extract from (0-indexed)
    # - Single int: extract from that layer only
    # - List of ints: extract from those layers
    # - null/None: auto-select middle layer
    # - "all": extract from every layer (warning: extremely large output)
    # - "every_4th": extract every 4th layer (recommended for multi-layer)
    layer_idx: Optional[int] = None
    layer_prefix: str = "model.layers"
    retain: str = "output"  # "input" or "output" of the layer
    # dataset — use any dataset name from DATASET_REGISTRY, a corpus
    # category from REASONING_CORPUS (e.g. "activation_caching"), or "all"
    dataset: str = "numina_math_cot"  # primary dataset for activation caching
    dataset_split: str = "train"
    max_examples: Optional[int] = None
    # generation — multi-trace support (guidance: 8-16 traces, T=0.6-1.0)
    max_new_tokens: int = 4096
    num_traces_per_prompt: int = 8       # guidance recommends 8-16
    temperature: float = 0.7             # range: 0.6-1.0
    do_sample: bool = False              # auto-set True if num_traces > 1
    use_chat_template: bool = True
    # task type tagging for conditioning
    task_type: Optional[str] = None      # "math", "code", "logic" — auto-detected from source
    # output
    output_dir: str = "data/reasoning_acts"
    memmap_file_size: int = 10_000_000   # elements per memmap file
    # labeling
    enable_labeling: bool = True
    enable_phase_tracking: bool = True
    # correctness conditioning
    track_correctness: bool = True


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping.get(dtype_str, torch.bfloat16)


def determine_middle_layer(model) -> int:
    if hasattr(model, "config"):
        n_layers = getattr(model.config, "num_hidden_layers", None)
        if n_layers is not None:
            return n_layers // 2
    for attr in ["model.layers", "transformer.h", "gpt_neox.layers"]:
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return len(obj) // 2
        except AttributeError:
            continue
    raise ValueError("Could not determine number of layers")


@torch.no_grad()
def extract_cot_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    config: SaveReasoningActsConfig,
    layer_name: str,
) -> dict:
    """
    Generate a CoT trace and capture residual stream activations
    at each generated token via a forward hook.

    Returns dict with:
        - activations: list of (hidden_dim,) tensors
        - generated_text: full generated text
        - generated_tokens: list of token strings
        - input_length: number of input tokens
    """
    # Prepare input
    if config.use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        prompt = format_chat_prompt(question, tokenizer)
    else:
        prompt = format_reasoning_prompt(question)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)
    input_length = inputs["input_ids"].shape[1]

    # Hook to capture activations at the last (newly generated) token
    activations = []

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        last_token_act = hidden[:, -1, :].detach().cpu().float()
        activations.append(last_token_act.squeeze(0))

    def input_hook_fn(module, input, output):
        inp = input[0] if isinstance(input, tuple) else input
        last_token_act = inp[:, -1, :].detach().cpu().float()
        activations.append(last_token_act.squeeze(0))

    # Get target layer module
    layer_module = model
    for part in layer_name.split("."):
        layer_module = getattr(layer_module, part)

    hook_func = input_hook_fn if config.retain == "input" else hook_fn
    handle = layer_module.register_forward_hook(hook_func)

    try:
        gen_kwargs = {"max_new_tokens": config.max_new_tokens}
        if config.do_sample or config.num_traces_per_prompt > 1:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = config.temperature
        else:
            gen_kwargs["do_sample"] = False

        output_ids = model.generate(**inputs, **gen_kwargs)

        generated_ids = output_ids[0, input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_tokens = [
            tokenizer.decode([tid], skip_special_tokens=False)
            for tid in generated_ids.tolist()
        ]
    finally:
        handle.remove()

    # Align activations with generated tokens
    n_generated = len(generated_ids)
    if len(activations) > n_generated:
        activations = activations[-n_generated:]
    elif len(activations) < n_generated:
        logger.warning(
            f"Got {len(activations)} activations but {n_generated} tokens. "
            f"Truncating tokens to match."
        )
        generated_tokens = generated_tokens[: len(activations)]

    return {
        "activations": activations,
        "generated_text": generated_text,
        "generated_tokens": generated_tokens,
        "input_length": input_length,
    }


def main():
    config_base = OmegaConf.structured(SaveReasoningActsConfig())
    OmegaConf.set_struct(config_base, False)
    config_cli = OmegaConf.from_cli()
    config_path = config_cli.pop("config", None)
    config_file = OmegaConf.load(config_path) if config_path else OmegaConf.create()
    config = OmegaConf.merge(config_base, config_file, config_cli)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Config: {config}")
    logger.info(f"Output directory: {output_dir}")

    # ── Load model ──────────────────────────────────────────────
    dtype = get_torch_dtype(config.torch_dtype)
    logger.info(f"Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        device_map=config.device,
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine layer
    layer_idx = config.layer_idx
    if layer_idx is None:
        layer_idx = determine_middle_layer(model)
        logger.info(f"Auto-selected middle layer: {layer_idx}")
    layer_name = f"{config.layer_prefix}.{layer_idx}"
    hidden_dim = model.config.hidden_size
    logger.info(f"Extracting from: {layer_name} ({config.retain}), dim={hidden_dim}")

    # ── Load dataset ────────────────────────────────────────────
    # Supports individual datasets (gsm8k, math, numina_math_cot, openr1_math, ...),
    # corpus categories (activation_caching, math_competition, coding, ...),
    # or "all" for the full reasoning corpus.
    examples = load_reasoning_corpus(
        datasets=config.dataset,
        max_examples_per_dataset=config.max_examples,
        split=config.dataset_split,
    )
    logger.info(f"Loaded {len(examples)} examples from dataset='{config.dataset}' split='{config.dataset_split}'")

    # ── Set up outputs ──────────────────────────────────────────
    writer = MemmapWriter(
        output_dir=output_dir,
        file_size=config.memmap_file_size,
        dtype=np.dtype("float32"),
    )

    labeler = CognitiveLabeler() if config.enable_labeling else None
    all_cognitive_labels = {name: [] for name in labeler.label_names} if labeler else {}
    all_phase_labels = []          # int per token
    all_correctness = []           # int per token (0/1/-1 for unknown)
    token_to_example = []
    token_to_trace = []            # which trace index within an example
    all_metadata = []

    # Running statistics
    running_sum = torch.zeros(hidden_dim, dtype=torch.float64)
    running_sq_sum = torch.zeros(hidden_dim, dtype=torch.float64)
    total_tokens = 0

    # ── Process examples ────────────────────────────────────────
    logger.info("Starting activation extraction...")
    start_time = time.time()
    num_traces = config.num_traces_per_prompt

    for ex_idx, example in enumerate(tqdm(examples, desc="Extracting CoT activations")):
        question = example["question"]

        # Determine gold answer for correctness tracking
        gold_answer = None
        if config.track_correctness:
            gold_answer = example.get("numeric_answer") or example.get("answer")
            if gold_answer is not None:
                gold_answer = str(gold_answer)

        for trace_idx in range(num_traces):
            try:
                result = extract_cot_activations(
                    model=model,
                    tokenizer=tokenizer,
                    question=question,
                    config=config,
                    layer_name=layer_name,
                )
            except Exception as e:
                logger.warning(f"Error on example {ex_idx} trace {trace_idx}: {e}")
                continue

            acts = result["activations"]
            if len(acts) == 0:
                continue

            # Determine correctness of this trace
            trace_correct = -1  # unknown
            if config.track_correctness and gold_answer is not None:
                pred = extract_final_answer(result["generated_text"])
                if pred is not None:
                    trace_correct = int(check_correctness(pred, gold_answer))

            # Write activations to memmap
            for act in acts:
                act_np = act.numpy().astype(np.float32)
                writer.write(act_np)

            # Update running statistics
            act_stack = torch.stack(acts)
            running_sum += act_stack.sum(dim=0).double()
            running_sq_sum += (act_stack ** 2).sum(dim=0).double()
            n_act = len(acts)
            total_tokens += n_act

            # Cognitive labels
            if labeler is not None:
                token_labels = labeler.label_tokens(result["generated_tokens"])
                for name in labeler.label_names:
                    all_cognitive_labels[name].extend(token_labels[name].tolist())

            # Phase labels
            if config.enable_phase_tracking:
                phases = assign_all_phases(n_act, result["generated_tokens"])
                all_phase_labels.extend([PHASE_TO_IDX.get(p, 2) for p in phases])

            # Correctness per token (same for all tokens in a trace)
            all_correctness.extend([trace_correct] * n_act)

            # Track mappings
            token_to_example.extend([ex_idx] * n_act)
            token_to_trace.extend([trace_idx] * n_act)

            # Per-trace metadata
            all_metadata.append({
                "example_idx": ex_idx,
                "trace_idx": trace_idx,
                "question": question[:200],
                "n_generated_tokens": n_act,
                "generated_text_preview": result["generated_text"][:300],
                "correct": trace_correct,
            })

        # Periodic progress
        if (ex_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"Processed {ex_idx + 1}/{len(examples)} examples, "
                f"{total_tokens} tokens, {elapsed:.1f}s"
            )

    # ── Flush and save ──────────────────────────────────────────
    writer.flush()
    (output_dir / "dtype.txt").write_text("float32")

    # Normalization statistics
    if total_tokens > 0:
        mean = (running_sum / total_tokens).float()
        var = ((running_sq_sum / total_tokens) - mean.double() ** 2).float()
        var = var.clamp(min=1e-8)
    else:
        mean = torch.zeros(hidden_dim)
        var = torch.ones(hidden_dim)

    torch.save(
        {"mean": mean.unsqueeze(0), "var": var.unsqueeze(0)},
        output_dir / "rep_statistics.pt",
    )

    # Cognitive labels
    if labeler is not None and all_cognitive_labels:
        with open(output_dir / "cognitive_labels.json", "w") as f:
            json.dump(all_cognitive_labels, f)
        for name, arr in all_cognitive_labels.items():
            n_pos = sum(arr)
            logger.info(f"  Label '{name}': {n_pos}/{len(arr)} ({n_pos / max(len(arr), 1) * 100:.1f}%)")

    # Phase labels
    if config.enable_phase_tracking and all_phase_labels:
        with open(output_dir / "phase_labels.json", "w") as f:
            json.dump(all_phase_labels, f)

    # Correctness labels
    if config.track_correctness and all_correctness:
        with open(output_dir / "correctness_labels.json", "w") as f:
            json.dump(all_correctness, f)
        n_correct = sum(1 for c in all_correctness if c == 1)
        n_incorrect = sum(1 for c in all_correctness if c == 0)
        n_unknown = sum(1 for c in all_correctness if c == -1)
        logger.info(f"  Correctness: {n_correct} correct, {n_incorrect} incorrect, {n_unknown} unknown tokens")

    # Metadata
    metadata = {
        "model_name": config.model_name,
        "layer_idx": layer_idx,
        "layer_name": layer_name,
        "retain": config.retain,
        "hidden_dim": hidden_dim,
        "dataset": config.dataset,
        "dataset_split": config.dataset_split,
        "n_examples": len(examples),
        "n_traces_per_prompt": num_traces,
        "n_traces_total": len(all_metadata),
        "total_tokens": total_tokens,
        "max_new_tokens": config.max_new_tokens,
        "token_to_example": token_to_example,
        "token_to_trace": token_to_trace,
        "examples": all_metadata,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    OmegaConf.save(config, output_dir / "config.yaml")

    elapsed = time.time() - start_time
    logger.info(
        f"Done! {total_tokens} activations from "
        f"{len(all_metadata)} traces ({len(examples)} examples) "
        f"saved to {output_dir} in {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
