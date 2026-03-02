"""
End-to-end benchmark evaluation demonstrating that GLP-guided
activation interventions can improve reasoning accuracy without
additional model training.

Evaluates on GSM8K and MATH benchmarks, comparing:
  1. Baseline (no intervention)
  2. Direct activation addition (no GLP projection)
  3. GLP-guided on-manifold steering

Reports accuracy, CoT length, per-operation frequency, and
accuracy-per-token efficiency.
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
from baukit import TraceDict
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from glp.denoiser import load_glp
from glp.script_steer import postprocess_on_manifold_wrapper
from reasoning.benchmarks import (
    load_gsm8k,
    load_math,
    format_chat_prompt,
    extract_final_answer,
    check_correctness,
)
from reasoning.label_cot_segments import CognitiveLabeler, summarize_labels
from reasoning.reasoning_dataset import ReasoningActDataset
from reasoning.script_reasoning_steer import (
    compute_all_steering_directions,
    reasoning_intervention,
    adaptive_reasoning_depth_intervention,
    error_recovery_intervention,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReasoningImproveConfig:
    # Reasoning model
    reasoning_model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    torch_dtype: str = "bfloat16"
    device: str = "cuda:0"
    # GLP model
    glp_weights_folder: str | None = None  # None = skip GLP projection
    glp_ckpt_name: str = "final"
    glp_u: float = 0.5
    glp_num_timesteps: int = 20
    # Activation data for steering directions
    acts_folder: str = "data/reasoning_acts"
    # Steering directions (pre-computed .pt file, or None to compute fresh)
    steering_directions_path: str | None = None
    # Steering configuration
    target_layer_prefix: str = "model.layers"
    target_layer_idx: int | None = None
    steer_mode: str = "static"  # "static", "adaptive_depth", "error_recovery"
    steer_alphas: dict = field(default_factory=lambda: {
        "verification": 3.0,
        "overthinking": -2.0,
        "answer_crystallisation": 1.0,
    })
    # Benchmark
    benchmark: str = "gsm8k"  # "gsm8k" or "math"
    benchmark_split: str = "test"
    max_examples: int | None = None
    # Generation
    max_new_tokens: int = 2048
    seed: int = 42
    # Evaluation methods to run
    run_baseline: bool = True
    run_direct_steering: bool = True
    run_glp_steering: bool = True
    # Output
    save_folder: str = "runs/reasoning_improvement"


@torch.no_grad()
def generate_cot(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    max_new_tokens: int = 2048,
    hook_fn=None,
    hook_layer: str | None = None,
    seed: int = 42,
) -> dict:
    """Generate a CoT trace, optionally with a steering hook."""
    torch.manual_seed(seed)

    prompt = format_chat_prompt(question, tokenizer)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(model.device)
    input_length = inputs["input_ids"].shape[1]

    if hook_fn is not None and hook_layer is not None:
        # Get layer module and register hook
        layer_module = model
        for part in hook_layer.split("."):
            layer_module = getattr(layer_module, part)

        handle = layer_module.register_forward_hook(
            lambda module, inp, out: hook_fn(out, hook_layer, inp)
        )
        try:
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        finally:
            handle.remove()
    else:
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

    generated_ids = output_ids[0, input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    generated_tokens = [
        tokenizer.decode([tid], skip_special_tokens=False)
        for tid in generated_ids.tolist()
    ]

    return {
        "generated_text": generated_text,
        "generated_tokens": generated_tokens,
        "n_tokens": len(generated_ids),
    }


def evaluate_single(
    result: dict,
    gold_answer: str,
    labeler: CognitiveLabeler,
) -> dict:
    """Evaluate a single generation result."""
    cot_text = result["generated_text"]
    n_tokens = result["n_tokens"]

    # Extract and check answer
    predicted = extract_final_answer(cot_text)
    correct = check_correctness(predicted, gold_answer) if predicted else False

    # Label cognitive operations
    token_labels = labeler.label_tokens(result["generated_tokens"])
    label_summary = summarize_labels(token_labels)

    return {
        "correct": correct,
        "predicted_answer": predicted,
        "n_tokens": n_tokens,
        "cognitive_summary": {
            name: stats["fraction"] for name, stats in label_summary.items()
        },
    }


def main(device="cuda:0"):
    default_config = OmegaConf.structured(ReasoningImproveConfig)
    OmegaConf.set_struct(default_config, False)
    config = OmegaConf.merge(default_config, OmegaConf.from_cli())

    save_folder = Path(config.save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, save_folder / "config.yaml")

    # ── Load model ───────────────────────────────────────────
    from reasoning.save_reasoning_acts import get_torch_dtype, determine_middle_layer

    dtype = get_torch_dtype(config.torch_dtype)
    logger.info(f"Loading: {config.reasoning_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.reasoning_model_name, trust_remote_code=True
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        config.reasoning_model_name,
        torch_dtype=dtype,
        device_map=config.device,
        trust_remote_code=True,
    )
    hf_model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Target layer
    target_layer_idx = config.target_layer_idx
    if target_layer_idx is None:
        target_layer_idx = determine_middle_layer(hf_model)
    target_layer = f"{config.target_layer_prefix}.{target_layer_idx}"
    logger.info(f"Steering at: {target_layer}")

    # ── Load benchmark ───────────────────────────────────────
    if config.benchmark == "gsm8k":
        examples = load_gsm8k(split=config.benchmark_split, max_examples=config.max_examples)
        get_gold = lambda ex: str(ex["numeric_answer"]) if ex.get("numeric_answer") is not None else None
    elif config.benchmark == "math":
        examples = load_math(split=config.benchmark_split, max_examples=config.max_examples)
        get_gold = lambda ex: ex.get("answer")
    else:
        raise ValueError(f"Unknown benchmark: {config.benchmark}")
    logger.info(f"Loaded {len(examples)} {config.benchmark} examples")

    # ── Load steering directions ─────────────────────────────
    if config.steering_directions_path and os.path.exists(config.steering_directions_path):
        directions = torch.load(config.steering_directions_path, map_location="cpu")
        logger.info(f"Loaded {len(directions)} steering directions from file")
    else:
        dataset = ReasoningActDataset(data_dir=config.acts_folder, load_labels=True)
        directions = compute_all_steering_directions(dataset)
        logger.info(f"Computed {len(directions)} steering directions")

    # ── Set up GLP projection ────────────────────────────────
    glp_postprocess = None
    if config.glp_weights_folder and config.run_glp_steering:
        glp_model = load_glp(
            config.glp_weights_folder, device=device, checkpoint=config.glp_ckpt_name
        )
        glp_postprocess = postprocess_on_manifold_wrapper(
            glp_model, u=config.glp_u, num_timesteps=config.glp_num_timesteps
        )
        logger.info("GLP on-manifold projection loaded")

    # ── Build hooks ──────────────────────────────────────────
    labeler = CognitiveLabeler()
    alphas = dict(config.steer_alphas)
    filtered_dirs = {k: v for k, v in directions.items() if k in alphas}
    filtered_alphas = {k: v for k, v in alphas.items() if k in directions}

    # Direct steering (no GLP)
    direct_hook = None
    if config.run_direct_steering and filtered_dirs:
        direct_hook = reasoning_intervention(
            filtered_dirs, filtered_alphas, postprocess_fn=None
        )

    # GLP-steered
    glp_hook = None
    if config.run_glp_steering and filtered_dirs and glp_postprocess is not None:
        glp_hook = reasoning_intervention(
            filtered_dirs, filtered_alphas, postprocess_fn=glp_postprocess
        )

    # ── Run evaluation ───────────────────────────────────────
    methods = {}
    if config.run_baseline:
        methods["baseline"] = None
    if config.run_direct_steering and direct_hook is not None:
        methods["direct_steering"] = direct_hook
    if config.run_glp_steering and glp_hook is not None:
        methods["glp_steering"] = glp_hook

    all_results = {method: [] for method in methods}

    for ex_idx, example in enumerate(tqdm(examples, desc="Evaluating")):
        question = example["question"]
        gold = get_gold(example)
        if gold is None:
            continue

        for method_name, hook_fn in methods.items():
            try:
                result = generate_cot(
                    model=hf_model,
                    tokenizer=tokenizer,
                    question=question,
                    max_new_tokens=config.max_new_tokens,
                    hook_fn=hook_fn,
                    hook_layer=target_layer if hook_fn else None,
                    seed=config.seed,
                )
                eval_result = evaluate_single(result, gold, labeler)
                eval_result["example_idx"] = ex_idx
                eval_result["question"] = question[:200]
                eval_result["gold_answer"] = gold
                all_results[method_name].append(eval_result)
            except Exception as e:
                logger.warning(f"Error on example {ex_idx} ({method_name}): {e}")

        # Periodic logging
        if (ex_idx + 1) % 20 == 0:
            for method_name, results in all_results.items():
                if results:
                    acc = sum(r["correct"] for r in results) / len(results) * 100
                    avg_tokens = np.mean([r["n_tokens"] for r in results])
                    logger.info(
                        f"  [{method_name}] {ex_idx + 1} examples: "
                        f"accuracy={acc:.1f}%, avg_tokens={avg_tokens:.0f}"
                    )

    # ── Compile results ──────────────────────────────────────
    summary = {}
    for method_name, results in all_results.items():
        if not results:
            continue
        n = len(results)
        n_correct = sum(r["correct"] for r in results)
        tokens = [r["n_tokens"] for r in results]

        # Aggregate cognitive operation frequencies
        cog_freqs = {}
        for r in results:
            for k, v in r["cognitive_summary"].items():
                cog_freqs.setdefault(k, []).append(v)
        cog_mean = {k: float(np.mean(v)) for k, v in cog_freqs.items()}

        method_summary = {
            "accuracy": n_correct / n if n > 0 else 0,
            "accuracy_pct": f"{n_correct / n * 100:.1f}%" if n > 0 else "N/A",
            "n_correct": n_correct,
            "n_total": n,
            "avg_tokens": float(np.mean(tokens)),
            "median_tokens": float(np.median(tokens)),
            "tokens_per_correct": float(np.sum(tokens) / max(n_correct, 1)),
            "cognitive_frequencies": cog_mean,
        }
        summary[method_name] = method_summary
        logger.info(
            f"\n{method_name}: accuracy={method_summary['accuracy_pct']} "
            f"({n_correct}/{n}), avg_tokens={method_summary['avg_tokens']:.0f}, "
            f"tokens/correct={method_summary['tokens_per_correct']:.0f}"
        )

    # ── Print comparison table ───────────────────────────────
    logger.info(f"\n{'=' * 80}")
    logger.info(f"REASONING IMPROVEMENT RESULTS — {config.benchmark.upper()}")
    logger.info(f"{'=' * 80}")
    logger.info(f"{'Method':<22} {'Accuracy':>10} {'Avg Tokens':>12} {'Tok/Correct':>14}")
    logger.info(f"{'-' * 22} {'-' * 10} {'-' * 12} {'-' * 14}")
    for method_name, s in summary.items():
        logger.info(
            f"{method_name:<22} {s['accuracy_pct']:>10} "
            f"{s['avg_tokens']:>12.0f} {s['tokens_per_correct']:>14.0f}"
        )

    # ── Save everything ──────────────────────────────────────
    with open(save_folder / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(save_folder / "detailed_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nResults saved to {save_folder}")


if __name__ == "__main__":
    main()
