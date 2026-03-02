"""
On-manifold activation interventions to encourage or suppress specific
reasoning behaviours using GLP-guided steering.

Implements three steering applications from the proposal:
  1. Adaptive reasoning depth — reduce overthinking, boost verification
  2. Error recovery — amplify backtracking when errors are detected
  3. Correctness-conditioned generation — bias toward correct-trace manifold

Mirrors and extends glp/script_steer.py for reasoning-specific use cases.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import einops
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from glp.denoiser import load_glp
from glp.script_steer import (
    postprocess_on_manifold_wrapper,
    addition_intervention,
    generate_with_intervention_wrapper,
)
from reasoning.reasoning_dataset import ReasoningActDataset
from reasoning.label_cot_segments import CognitiveLabeler
from reasoning.benchmarks import format_chat_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================
#   Steering Direction Vectors
# ==============================
def compute_steering_direction(
    dataset: ReasoningActDataset,
    label_name: str,
    max_per_class: int = 2000,
    seed: int = 42,
) -> torch.Tensor:
    """
    Compute a DiffMean steering vector for a cognitive operation:
    mean(positive activations) - mean(negative activations).

    Returns a unit-norm direction vector of shape (hidden_dim,).
    """
    rng = np.random.RandomState(seed)

    pos_acts = []
    neg_acts = []

    for i in range(len(dataset)):
        item = dataset[i]
        if "labels" not in item or label_name not in item["labels"]:
            continue
        act = item["activations"].squeeze(0)
        if item["labels"][label_name]:
            pos_acts.append(act)
        else:
            neg_acts.append(act)

    if len(pos_acts) == 0 or len(neg_acts) == 0:
        raise ValueError(f"No positive or negative samples for label '{label_name}'")

    # Balance and subsample
    n = min(len(pos_acts), len(neg_acts), max_per_class)
    idx_pos = rng.choice(len(pos_acts), n, replace=False)
    idx_neg = rng.choice(len(neg_acts), n, replace=False)

    pos_mean = torch.stack([pos_acts[i] for i in idx_pos]).mean(dim=0)
    neg_mean = torch.stack([neg_acts[i] for i in idx_neg]).mean(dim=0)

    direction = pos_mean - neg_mean
    direction = direction / direction.norm()

    logger.info(
        f"Computed steering direction for '{label_name}': "
        f"{n} samples per class, norm before normalizing = {(pos_mean - neg_mean).norm():.4f}"
    )
    return direction


def compute_all_steering_directions(
    dataset: ReasoningActDataset,
    label_names: list[str] | None = None,
    max_per_class: int = 2000,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Compute DiffMean steering directions for all cognitive operations."""
    if label_names is None:
        label_names = list(dataset.labels.keys()) if dataset.labels else []

    directions = {}
    for name in label_names:
        try:
            directions[name] = compute_steering_direction(
                dataset, name, max_per_class, seed
            )
        except ValueError as e:
            logger.warning(f"Skipping '{name}': {e}")
    return directions


# ==============================
#   Reasoning-Specific Steering
# ==============================
def reasoning_intervention(
    directions: dict[str, torch.Tensor],
    alphas: dict[str, float],
    postprocess_fn=None,
):
    """
    Create a hook function that applies multiple reasoning steering
    directions simultaneously, with optional GLP on-manifold projection.

    Args:
        directions: Dict mapping label names to unit direction vectors.
        alphas: Dict mapping label names to steering coefficients.
            Positive alpha amplifies the behaviour, negative suppresses.
        postprocess_fn: Optional GLP on-manifold projection function.
    """
    if postprocess_fn is None:
        postprocess_fn = lambda x: x

    # Pre-compute combined steering vector
    combined = None
    for name, direction in directions.items():
        alpha = alphas.get(name, 0.0)
        if alpha == 0.0:
            continue
        scaled = alpha * direction
        if combined is None:
            combined = scaled
        else:
            combined = combined + scaled

    if combined is None:
        combined = torch.zeros_like(list(directions.values())[0])

    def rep_act(output, layer_name, inputs):
        use_tuple = isinstance(output, tuple)
        act = output[0] if use_tuple else output
        w = combined.to(device=act.device, dtype=act.dtype)
        # Apply to the last generated token only
        act[:, [-1], :] = postprocess_fn(act[:, [-1], :] + w[None, None, :])
        return (act, *output[1:]) if use_tuple else act

    return rep_act


def adaptive_reasoning_depth_intervention(
    directions: dict[str, torch.Tensor],
    labeler: CognitiveLabeler,
    postprocess_fn=None,
    boost_verification_alpha: float = 3.0,
    suppress_overthinking_alpha: float = -2.0,
    boost_crystallisation_alpha: float = 2.0,
):
    """
    Create a dynamic steering hook that monitors the generated text
    and adaptively adjusts steering based on detected cognitive patterns.

    When overthinking is detected → boost answer_crystallisation
    When low verification → boost verification
    """
    if postprocess_fn is None:
        postprocess_fn = lambda x: x

    generated_tokens = []

    def rep_act(output, layer_name, inputs):
        use_tuple = isinstance(output, tuple)
        act = output[0] if use_tuple else output

        # Detect current cognitive state from recent tokens
        # (we accumulate tokens as they're generated)
        window_text = "".join(generated_tokens[-50:]) if generated_tokens else ""
        window_labels = labeler.label_text(window_text)

        # Determine steering direction dynamically
        steer = torch.zeros(act.shape[-1], device=act.device, dtype=act.dtype)

        if window_labels.get("overthinking", False):
            if "answer_crystallisation" in directions:
                d = directions["answer_crystallisation"].to(act.device, act.dtype)
                steer = steer + boost_crystallisation_alpha * d
            if "overthinking" in directions:
                d = directions["overthinking"].to(act.device, act.dtype)
                steer = steer + suppress_overthinking_alpha * d

        if not window_labels.get("verification", False) and len(generated_tokens) > 100:
            if "verification" in directions:
                d = directions["verification"].to(act.device, act.dtype)
                steer = steer + boost_verification_alpha * d

        if steer.norm() > 0:
            act[:, [-1], :] = postprocess_fn(act[:, [-1], :] + steer[None, None, :])

        return (act, *output[1:]) if use_tuple else act

    return rep_act, generated_tokens


def error_recovery_intervention(
    directions: dict[str, torch.Tensor],
    labeler: CognitiveLabeler,
    postprocess_fn=None,
    backtrack_alpha: float = 4.0,
    strategy_switch_alpha: float = 3.0,
):
    """
    Create a steering hook that triggers error recovery when errors
    or circular reasoning are detected in the generated text.
    """
    if postprocess_fn is None:
        postprocess_fn = lambda x: x

    generated_tokens = []

    def rep_act(output, layer_name, inputs):
        use_tuple = isinstance(output, tuple)
        act = output[0] if use_tuple else output

        window_text = "".join(generated_tokens[-50:]) if generated_tokens else ""
        window_labels = labeler.label_text(window_text)

        steer = torch.zeros(act.shape[-1], device=act.device, dtype=act.dtype)

        if window_labels.get("error_recognition", False) or window_labels.get("circular_reasoning", False):
            if "backtracking" in directions:
                d = directions["backtracking"].to(act.device, act.dtype)
                steer = steer + backtrack_alpha * d
            if "strategy_switching" in directions:
                d = directions["strategy_switching"].to(act.device, act.dtype)
                steer = steer + strategy_switch_alpha * d

        if steer.norm() > 0:
            act[:, [-1], :] = postprocess_fn(act[:, [-1], :] + steer[None, None, :])

        return (act, *output[1:]) if use_tuple else act

    return rep_act, generated_tokens


# ==============================
#   Config and Main
# ==============================
@dataclass
class ReasoningSteerConfig:
    # Reasoning model
    reasoning_model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    torch_dtype: str = "bfloat16"
    device: str = "cuda:0"
    # GLP model
    glp_weights_folder: str = "generative-latent-prior/glp-llama8b-d6"
    glp_ckpt_name: str = "final"
    use_glp_projection: bool = True
    glp_u: float = 0.5
    glp_num_timesteps: int = 20
    # Activation data for computing directions
    acts_folder: str = "data/reasoning_acts"
    # Steering configuration
    steer_mode: str = "static"  # "static", "adaptive_depth", "error_recovery"
    target_layer_prefix: str = "model.layers"
    target_layer_idx: int | None = None  # None = middle layer
    # Static steering alphas (label_name -> coefficient)
    # Positive = amplify, negative = suppress
    steer_alphas: dict = field(default_factory=lambda: {
        "verification": 3.0,
        "overthinking": -2.0,
    })
    # Generation
    max_new_tokens: int = 2048
    # Output
    save_folder: str = "runs/reasoning_steering"
    # Test prompts
    test_prompts: list[str] | None = None


def steering_demo(device="cuda:0"):
    """
    Demonstrate reasoning steering on test prompts.
    Compares unsteered vs GLP-steered generation.
    """
    default_config = OmegaConf.structured(ReasoningSteerConfig)
    OmegaConf.set_struct(default_config, False)
    config = OmegaConf.merge(default_config, OmegaConf.from_cli())

    save_folder = Path(config.save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    # ── Load reasoning model ─────────────────────────────────
    from reasoning.save_reasoning_acts import get_torch_dtype, determine_middle_layer

    dtype = get_torch_dtype(config.torch_dtype)
    logger.info(f"Loading reasoning model: {config.reasoning_model_name}")
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

    # Determine target layer
    target_layer_idx = config.target_layer_idx
    if target_layer_idx is None:
        target_layer_idx = determine_middle_layer(hf_model)
    target_layer = f"{config.target_layer_prefix}.{target_layer_idx}"
    logger.info(f"Steering at layer: {target_layer}")

    # ── Load GLP for on-manifold projection ──────────────────
    postprocess_fn = None
    if config.use_glp_projection:
        glp_model = load_glp(
            config.glp_weights_folder, device=device, checkpoint=config.glp_ckpt_name
        )
        postprocess_fn = postprocess_on_manifold_wrapper(
            glp_model, u=config.glp_u, num_timesteps=config.glp_num_timesteps
        )
        logger.info("GLP on-manifold projection enabled")

    # ── Compute steering directions ──────────────────────────
    dataset = ReasoningActDataset(data_dir=config.acts_folder, load_labels=True)
    directions = compute_all_steering_directions(dataset)
    logger.info(f"Computed {len(directions)} steering directions")

    # ── Set up steering hook ─────────────────────────────────
    labeler = CognitiveLabeler()

    if config.steer_mode == "adaptive_depth":
        hook_fn, gen_tokens = adaptive_reasoning_depth_intervention(
            directions, labeler, postprocess_fn=postprocess_fn
        )
    elif config.steer_mode == "error_recovery":
        hook_fn, gen_tokens = error_recovery_intervention(
            directions, labeler, postprocess_fn=postprocess_fn
        )
    else:
        # Static steering
        alphas_dict = dict(config.steer_alphas)
        filtered_directions = {
            k: v for k, v in directions.items() if k in alphas_dict
        }
        filtered_alphas = {
            k: v for k, v in alphas_dict.items() if k in directions
        }
        hook_fn = reasoning_intervention(
            filtered_directions, filtered_alphas, postprocess_fn=postprocess_fn
        )
        gen_tokens = None

    # ── Generate with and without steering ───────────────────
    test_prompts = config.test_prompts or [
        "What is 23 * 47?",
        "If a train travels at 60 mph for 2.5 hours, how far does it go?",
        "Solve for x: 3x + 7 = 22",
    ]

    generate_fn = generate_with_intervention_wrapper(seed=42)
    results = []

    for prompt in tqdm(test_prompts, desc="Generating"):
        logger.info(f"\nPrompt: {prompt}")

        # Unsteered baseline
        baseline_text = generate_fn(
            prompt, hf_model, tokenizer,
            generate_kwargs={"max_new_tokens": config.max_new_tokens},
        )

        # Steered generation
        steered_text = generate_fn(
            prompt, hf_model, tokenizer,
            generate_kwargs={"max_new_tokens": config.max_new_tokens},
            layers=[target_layer],
            intervention_wrapper=lambda **kw: hook_fn,
        )

        results.append({
            "prompt": prompt,
            "baseline": baseline_text[0] if isinstance(baseline_text, list) else baseline_text,
            "steered": steered_text[0] if isinstance(steered_text, list) else steered_text,
            "steer_mode": config.steer_mode,
        })

        logger.info(f"  Baseline length: {len(results[-1]['baseline'])} chars")
        logger.info(f"  Steered length: {len(results[-1]['steered'])} chars")

    # ── Save results ─────────────────────────────────────────
    with open(save_folder / "steering_results.json", "w") as f:
        json.dump(results, f, indent=2)
    OmegaConf.save(config, save_folder / "config.yaml")

    # Save steering directions for reuse
    torch.save(
        {name: vec.cpu() for name, vec in directions.items()},
        save_folder / "steering_directions.pt",
    )

    logger.info(f"\nResults saved to {save_folder}")


if __name__ == "__main__":
    steering_demo()
