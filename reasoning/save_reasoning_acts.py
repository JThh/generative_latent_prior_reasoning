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
import importlib.util
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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

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

MODEL_CONFIG_FILES = ("config.json",)
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
)
MODEL_WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


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
    model_source: Optional[str] = None
    torch_dtype: str = "bfloat16"
    device: str = "cuda:0"
    cache_dir: Optional[str] = None
    local_files_only: bool = False
    disable_xet: bool = True
    enable_hf_transfer: bool = True
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
    colab_mode: bool = False              # enable Colab-friendly defaults
    flush_every_n_traces: int = 25        # periodic writer flush/checkpoint
    max_saved_tokens: Optional[int] = None
    keep_token_mappings_in_memory: bool = True
    keep_examples_metadata_in_memory: bool = True
    save_trace_metadata_jsonl: bool = True
    # labeling
    enable_labeling: bool = True
    enable_phase_tracking: bool = True
    # correctness conditioning
    track_correctness: bool = True
    # reasoning marker tokens to keep activations for
    reasoning_markers: tuple[str, ...] = ("<think>", "</think>")
    stop_at_reasoning_end_marker: bool = True
    reasoning_end_marker: str = "</think>"


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping.get(dtype_str, torch.bfloat16)


def resolve_device(requested_device: str) -> str:
    """
    Resolve runtime device safely.

    Falls back to CPU if CUDA/MPS is requested but unavailable.
    """
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        fallback = "mps" if mps_available else "cpu"
        logger.warning(
            f"Requested device '{requested_device}' but CUDA is unavailable. "
            f"Falling back to '{fallback}'."
        )
        return fallback

    if requested_device == "mps":
        if not mps_available:
            logger.warning(
                "Requested device 'mps' but MPS is unavailable. Falling back to 'cpu'."
            )
            return "cpu"

    return requested_device


def resolve_dtype_for_device(dtype: torch.dtype, device: str) -> torch.dtype:
    """Adjust dtype for backend limitations."""
    if device == "mps" and dtype == torch.bfloat16:
        logger.warning("MPS + bfloat16 can be unstable; using float16 instead.")
        return torch.float16
    return dtype


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


def find_marker_token_indices(generated_tokens: list[str], markers: list[str]) -> list[int]:
    """
    Return token indices whose decoded token text overlaps any marker string.

    This works even when a tokenizer splits a marker like `<think>` into
    multiple pieces such as `<`, `think`, `>`.
    """
    joined_text = ""
    token_char_ranges = []
    for token in generated_tokens:
        start = len(joined_text)
        joined_text += token
        token_char_ranges.append((start, len(joined_text)))

    if not joined_text:
        return []

    spans = []
    for marker in markers:
        search_start = 0
        while True:
            idx = joined_text.find(marker, search_start)
            if idx == -1:
                break
            spans.append((idx, idx + len(marker)))
            search_start = idx + 1

    if not spans:
        return []

    matches = []
    for tok_idx, (tok_start, tok_end) in enumerate(token_char_ranges):
        if any(tok_start < span_end and tok_end > span_start for span_start, span_end in spans):
            matches.append(tok_idx)
    return sorted(set(matches))


def resolve_model_load_kwargs(config: SaveReasoningActsConfig) -> tuple[dict, Optional[str]]:
    """
    Build `from_pretrained` kwargs and an optional post-load device move target.

    `device_map` is intended for Accelerate sharding strategies (for example
    "auto" or a module->device dict). Passing a raw device string like
    "cuda:0" through `device_map` can trigger a slower path and obscures
    whether the load is blocked on download or placement.
    """
    device = config.device
    if isinstance(device, str) and device in {
        "auto",
        "balanced",
        "balanced_low_0",
        "sequential",
    }:
        return {"device_map": device}, None

    if isinstance(device, dict):
        return {"device_map": device}, None

    return {}, device


def get_model_cache_dir(cache_dir: Optional[str], model_name: str) -> Optional[Path]:
    if not cache_dir:
        return None
    repo_id = model_name.replace("/", "--")
    return Path(cache_dir) / f"models--{repo_id}"


def has_any_file(directory: Path, filenames: tuple[str, ...]) -> bool:
    return any((directory / name).exists() for name in filenames)


def is_local_model_dir(path: Path) -> bool:
    """
    Return True when `path` looks like a self-contained model directory.

    This supports exported model folders, shared artifact directories, and
    ad-hoc local caches without assuming a specific Hugging Face cache layout.
    """
    if not path.is_dir():
        return False

    has_config = has_any_file(path, MODEL_CONFIG_FILES)
    has_tokenizer = has_any_file(path, TOKENIZER_FILES)
    has_weights = has_any_file(path, MODEL_WEIGHT_FILES)
    return has_config and has_tokenizer and has_weights


def get_local_model_dir(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    candidate = Path(path_str).expanduser()
    if is_local_model_dir(candidate):
        return candidate.resolve()
    return None


def detect_stale_incomplete_weights(
    cache_dir: Optional[str],
    model_name: str,
) -> Optional[str]:
    """
    Detect a broken local Hugging Face cache state that causes silent stalls.

    A common failure mode is: tokenizer/config are present, but the model
    weights were interrupted mid-download, leaving only `.incomplete` files and
    a leftover lock. In that case `from_pretrained` can block for a long time
    while retrying or waiting on the lock.
    """
    model_cache_dir = get_model_cache_dir(cache_dir, model_name)
    if model_cache_dir is None or not model_cache_dir.exists():
        return None

    snapshot_dir = model_cache_dir / "snapshots"
    has_weight_file = False
    if snapshot_dir.exists():
        for pattern in (
            "*.safetensors",
            "*.safetensors.index.json",
            "pytorch_model*.bin",
            "pytorch_model*.bin.index.json",
        ):
            if any(snapshot_dir.rglob(pattern)):
                has_weight_file = True
                break

    incomplete_files = sorted((model_cache_dir / "blobs").glob("*.incomplete"))
    lock_dir = Path(cache_dir) / ".locks" / model_cache_dir.name
    lock_files = sorted(lock_dir.glob("*.lock")) if lock_dir.exists() else []

    if has_weight_file or not incomplete_files:
        return None

    details = []
    details.extend(str(path) for path in incomplete_files[:3])
    details.extend(str(path) for path in lock_files[:3])
    detail_str = ", ".join(details)
    return (
        "Detected an incomplete Hugging Face model download with no usable local "
        f"weight file for '{model_name}'. Stale cache artifacts: {detail_str}. "
        "Remove the stale `.incomplete` and `.lock` files, or use a fresh "
        "`cache_dir`, then rerun."
    )


def resolve_model_source(
    config: SaveReasoningActsConfig,
    cache_dir: Optional[str],
) -> tuple[str, Optional[str]]:
    """
    Choose the model/tokenizer source path.

    Resolution order:
      1. explicit `model_source`
      2. `model_name` if it points to a local model directory
      3. `cache_dir` if it points to a local model directory
      4. otherwise the remote Hugging Face model id in `model_name`
    """
    for label, candidate in (
        ("explicit model source", config.model_source),
        ("local model_name path", config.model_name),
        ("local cache directory", cache_dir),
    ):
        local_dir = get_local_model_dir(candidate)
        if local_dir is not None:
            logger.info(f"Using {label}: {local_dir}")
            return str(local_dir), None

    return config.model_name, cache_dir


def configure_hf_downloads(config: SaveReasoningActsConfig) -> Optional[str]:
    """
    Configure Hugging Face download behavior before any model/tokenizer loads.

    Returning a cache dir lets the caller pass the same location through
    `from_pretrained`, keeping behavior explicit and consistent.
    """
    cache_dir = config.cache_dir
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", cache_dir)
        os.environ.setdefault("HF_HUB_CACHE", cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)

    if config.disable_xet:
        # The Xet transport can be much slower on networked filesystems.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    if config.enable_hf_transfer:
        if importlib.util.find_spec("hf_transfer") is not None:
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        else:
            logger.info(
                "hf_transfer is not installed; using the default Hugging Face downloader."
            )

    return cache_dir


def apply_colab_defaults(config: SaveReasoningActsConfig) -> None:
    """
    Apply safer defaults for long runs on Google Colab.
    """
    if not config.colab_mode:
        return

    if str(config.output_dir) == "data/reasoning_acts":
        config.output_dir = "/content/reasoning_acts"
    if config.memmap_file_size == 10_000_000:
        # 1,000,000 fp32 elements ~= 4MB per memmap shard.
        config.memmap_file_size = 1_000_000
    config.keep_token_mappings_in_memory = False
    config.keep_examples_metadata_in_memory = False
    config.save_trace_metadata_jsonl = True

    logger.info(
        "Colab mode enabled: using managed saving defaults "
        f"(output_dir={config.output_dir}, memmap_file_size={config.memmap_file_size}, "
        "streaming trace metadata)."
    )


class StopOnTokenSequence(StoppingCriteria):
    """
    Stop generation once a target token-id sequence appears as a suffix
    of the generated continuation.
    """

    def __init__(self, prompt_length: int, stop_sequence_ids: list[int]):
        self.prompt_length = prompt_length
        self.stop_sequence_ids = stop_sequence_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if not self.stop_sequence_ids:
            return False
        generated = input_ids[0, self.prompt_length :].tolist()
        n = len(self.stop_sequence_ids)
        if len(generated) < n:
            return False
        return generated[-n:] == self.stop_sequence_ids


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
        - generated_ids: list of token ids
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
        if config.stop_at_reasoning_end_marker:
            end_marker_ids = tokenizer.encode(
                config.reasoning_end_marker,
                add_special_tokens=False,
            )
            if end_marker_ids:
                gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                    [StopOnTokenSequence(input_length, end_marker_ids)]
                )
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
        "generated_ids": generated_ids.tolist(),
        "input_length": input_length,
    }


def main():
    config_base = OmegaConf.structured(SaveReasoningActsConfig())
    OmegaConf.set_struct(config_base, False)
    config_cli = OmegaConf.from_cli()
    config_path = config_cli.pop("config", None)
    config_file = OmegaConf.load(config_path) if config_path else OmegaConf.create()
    config = OmegaConf.merge(config_base, config_file, config_cli)
    apply_colab_defaults(config)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Config: {config}")
    logger.info(f"Output directory: {output_dir}")

    # ── Load model ──────────────────────────────────────────────
    config.device = resolve_device(str(config.device))
    dtype = resolve_dtype_for_device(get_torch_dtype(config.torch_dtype), config.device)
    cache_dir = configure_hf_downloads(config)
    logger.info(f"Loading model: {config.model_name}")
    if cache_dir:
        logger.info(f"Using Hugging Face cache: {cache_dir}")
    if config.local_files_only:
        logger.info("Loading from local cache only; remote download is disabled.")
    model_source, load_cache_dir = resolve_model_source(config, cache_dir)
    logger.info("Loading tokenizer...")
    load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=True,
        cache_dir=load_cache_dir,
        local_files_only=config.local_files_only,
    )
    logger.info(f"Tokenizer loaded in {time.time() - load_start:.1f}s")

    model_load_kwargs, model_move_device = resolve_model_load_kwargs(config)
    stale_cache_msg = None
    if model_source == config.model_name:
        stale_cache_msg = detect_stale_incomplete_weights(cache_dir, config.model_name)
    if stale_cache_msg:
        raise RuntimeError(stale_cache_msg)

    logger.info(
        "Loading model weights..."
        + (
            f" (device_map={model_load_kwargs['device_map']})"
            if "device_map" in model_load_kwargs
            else " (CPU load, then explicit device move)"
        )
    )
    load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=dtype,
        trust_remote_code=True,
        cache_dir=load_cache_dir,
        local_files_only=config.local_files_only,
        low_cpu_mem_usage=True,
        **model_load_kwargs,
    )
    logger.info(f"Model weights loaded in {time.time() - load_start:.1f}s")

    if model_move_device is not None:
        logger.info(f"Moving model to device: {model_move_device}")
        move_start = time.time()
        model = model.to(model_move_device)
        logger.info(f"Model moved to {model_move_device} in {time.time() - move_start:.1f}s")
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # For now, always save a single middle layer to keep the dataset compact.
    layer_idx = determine_middle_layer(model)
    logger.info(f"Using middle layer only: {layer_idx}")
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
    token_to_example = [] if config.keep_token_mappings_in_memory else None
    token_to_trace = [] if config.keep_token_mappings_in_memory else None  # which trace index within an example
    all_metadata = [] if config.keep_examples_metadata_in_memory else None
    trace_metadata_path = output_dir / "trace_metadata.jsonl"
    if config.save_trace_metadata_jsonl:
        trace_metadata_path.write_text("")

    # Running statistics
    running_sum = torch.zeros(hidden_dim, dtype=torch.float64)
    running_sq_sum = torch.zeros(hidden_dim, dtype=torch.float64)
    total_tokens = 0
    traces_processed = 0
    stop_due_to_token_cap = False

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

            marker_indices = find_marker_token_indices(
                result["generated_tokens"],
                list(config.reasoning_markers),
            )
            if not marker_indices:
                continue

            acts = [result["activations"][idx] for idx in marker_indices]
            saved_tokens = [result["generated_tokens"][idx] for idx in marker_indices]
            if len(acts) == 0:
                continue

            if config.max_saved_tokens is not None:
                remaining = config.max_saved_tokens - total_tokens
                if remaining <= 0:
                    stop_due_to_token_cap = True
                    break
                if len(acts) > remaining:
                    acts = acts[:remaining]
                    saved_tokens = saved_tokens[:remaining]
                    marker_indices = marker_indices[:remaining]

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
                token_labels = labeler.label_tokens(saved_tokens)
                for name in labeler.label_names:
                    all_cognitive_labels[name].extend(token_labels[name].tolist())

            # Phase labels
            if config.enable_phase_tracking:
                phases = assign_all_phases(n_act, saved_tokens)
                all_phase_labels.extend([PHASE_TO_IDX.get(p, 2) for p in phases])

            # Correctness per token (same for all tokens in a trace)
            all_correctness.extend([trace_correct] * n_act)

            # Track mappings
            if token_to_example is not None:
                token_to_example.extend([ex_idx] * n_act)
            if token_to_trace is not None:
                token_to_trace.extend([trace_idx] * n_act)

            # Per-trace metadata
            trace_metadata = {
                "example_idx": ex_idx,
                "trace_idx": trace_idx,
                "question": question[:200],
                "n_generated_tokens": len(result["generated_tokens"]),
                "n_reasoning_marker_tokens": n_act,
                "n_saved_tokens": n_act,
                "saved_token_indices": marker_indices,
                "saved_tokens": saved_tokens,
                "generated_text_preview": result["generated_text"][:300],
                "correct": trace_correct,
            }
            if all_metadata is not None:
                all_metadata.append(trace_metadata)
            if config.save_trace_metadata_jsonl:
                with open(trace_metadata_path, "a") as f:
                    f.write(json.dumps(trace_metadata) + "\n")

            traces_processed += 1
            if config.flush_every_n_traces > 0 and traces_processed % config.flush_every_n_traces == 0:
                writer.flush()
                with open(output_dir / "progress.json", "w") as f:
                    json.dump(
                        {
                            "examples_seen": ex_idx + 1,
                            "traces_processed": traces_processed,
                            "total_saved_tokens": total_tokens,
                        },
                        f,
                        indent=2,
                    )

            if config.max_saved_tokens is not None and total_tokens >= config.max_saved_tokens:
                stop_due_to_token_cap = True
                break

        # Periodic progress
        if (ex_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"Processed {ex_idx + 1}/{len(examples)} examples, "
                f"{total_tokens} tokens, {elapsed:.1f}s"
            )
        if stop_due_to_token_cap:
            logger.info(
                f"Reached max_saved_tokens={config.max_saved_tokens}. "
                "Stopping early by design."
            )
            break

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
        "n_traces_total": traces_processed,
        "total_tokens": total_tokens,
        "saved_token_selector": "thinking_markers_only",
        "max_new_tokens": config.max_new_tokens,
        "max_saved_tokens": config.max_saved_tokens,
        "token_to_example": token_to_example if token_to_example is not None else [],
        "token_to_trace": token_to_trace if token_to_trace is not None else [],
        "examples": all_metadata if all_metadata is not None else [],
        "token_mappings_streamed": token_to_example is None,
        "trace_metadata_jsonl": str(trace_metadata_path) if config.save_trace_metadata_jsonl else None,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    OmegaConf.save(config, output_dir / "config.yaml")

    elapsed = time.time() - start_time
    logger.info(
        f"Done! {total_tokens} activations from "
        f"{traces_processed} traces ({len(examples)} examples) "
        f"saved to {output_dir} in {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
