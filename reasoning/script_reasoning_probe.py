"""
Probing for cognitive operations in reasoning models using GLP meta-neurons.

Compares GLP meta-neuron probing against linear probing baselines across
the full Reasoning Behaviour Ontology (11 behaviour types), and includes
faithfulness probing — testing whether internal meta-neuron activations
match the verbalised cognitive operations.

Directly mirrors the structure of glp/script_probe.py.
"""

import glob
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass

import einops
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import bootstrap
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from tqdm import tqdm

from glp.denoiser import load_glp
from glp.script_probe import (
    run_sklearn_logreg,
    run_sklearn_logreg_batched,
    prefilter_and_reshape_to_oned,
    get_meta_neurons_wrapper,
    get_meta_neurons_layer_time,
    get_meta_neurons_locations,
    compile_probe_results,
)
from reasoning.reasoning_dataset import (
    ReasoningActDataset,
    get_label_balanced_split,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================
#   Reasoning Probing Config
# ==============================
@dataclass
class ReasoningProbingConfig:
    save_folder: str = "runs/reasoning_probing"
    acts_folder: str = "data/reasoning_acts"
    weights_folder: str = "generative-latent-prior/glp-llama8b-d6"
    ckpt_name: str = "final"
    # Probing parameters
    u: float = 0.9
    topk: int = 512
    seed: int = 42
    batch_size: int | None = None
    max_per_class: int = 2000
    train_ratio: float = 0.8
    # Which labels to probe
    labels: list[str] | None = None  # None = all available
    # Baseline comparison
    run_linear_baseline: bool = True
    run_glp_probing: bool = True
    # Faithfulness probing
    run_faithfulness: bool = True


# ==============================
#   Linear Baseline Probing
# ==============================
def probe_linear_baseline(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    seed: int = 42,
) -> dict:
    """
    Run linear logistic regression probe on raw activations.
    Returns dict with val_auc and test_auc.
    """
    X_train_np = X_train.detach().cpu().numpy()
    X_test_np = X_test.detach().cpu().numpy()
    y_train_np = y_train.detach().cpu().numpy()
    y_test_np = y_test.detach().cpu().numpy()

    metrics = run_sklearn_logreg(
        X_train_np, y_train_np,
        X_test_np, y_test_np,
        seed=seed,
    )
    return metrics


# ==============================
#   GLP Meta-Neuron Probing
# ==============================
def probe_glp_meta_neurons(
    model,
    device: str,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    u: float = 0.9,
    topk: int = 512,
    seed: int = 42,
    batch_size: int | None = None,
) -> dict:
    """
    Extract GLP meta-neuron features and run 1-D scalar probing.
    Returns dict with val_aucs, test_aucs, and best location info.
    """
    u_tensor = torch.tensor([u])[:, None]
    layers = get_meta_neurons_locations(model)

    # Get meta-neuron features
    X_train_diffusion, (layer_size, u_size) = get_meta_neurons_layer_time(
        model, device, X_train, u_tensor, layers, seed, batch_size
    )
    X_test_diffusion, _ = get_meta_neurons_layer_time(
        model, device, X_test, u_tensor, layers, seed, batch_size
    )

    # Prefilter and reshape for 1-D probing
    X_train_filtered, X_test_filtered, top_batch_idxs = prefilter_and_reshape_to_oned(
        X_train_diffusion, X_test_diffusion, y_train, device, topk=topk
    )

    # Run logistic regression on each meta-neuron
    val_aucs, test_aucs = run_sklearn_logreg_batched(
        X_train_filtered, y_train, X_test_filtered, y_test, device=device
    )

    format_aucs = lambda aucs: {
        idx: auc.item() for idx, auc in zip(top_batch_idxs, aucs)
    }

    return {
        "val_aucs": format_aucs(val_aucs),
        "test_aucs": format_aucs(test_aucs),
        "layers": layers,
        "u": [u],
        "best_val_auc": float(val_aucs.max()),
        "best_test_auc": float(test_aucs[val_aucs.argmax()]),
    }


# ==============================
#   Faithfulness Probing
# ==============================
def probe_faithfulness(
    model,
    device: str,
    dataset: ReasoningActDataset,
    label_name: str,
    u: float = 0.9,
    seed: int = 42,
    batch_size: int | None = None,
    max_samples: int = 1000,
) -> dict:
    """
    Measure alignment between GLP meta-neuron activations and
    verbalised cognitive operations.

    For tokens where the CoT text claims a cognitive operation
    (e.g., "let me verify") vs. tokens where it does NOT claim it,
    measure whether the corresponding GLP meta-neuron activates
    differently. High alignment = faithful; low alignment = unfaithful
    (model says one thing but internal computation differs).
    """
    # Get balanced data
    X_train, y_train, X_test, y_test = get_label_balanced_split(
        dataset, label_name,
        max_per_class=max_samples // 2,
        seed=seed,
    )

    if len(X_train) == 0 or len(X_test) == 0:
        return {"faithfulness_auc": None, "n_samples": 0}

    # GLP meta-neuron response
    glp_results = probe_glp_meta_neurons(
        model, device, X_train, y_train, X_test, y_test,
        u=u, topk=256, seed=seed, batch_size=batch_size,
    )

    # Linear baseline response
    linear_results = probe_linear_baseline(
        X_train, y_train, X_test, y_test, seed=seed,
    )

    return {
        "faithfulness_auc_glp": glp_results["best_test_auc"],
        "faithfulness_auc_linear": linear_results["test_auc"],
        "gap": glp_results["best_test_auc"] - linear_results["test_auc"],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "interpretation": (
            "HIGH alignment" if glp_results["best_test_auc"] > 0.8
            else "MODERATE alignment" if glp_results["best_test_auc"] > 0.65
            else "LOW alignment (possible unfaithful reasoning)"
        ),
    }


# ==============================
#   Main Function
# ==============================
def reasoning_probing(device="cuda:0"):
    default_config = OmegaConf.structured(ReasoningProbingConfig)
    OmegaConf.set_struct(default_config, False)
    config = OmegaConf.merge(default_config, OmegaConf.from_cli())

    save_folder = config.save_folder
    os.makedirs(save_folder, exist_ok=True)

    # Load dataset
    dataset = ReasoningActDataset(
        data_dir=config.acts_folder,
        load_labels=True,
    )
    logger.info(f"Loaded dataset with {len(dataset)} tokens")

    # Determine which labels to probe
    if config.labels is not None:
        label_names = list(config.labels)
    elif dataset.labels is not None:
        label_names = list(dataset.labels.keys())
    else:
        raise ValueError("No cognitive labels found in dataset")
    logger.info(f"Probing {len(label_names)} cognitive operations: {label_names}")

    # Load GLP model
    model = None
    if config.run_glp_probing or config.run_faithfulness:
        model = load_glp(
            config.weights_folder, device=device, checkpoint=config.ckpt_name
        )
        weights_name = os.path.basename(config.weights_folder)
    else:
        weights_name = "baseline"

    # ── Probe each cognitive operation ──────────────────────────
    all_results = {}

    for label_name in tqdm(label_names, desc="Probing cognitive operations"):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Probing: {label_name}")
        logger.info(f"{'=' * 60}")

        # Get balanced train/test split
        try:
            X_train, y_train, X_test, y_test = get_label_balanced_split(
                dataset, label_name,
                train_ratio=config.train_ratio,
                max_per_class=config.max_per_class,
                seed=config.seed,
            )
        except Exception as e:
            logger.warning(f"Could not create split for '{label_name}': {e}")
            continue

        if len(X_train) < 10 or len(X_test) < 10:
            logger.warning(f"Too few samples for '{label_name}', skipping")
            continue

        label_results = {"label_name": label_name}

        # Linear baseline
        if config.run_linear_baseline:
            linear_metrics = probe_linear_baseline(
                X_train, y_train, X_test, y_test, seed=config.seed
            )
            label_results["linear_baseline"] = linear_metrics
            logger.info(f"  Linear baseline: test AUC = {linear_metrics['test_auc']:.4f}")

        # GLP meta-neurons
        if config.run_glp_probing and model is not None:
            glp_metrics = probe_glp_meta_neurons(
                model, device, X_train, y_train, X_test, y_test,
                u=config.u, topk=config.topk, seed=config.seed,
                batch_size=config.batch_size,
            )
            label_results["glp_meta_neurons"] = {
                "best_val_auc": glp_metrics["best_val_auc"],
                "best_test_auc": glp_metrics["best_test_auc"],
                "u": config.u,
                "topk": config.topk,
            }
            logger.info(f"  GLP meta-neurons: test AUC = {glp_metrics['best_test_auc']:.4f}")

            if config.run_linear_baseline:
                improvement = glp_metrics["best_test_auc"] - linear_metrics["test_auc"]
                label_results["improvement_over_linear"] = improvement
                logger.info(f"  Improvement over linear: {improvement:+.4f}")

        # Faithfulness probing
        if config.run_faithfulness and model is not None:
            faithfulness = probe_faithfulness(
                model, device, dataset, label_name,
                u=config.u, seed=config.seed, batch_size=config.batch_size,
            )
            label_results["faithfulness"] = faithfulness
            logger.info(f"  Faithfulness: {faithfulness['interpretation']}")

        all_results[label_name] = label_results

        # Save per-label results
        label_save_path = f"{save_folder}/{label_name}/{weights_name}/{config.ckpt_name}.json"
        os.makedirs(os.path.dirname(label_save_path), exist_ok=True)
        with open(label_save_path, "w") as f:
            json.dump(label_results, f, indent=2)

    # ── Compile summary ────────────────────────────────────────
    summary = {
        "config": OmegaConf.to_container(config, resolve=True),
        "results": all_results,
    }

    summary_path = f"{save_folder}/reasoning_probing_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    logger.info(f"\n{'=' * 80}")
    logger.info(f"REASONING PROBING RESULTS SUMMARY")
    logger.info(f"{'=' * 80}")
    logger.info(f"{'Label':<25} {'Linear AUC':>12} {'GLP AUC':>12} {'Delta':>10} {'Faithful':>12}")
    logger.info(f"{'-' * 25} {'-' * 12} {'-' * 12} {'-' * 10} {'-' * 12}")

    for name, res in all_results.items():
        linear_auc = res.get("linear_baseline", {}).get("test_auc", float("nan"))
        glp_auc = res.get("glp_meta_neurons", {}).get("best_test_auc", float("nan"))
        delta = res.get("improvement_over_linear", float("nan"))
        faith = res.get("faithfulness", {}).get("faithfulness_auc_glp", float("nan"))
        logger.info(f"{name:<25} {linear_auc:>12.4f} {glp_auc:>12.4f} {delta:>+10.4f} {faith:>12.4f}")

    logger.info(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    reasoning_probing()
