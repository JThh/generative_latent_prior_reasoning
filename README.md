# Learning a Generative Meta-Model of LLM Activations
**Grace Luo, Jiahai Feng, Trevor Darrell, Alec Radford, Jacob Steinhardt**

This repository contains the PyTorch implementation of the paper "Learning a Generative Meta-Model of LLM Activations". The code walks through our proposed method for training an activation diffusion model, and using it for applications like on-manifold steering and scalar probing. We call this model a GLP, or Generative Latent Prior.

[[`Project Page`](https://generative-latent-prior.github.io)][[`arXiv`](https://arxiv.org/abs/2602.06964)]

## Compute
🌟 **TLDR:** Most of the scripts in this README take less than 24GB of VRAM, so they should fit on an Nvidia RTX 4090.

We want everyone to have a chance to try our models out, even in this economy. All of our released GLPs were trained on a billion [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) activations using two Nvidia A100 80GB GPUs (one for activation caching and the other for training), but with some ingenuity you can probably make it work on smaller GPUs too.

## Setup
This code was tested with Python 3.11. To set up the environment, please run:
```
conda env create -f environment.yaml
conda activate glp
pip install vllm==0.9.2 
pip install transformers==4.47.0
pip install -e .
```
You'll need to do the installation in the exact order above, and ignore any pip warnings. We used this exact setup, which was the only way we could get vllm/nnsight/transformers to work together.

## Pre-Trained Weights
You can view all the weights on [our HuggingFace page](https://huggingface.co/generative-latent-prior).

🌟 **TLDR:** For a quickstart, run
```
from glp.denoiser import load_glp
model = load_glp("generative-latent-prior/glp-llama8b-d6", device="cuda:0", checkpoint="final")
```

This grabs our main GLP trained on [Llama8B-Base](https://huggingface.co/meta-llama/Llama-3.1-8B) activations.
| Llama8B | Link |
|-|-|
| glp-llama8b-d6 | [Link](https://huggingface.co/generative-latent-prior/glp-llama8b-d6) |

If you're interested in diving deeper and studying scaling behavior, we also provide [Llama1B-Base](https://huggingface.co/meta-llama/Llama-3.2-1B) GLPs and all intermediate checkpoints.
| Llama1B | Link |
|-|-|
| glp-llama1b-d3 | [Link](https://huggingface.co/generative-latent-prior/glp-llama1b-d3) |
| glp-llama1b-d6 | [Link](https://huggingface.co/generative-latent-prior/glp-llama1b-d6)|
| glp-llama1b-d12 | [Link](https://huggingface.co/generative-latent-prior/glp-llama1b-d12)|
| glp-llama1b-d24 | [Link](https://huggingface.co/generative-latent-prior/glp-llama1b-d24) |
| glp-llama1b-d12-multi | [Link](https://huggingface.co/generative-latent-prior/glp-llama1b-d12-multi) |

Unless otherwise specified, GLPs are trained on the middlemost layer (Layer 15 for Llama8B, Layer 07 for Llama1B). We also provide a multi-layer GLP trained on all Layers 00-15 of Llama1B, called `glp-llama1b-d12-multi`. You can also directly transfer these GLPs, which were trained on Base models, onto Instruct models, as shown in the paper.

*Note:* Each intermediate checkpoint is labeled by "epoch," which corresponds to 1M activations. This means `epoch_1024` was trained on 1024M &asymp; 1B activations (and `final` is the same as `epoch_1024`). 
We use the term "epoch" loosely; in reality we stream data without repetition (so no activation is seen twice).

## Demo
🌟 **TLDR:** For a quickstart, walk through our demo notebook at `glp_demo.ipynb`.

In the demo, we'll walk through loading a GLP, generating activations, then using it for on-manifold steering.

## Applications
- **Scalar 1-D Probing:** Evaluate on the 113 binary classification datasets from [Kantamneni et. al., 2025](https://github.com/JoshEngels/SAE-Probes), by running `python3 glp/script_probe.py`.
- **On-Manifold Steering:** Post-process [Persona Vectors](https://github.com/safety-research/persona_vectors) by following the instructions at `integrations/persona_vectors/README.md`.

*Note:* In the paper, we use the variable `t` to denote the timestep. In the codebase, we follow the [diffusers](https://github.com/huggingface/diffusers) scheduler convention and use `u = 1 - t` instead.

## Training
🌟 **TLDR:** For a quickstart, train a toy Llama1B GLP in a few minutes.
```
# download data
huggingface-cli download generative-latent-prior/llama1b-layer07-fineweb-1M \
    --repo-type dataset  \
    --local-dir data/llama1b-layer07-fineweb-1M \
    --local-dir-use-symlinks False
# launch training
conda activate glp
python3 glp_train.py config=configs/train_llama1b_static.yaml
```

Currently training is pre-set to a small static sanity dataset with 1M activations,
representing the first 1M activations of the full dynamic dataset.
Even on this small dataset, you should see a beautiful loss curve that _just goes down_.
You can also download the [Llama8B sanity dataset](https://huggingface.co/datasets/generative-latent-prior/llama8b-layer15-fineweb-1M). Training on the full one billion activations takes 5.6 days for the Llama8B GLP.

## Reasoning GLP

We extend the GLP framework to reasoning models that generate chain-of-thought (CoT) traces. Reasoning GLPs are trained on residual stream activations captured during CoT generation, enabling three applications: **cognitive probing**, **reasoning steering**, and **reasoning improvement**.

### Supported Reasoning Models

| Model | HuggingFace ID | Params | d_model | Layers | Config |
|-|-|-|-|-|-|
| R1-Distill-Qwen-1.5B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 1.5B | 1,536 | 28 | `train_deepseek_r1_1.5b.yaml` |
| R1-Distill-Qwen-7B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 7B | 3,584 | 28 | `train_deepseek_r1_7b.yaml` |
| R1-Distill-Llama-8B | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | 8B | 4,096 | 32 | `train_deepseek_r1_llama8b.yaml` |
| Qwen3-1.7B | `Qwen/Qwen3-1.7B` | 1.7B | 2,048 | 28 | `train_qwen3_1.7b.yaml` |
| Qwen3-4B | `Qwen/Qwen3-4B` | 4B | 2,560 | 36 | `train_qwen3_4b.yaml` |
| Phi-4-Reasoning | `microsoft/Phi-4-mini-reasoning` | 3.8B | 3,072 | 32 | `train_phi4_reasoning.yaml` |

### Step 1: Cache Reasoning Activations

The original GLP was trained on FineWeb activations from base Llama models. For Reasoning GLP, we cache activations from reasoning models processing **reasoning-triggering datasets**. The primary datasets are:

| Dataset | HuggingFace ID | Size | Purpose |
|-|-|-|-|
| NuminaMath-CoT | `AI-MO/NuminaMath-CoT` | 860K | Primary — diverse math with CoT solutions |
| OpenR1-Math-220k | `open-r1/OpenR1-Math-220k` | 220K | Complement — R1-style reasoning traces |
| MetaMathQA | `meta-math/MetaMathQA` | 395K | Augmented rephrasing of GSM8K/MATH |

Generate multi-trace CoT and extract residual stream activations:

```bash
# Primary: NuminaMath-CoT with 8 traces per prompt (captures correct + incorrect reasoning)
python reasoning/save_reasoning_acts.py \
    model_name=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    dataset=numina_math_cot \
    output_dir=data/deepseek-r1-1.5b-layer14-reasoning \
    num_traces_per_prompt=8 \
    temperature=0.7
```

```bash
# Combined corpus (NuminaMath + OpenR1-Math):
python reasoning/save_reasoning_acts.py \
    dataset=activation_caching \
    max_examples=50000
```

This produces memmap activation files, normalization statistics, per-token cognitive labels (11 behaviours), reasoning phase labels, and per-trace correctness labels.

### Step 2: Train Reasoning GLP

Train a Reasoning GLP using the same `glp_train.py` as the original GLP:

```bash
python glp_train.py config=configs/train_deepseek_r1_1.5b.yaml
```

### Step 3: Reasoning Probing

Probe for cognitive operations (verification, backtracking, error recognition, etc.) using GLP meta-neurons vs linear baselines, including faithfulness analysis:

```bash
python reasoning/script_reasoning_probe.py \
    acts_folder=data/deepseek-r1-1.5b-layer14-reasoning \
    weights_folder=runs/glp-deepseek-r1-1.5b-d6 \
    run_faithfulness=True
```

### Step 4: Reasoning Steering

Steer reasoning behaviour via GLP-guided on-manifold interventions:

```bash
# Static steering (boost verification, suppress overthinking)
python reasoning/script_reasoning_steer.py \
    acts_folder=data/deepseek-r1-1.5b-layer14-reasoning \
    glp_weights_folder=runs/glp-deepseek-r1-1.5b-d6 \
    steer_mode=static

# Adaptive reasoning depth (dynamic intervention based on detected patterns)
python reasoning/script_reasoning_steer.py steer_mode=adaptive_depth

# Error recovery (amplify backtracking when errors detected)
python reasoning/script_reasoning_steer.py steer_mode=error_recovery
```

### Step 5: Benchmark Evaluation

Compare baseline vs direct steering vs GLP-steered accuracy on GSM8K/MATH:

```bash
python reasoning/script_reasoning_improve.py \
    benchmark=gsm8k \
    acts_folder=data/deepseek-r1-1.5b-layer14-reasoning \
    glp_weights_folder=runs/glp-deepseek-r1-1.5b-d6 \
    max_examples=200
```

### Reasoning Behaviour Ontology

The labeler identifies 11 cognitive operations organized hierarchically:

| Category | Behaviour | Description |
|-|-|-|
| Linear | `step_by_step_deduction` | Sequential logical inference |
| Linear | `calculation_execution` | Arithmetic / symbolic computation |
| Non-linear | `verification` | Checking previous results |
| Non-linear | `backtracking` | Abandoning the current approach |
| Non-linear | `strategy_switching` | Adopting a different method |
| Meta-cognitive | `subgoal_formation` | Decomposing into sub-problems |
| Meta-cognitive | `confidence_assessment` | Evaluating certainty |
| Meta-cognitive | `error_recognition` | Detecting mistakes |
| Failure mode | `circular_reasoning` | Repeating without progress |
| Failure mode | `overthinking` | Excessive deliberation |
| Termination | `answer_crystallisation` | Converging on a final answer |

## Roadmap
Currently this codebase is in its initial release. All features marked as complete below are stable and ready to use. The others are still in progress.
- [x] Release pre-trained GLP weights
- [x] Release training code at `glp_train.py`
- [x] Release Persona Vectors steering at `integrations/persona_vectors`
- [x] Release 1-D probing at `glp/script_probe.py` 
- [ ] Release dynamic producer-consumer data pipeline at `glp_save.py`
- [x] Reasoning GLP: activation caching pipeline at `reasoning/save_reasoning_acts.py`
- [x] Reasoning GLP: cognitive probing at `reasoning/script_reasoning_probe.py`
- [x] Reasoning GLP: on-manifold steering at `reasoning/script_reasoning_steer.py`
- [x] Reasoning GLP: benchmark evaluation at `reasoning/script_reasoning_improve.py`

## Citing
```
@article{luo2026glp,
  title={Learning a Generative Meta-Model of LLM Activations},
  author={Grace Luo and Jiahai Feng and Trevor Darrell and Alec Radford and Jacob Steinhardt},
  journal={arXiv preprint arXiv:2602.06964},
  year={2026}
}
```