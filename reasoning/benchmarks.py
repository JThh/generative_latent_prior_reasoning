"""
Utilities for loading and parsing reasoning benchmarks for activation caching.

The original GLP used FineWeb (general web text) to generate activations from
base LLMs. For Reasoning GLP, we need datasets that trigger chain-of-thought
reasoning: mathematical problem-solving, coding, and logical reasoning.

Supported datasets (post-training / activation caching):
  - NuminaMath-CoT: 860K math competition problems with CoT (AI-MO/NuminaMath-CoT)
  - OpenR1-Math-220k: R1-style reasoning traces (open-r1/OpenR1-Math-220k)
  - MetaMathQA: 395K augmented math reasoning (meta-math/MetaMathQA)

Supported datasets (evaluation benchmarks):
  - GSM8K: Grade-school math (openai/gsm8k)
  - MATH: Competition mathematics (hendrycks/competition_math)
  - AIME: American Invitational Mathematics Examination
  - ARC-Challenge: AI2 Reasoning Challenge (allenai/ai2_arc)
  - HumanEval: Coding problems (openai/openai_humaneval)
  - MBPP: Python programming problems (google-research-datasets/mbpp)
"""

import math
import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ==========================================================
#     Dataset Loading — Math
# ==========================================================
def load_gsm8k(
    split: str = "test",
    max_examples: Optional[int] = None,
) -> list[dict]:
    """
    Load GSM8K (grade-school math) from HuggingFace.

    Returns list of dicts with: question, answer_text, numeric_answer.
    ~7.5K train, ~1.3K test examples.
    """
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    examples = []
    for row in ds:
        question = row["question"]
        answer_text = row["answer"]
        # GSM8K answers end with "#### <number>"
        numeric_str = answer_text.split("####")[-1].strip().replace(",", "")
        try:
            numeric_answer = float(numeric_str)
        except ValueError:
            numeric_answer = None

        examples.append({
            "question": question,
            "answer_text": answer_text,
            "numeric_answer": numeric_answer,
            "source": "gsm8k",
        })

    logger.info(f"Loaded {len(examples)} GSM8K examples (split={split})")
    return examples


def load_math(
    split: str = "test",
    max_examples: Optional[int] = None,
    difficulty: Optional[list[int]] = None,
) -> list[dict]:
    """
    Load MATH competition dataset from HuggingFace.

    Returns list of dicts with: question, answer_text, answer, subject, level.
    ~7.5K train, ~5K test examples across 7 subjects and 5 difficulty levels.
    """
    from datasets import load_dataset

    ds = load_dataset("hendrycks/competition_math", split=split)

    examples = []
    for row in ds:
        level_match = re.search(r"(\d+)", str(row.get("level", "")))
        level = int(level_match.group(1)) if level_match else None

        if difficulty is not None and level not in difficulty:
            continue

        solution = row.get("solution", "")
        answer = extract_boxed(solution) or row.get("answer", "")

        examples.append({
            "question": row["problem"],
            "answer_text": solution,
            "answer": answer,
            "subject": row.get("type", "unknown"),
            "level": level,
            "source": "math",
        })

        if max_examples is not None and len(examples) >= max_examples:
            break

    logger.info(f"Loaded {len(examples)} MATH examples (split={split})")
    return examples


def load_aime(
    max_examples: Optional[int] = None,
) -> list[dict]:
    """
    Load AIME (American Invitational Mathematics Examination) problems.

    Uses the AI-MO/aimo-validation-aime dataset on HuggingFace.
    These are competition-level problems requiring extended reasoning.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    except Exception:
        logger.warning(
            "Could not load AIME dataset from AI-MO/aimo-validation-aime. "
            "Falling back to empty list. Install or check dataset availability."
        )
        return []

    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    examples = []
    for row in ds:
        question = row.get("problem", row.get("question", ""))
        answer = str(row.get("answer", ""))

        examples.append({
            "question": question,
            "answer": answer,
            "numeric_answer": _try_float(answer),
            "source": "aime",
        })

    logger.info(f"Loaded {len(examples)} AIME examples")
    return examples


def load_metamathqa(
    split: str = "train",
    max_examples: Optional[int] = None,
) -> list[dict]:
    """
    Load MetaMathQA — augmented math reasoning dataset with diverse
    rephrasing of GSM8K and MATH problems. 395K examples.

    Excellent for activation caching because it provides many
    reasoning-triggering prompts across difficulty levels.
    """
    from datasets import load_dataset

    ds = load_dataset("meta-math/MetaMathQA", split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    examples = []
    for row in ds:
        question = row.get("query", row.get("question", ""))
        answer_text = row.get("response", row.get("answer", ""))

        # Try to extract numeric answer
        numeric_answer = None
        if "####" in answer_text:
            try:
                numeric_answer = float(answer_text.split("####")[-1].strip().replace(",", ""))
            except ValueError:
                pass
        if numeric_answer is None:
            boxed = extract_boxed(answer_text)
            if boxed:
                numeric_answer = _try_float(boxed)

        examples.append({
            "question": question,
            "answer_text": answer_text,
            "numeric_answer": numeric_answer,
            "answer": str(numeric_answer) if numeric_answer is not None else extract_boxed(answer_text),
            "source": "metamathqa",
        })

    logger.info(f"Loaded {len(examples)} MetaMathQA examples (split={split})")
    return examples


# ==========================================================
#     Dataset Loading — Coding
# ==========================================================
def load_humaneval(
    max_examples: Optional[int] = None,
) -> list[dict]:
    """
    Load HumanEval coding benchmark from HuggingFace. 164 Python problems.

    Returns dicts with: question (prompt), answer (canonical_solution), task_id.
    """
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    examples = []
    for row in ds:
        examples.append({
            "question": row["prompt"],
            "answer": row.get("canonical_solution", ""),
            "task_id": row.get("task_id", ""),
            "entry_point": row.get("entry_point", ""),
            "test": row.get("test", ""),
            "source": "humaneval",
        })

    logger.info(f"Loaded {len(examples)} HumanEval examples")
    return examples


def load_mbpp(
    split: str = "test",
    max_examples: Optional[int] = None,
) -> list[dict]:
    """
    Load MBPP (Mostly Basic Python Problems) from HuggingFace.
    ~374 train, ~500 test problems.
    """
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    examples = []
    for row in ds:
        examples.append({
            "question": row["text"],
            "answer": row.get("code", ""),
            "task_id": row.get("task_id", ""),
            "source": "mbpp",
        })

    logger.info(f"Loaded {len(examples)} MBPP examples (split={split})")
    return examples


# ==========================================================
#     Dataset Loading — Logic / Reasoning
# ==========================================================
def load_arc_challenge(
    split: str = "test",
    max_examples: Optional[int] = None,
) -> list[dict]:
    """
    Load ARC-Challenge (AI2 Reasoning Challenge) from HuggingFace.
    Multiple-choice science questions requiring reasoning. ~1.2K test.
    """
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    examples = []
    for row in ds:
        choices = row["choices"]
        choices_text = "\n".join(
            f"  ({label}) {text}"
            for label, text in zip(choices["label"], choices["text"])
        )
        question = f"{row['question']}\n{choices_text}"
        answer = row.get("answerKey", "")

        examples.append({
            "question": question,
            "answer": answer,
            "source": "arc_challenge",
        })

    logger.info(f"Loaded {len(examples)} ARC-Challenge examples (split={split})")
    return examples


# ==========================================================
#     Dataset Loading — Post-Training / Activation Caching
# ==========================================================
def load_numina_math_cot(
    split: str = "train",
    max_examples: Optional[int] = None,
) -> list[dict]:
    """
    Load NuminaMath-CoT — 860K math competition problems with
    Chain-of-Thought solutions from the AI-MO/NuminaMath-CoT dataset.

    This is one of the largest public math reasoning datasets, covering
    problems from Chinese high school exams, AMC/AIME, Olympiads, and
    Art of Problem Solving (AoPS). Each problem has a detailed step-by-step
    CoT solution — ideal for generating diverse reasoning activations.

    Recommended as the PRIMARY dataset for activation caching because:
      - 860K problems = abundant diverse reasoning traces
      - Covers easy → competition difficulty spectrum
      - CoT solutions provide ground-truth reasoning structure
      - Problems are similar to what reasoning models were post-trained on
    """
    from datasets import load_dataset

    ds = load_dataset("AI-MO/NuminaMath-CoT", split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    examples = []
    for row in ds:
        # NuminaMath-CoT has 'problem' and 'solution' fields
        question = row.get("problem", row.get("question", ""))
        solution = row.get("solution", row.get("answer", ""))

        # Extract answer from solution
        answer = extract_boxed(solution)
        numeric_answer = _try_float(answer) if answer else None

        examples.append({
            "question": question,
            "answer_text": solution,
            "answer": answer,
            "numeric_answer": numeric_answer,
            "source": "numina_math_cot",
        })

    logger.info(f"Loaded {len(examples)} NuminaMath-CoT examples (split={split})")
    return examples


def load_openr1_math(
    split: str = "train",
    max_examples: Optional[int] = None,
) -> list[dict]:
    """
    Load OpenR1-Math-220k — DeepSeek-R1-style reasoning traces
    from the open-r1 replication effort.

    220K math problems with extended reasoning traces generated
    by R1-style models. Contains the kind of reasoning traces
    (with <think>...</think> blocks) that our target models produce.

    Recommended as a COMPLEMENT to NuminaMath because:
      - Reasoning traces match the style of DeepSeek-R1 distillations
      - Contains verification, backtracking, and strategy-switching patterns
      - Problems were curated for RL-based reasoning training
    """
    from datasets import load_dataset

    ds = load_dataset("open-r1/OpenR1-Math-220k", split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    examples = []
    for row in ds:
        # OpenR1-Math has 'problem' and 'solution'/'generated_solution' fields
        question = row.get("problem", row.get("question", ""))
        solution = row.get("solution", row.get("generated_solution", ""))
        answer = row.get("answer", "") or extract_boxed(solution) or ""

        examples.append({
            "question": question,
            "answer_text": solution,
            "answer": answer,
            "numeric_answer": _try_float(answer) if answer else None,
            "source": "openr1_math",
        })

    logger.info(f"Loaded {len(examples)} OpenR1-Math examples (split={split})")
    return examples


# ==========================================================
#     Unified Dataset Loader
# ==========================================================

# Registry of all supported datasets
DATASET_REGISTRY = {
    # Post-training / activation caching datasets (large, diverse)
    "numina_math_cot": load_numina_math_cot,
    "openr1_math": load_openr1_math,
    "metamathqa": load_metamathqa,
    # Evaluation benchmarks
    "gsm8k": load_gsm8k,
    "math": load_math,
    "aime": load_aime,
    "humaneval": load_humaneval,
    "mbpp": load_mbpp,
    "arc_challenge": load_arc_challenge,
}

# Recommended datasets for reasoning activation caching, by category
# The 'activation_caching' category contains the primary datasets for
# generating the activation dataset that the Reasoning GLP is trained on.
REASONING_CORPUS = {
    "activation_caching": ["numina_math_cot", "openr1_math"],
    "math_easy": ["gsm8k"],
    "math_competition": ["math", "aime"],
    "math_augmented": ["metamathqa"],
    "coding": ["humaneval", "mbpp"],
    "logic": ["arc_challenge"],
}


def load_reasoning_corpus(
    datasets: list[str] | str = "all",
    max_examples_per_dataset: Optional[int] = None,
    split: str = "train",
) -> list[dict]:
    """
    Load a combined reasoning task corpus for activation caching.

    Args:
        datasets: List of dataset names, a category name from
            REASONING_CORPUS, or "all" for the full corpus.
        max_examples_per_dataset: Cap per dataset (None = all).
        split: Dataset split to use (where applicable).

    Returns:
        Combined list of examples from all requested datasets.
    """
    if datasets == "all":
        dataset_names = list(DATASET_REGISTRY.keys())
    elif isinstance(datasets, str) and datasets in REASONING_CORPUS:
        dataset_names = REASONING_CORPUS[datasets]
    elif isinstance(datasets, str):
        dataset_names = [datasets]
    else:
        dataset_names = list(datasets)

    all_examples = []
    for name in dataset_names:
        if name not in DATASET_REGISTRY:
            logger.warning(f"Unknown dataset '{name}', skipping")
            continue

        loader = DATASET_REGISTRY[name]
        try:
            # Pass split if the loader accepts it
            import inspect
            sig = inspect.signature(loader)
            kwargs = {}
            if "split" in sig.parameters:
                kwargs["split"] = split
            if "max_examples" in sig.parameters:
                kwargs["max_examples"] = max_examples_per_dataset
            examples = loader(**kwargs)
            all_examples.extend(examples)
        except Exception as e:
            logger.warning(f"Error loading '{name}': {e}")

    logger.info(
        f"Loaded {len(all_examples)} total examples from "
        f"{len(dataset_names)} datasets: {dataset_names}"
    )
    return all_examples


# ==========================================================
#     Answer Extraction
# ==========================================================
def extract_boxed(text: str) -> Optional[str]:
    """
    Extract content from \\boxed{...} in LaTeX text.
    Handles nested braces correctly.
    """
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None

    start = idx + len("\\boxed{")
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1

    if depth == 0:
        return text[start:pos - 1].strip()
    return None


def extract_final_answer(cot_text: str) -> Optional[str]:
    """
    Extract the final numeric or symbolic answer from a generated CoT trace.

    Tries multiple extraction patterns in order of specificity:
    1. \\boxed{...} notation
    2. "The answer is X" / "the final answer is X"
    3. "#### X" (GSM8K format)
    4. Last number in the text
    """
    # 1. Boxed notation
    boxed = extract_boxed(cot_text)
    if boxed is not None:
        return boxed

    # 2. "the answer is" patterns
    patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\s]*([^\n]+)",
        r"(?:therefore|thus|hence|so)\s*,?\s*(?:the\s+)?(?:answer\s+is\s*)?[:\s]*([^\n]+)",
    ]
    for pat in patterns:
        match = re.search(pat, cot_text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            answer = re.split(r"(?:(?:\s+with)|(?:\s+because)|(?:\s+where))\b", answer, maxsplit=1)[0]
            answer = answer.rstrip(" .,;:!?)")
            return answer

    # 3. #### notation
    if "####" in cot_text:
        answer = cot_text.split("####")[-1].strip()
        if answer:
            return answer

    # 4. Last number
    numbers = re.findall(r"\-?\d[\d,]*\.?\d*", cot_text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def normalize_answer(answer: str) -> Optional[float]:
    """Normalize an answer string to float for comparison."""
    if answer is None:
        return None

    answer = str(answer).strip()

    # Remove common wrappers and formatting noise.
    answer = answer.replace("$", "").replace(",", "")
    answer = re.sub(r"\\left|\\right", "", answer)
    answer = answer.strip().strip(" \t\n\r.,;:!?)(")

    # Handle `x = ...` / `y=...` / `answer = ...`.
    eq_match = re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*(.+)$", answer)
    if eq_match:
        answer = eq_match.group(1).strip()

    # Handle boxed expressions if still present.
    boxed = extract_boxed(answer)
    if boxed is not None:
        answer = boxed.strip()

    # Handle LaTeX fractions like \frac{3}{4}.
    frac_latex_match = re.match(
        r"^\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}$",
        answer,
    )
    if frac_latex_match:
        num = float(frac_latex_match.group(1))
        den = float(frac_latex_match.group(2))
        if den == 0:
            return None
        return num / den

    answer = answer.replace(" ", "")

    # Handle fractions
    frac_match = re.match(r"^(-?\d+)\s*/\s*(\d+)$", answer)
    if frac_match:
        num, den = float(frac_match.group(1)), float(frac_match.group(2))
        return num / den if den != 0 else None

    # Handle percentages explicitly.
    pct_match = re.match(r"^(-?\d+(?:\.\d+)?)%$", answer)
    if pct_match:
        return float(pct_match.group(1)) / 100.0

    try:
        val = float(answer)
        if not math.isfinite(val):
            return None
        return val
    except ValueError:
        return None


def canonicalize_answer_text(answer: str) -> str:
    """
    Canonicalize symbolic/text answers for robust string matching.
    """
    answer = str(answer).strip().lower()
    answer = answer.replace("$", "")
    answer = re.sub(r"\\left|\\right", "", answer)
    answer = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", answer)
    answer = re.sub(r"^[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*", "", answer)  # drop `x =`
    answer = re.sub(r"^\s*(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\-]?\s*", "", answer)
    answer = answer.strip().strip(" \t\n\r.,;:!?)(")
    answer = re.sub(r"\s+", "", answer)
    return answer


def check_correctness(
    predicted: str,
    gold: str,
    tolerance: float = 1e-6,
) -> bool:
    """
    Check if a predicted answer matches the gold answer.
    Numeric comparison with tolerance, fallback to string match.
    """
    if predicted is None or gold is None:
        return False

    pred_num = normalize_answer(predicted)
    gold_num = normalize_answer(gold)

    if pred_num is not None and gold_num is not None:
        if gold_num == 0:
            return abs(pred_num) < tolerance
        return abs(pred_num - gold_num) / max(abs(gold_num), 1e-10) < tolerance

    pred_clean = canonicalize_answer_text(predicted)
    gold_clean = canonicalize_answer_text(gold)
    return pred_clean == gold_clean


# ==========================================================
#     Prompt Formatting
# ==========================================================
def format_reasoning_prompt(
    question: str,
    system_prompt: Optional[str] = None,
) -> str:
    """Format a question into a reasoning prompt for CoT generation."""
    if system_prompt is None:
        system_prompt = (
            "You are a helpful assistant that solves problems step by step. "
            "Think carefully and show your reasoning."
        )
    return f"{system_prompt}\n\nProblem: {question}\n\nSolution:"


def format_chat_prompt(
    question: str,
    tokenizer,
    system_prompt: Optional[str] = None,
    enable_thinking: Optional[bool] = True,
) -> str:
    """Format using the model's chat template (preferred for reasoning models)."""
    if system_prompt is None:
        system_prompt = "You are a helpful assistant. Please reason step by step."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            # Backward compatibility for tokenizers/templates without enable_thinking.
            kwargs.pop("enable_thinking", None)
            return tokenizer.apply_chat_template(messages, **kwargs)
    else:
        return format_reasoning_prompt(question, system_prompt)


# ==========================================================
#     Helpers
# ==========================================================
def _try_float(s: str) -> Optional[float]:
    """Try to parse a string as float, return None on failure."""
    try:
        return float(str(s).replace(",", ""))
    except (ValueError, TypeError):
        return None
