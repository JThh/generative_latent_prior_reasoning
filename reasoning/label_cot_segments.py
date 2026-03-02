"""
Heuristic + keyword-based labeling of chain-of-thought (CoT) segments
into cognitive operation categories, following the Reasoning Behaviour
Ontology defined in the Reasoning GLP proposal.

The labeler is intentionally simple — GLP meta-neurons should discover
finer-grained structure beyond what these surface-level heuristics capture.

Categories follow the hierarchical ontology:
  Linear:       step_by_step_deduction, calculation_execution
  Non-linear:   verification, backtracking, strategy_switching
  Meta-cognitive: subgoal_formation, confidence_assessment, error_recognition
  Failure mode: circular_reasoning, overthinking
  Termination:  answer_crystallisation
"""

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ==============================
#   Cognitive Operation Labels
# ==============================
# Each label is defined by a set of keyword/regex triggers.
# A token is labeled positive if ANY trigger matches the surrounding text window.

DEFAULT_LABEL_SPECS = {
    # ── Linear behaviours ──────────────────────────────────────
    "step_by_step_deduction": {
        "category": "linear",
        "description": "Sequential logical or mathematical inference",
        "keywords": [
            "therefore", "since", "because", "it follows that",
            "which means", "this implies", "given that", "we know that",
            "from this", "using the fact", "by definition",
            "applying", "according to", "based on",
        ],
    },
    "calculation_execution": {
        "category": "linear",
        "description": "Performing arithmetic or symbolic computation",
        "regex_patterns": [
            r"\d+\s*[\+\-\*\/\×\÷]\s*\d+",      # e.g., 3 + 5
            r"\d+\s*=\s*\d+",                      # e.g., 15 = 15
            r"\d+\s*[\>\<\≥\≤]\s*\d+",            # comparisons
            r"\d+\s*\^\s*\d+",                     # exponents
        ],
        "keywords": [
            "calculate", "computing", "evaluating", "substituting",
            "plugging in", "simplifying", "expanding", "factoring",
            "multiplying", "dividing", "adding", "subtracting",
        ],
    },
    # ── Non-linear behaviours ──────────────────────────────────
    "verification": {
        "category": "non-linear",
        "description": "Checking a previous result for correctness",
        "keywords": [
            "let me check", "let me verify", "verify", "double-check",
            "double check", "is correct", "checking", "to confirm",
            "let's check", "let's verify", "we can check", "we can verify",
            "sanity check", "makes sense", "consistent with",
            "does this hold", "is this right", "checking our work",
        ],
    },
    "backtracking": {
        "category": "non-linear",
        "description": "Abandoning current approach, returning to an earlier point",
        "keywords": [
            "wait", "actually", "no,", "that's wrong", "that is wrong",
            "let me reconsider", "let me rethink", "on second thought",
            "I made a mistake", "I made an error", "let me redo",
            "let me try again", "hmm", "hold on", "scratch that",
            "going back", "let me restart", "that doesn't seem right",
            "let me go back", "starting over",
        ],
    },
    "strategy_switching": {
        "category": "non-linear",
        "description": "Adopting a fundamentally different solution approach",
        "keywords": [
            "alternatively", "another approach", "different method",
            "instead,", "let me try a different", "different way",
            "another way", "new approach", "switch to",
            "try using", "what if we", "perhaps we should",
            "let's use", "a better approach", "more efficient",
        ],
    },
    # ── Meta-cognitive behaviours ──────────────────────────────
    "subgoal_formation": {
        "category": "meta-cognitive",
        "description": "Decomposing the problem into sub-problems",
        "keywords": [
            "first,", "second,", "third,", "next,", "then,",
            "step 1", "step 2", "step 3", "my approach",
            "let me think", "I need to", "I should",
            "let's start", "the plan is", "to solve this",
            "breaking this down", "break this down",
            "I'll start by", "I will start by", "the key steps",
            "sub-problem", "subproblem", "part 1", "part 2",
        ],
    },
    "confidence_assessment": {
        "category": "meta-cognitive",
        "description": "Internal evaluation of certainty in current approach",
        "keywords": [
            "I'm confident", "I'm sure", "I think this is right",
            "I'm not sure", "I'm uncertain", "this might be wrong",
            "probably", "likely", "unlikely", "it seems",
            "I believe", "I suspect", "this should be",
            "if I'm not mistaken", "I'm fairly certain",
            "this is tricky", "not obvious", "hard to tell",
        ],
    },
    "error_recognition": {
        "category": "meta-cognitive",
        "description": "Detecting a mistake in previous reasoning",
        "keywords": [
            "mistake", "error", "incorrect", "that doesn't work",
            "that can't be right", "impossible", "contradiction",
            "this is wrong", "this doesn't make sense", "that's not right",
            "doesn't add up", "I went wrong", "went wrong",
            "off by", "miscalculated", "miscounted", "typo",
            "I see the error", "the issue is", "the problem is",
        ],
    },
    # ── Failure modes ──────────────────────────────────────────
    "circular_reasoning": {
        "category": "failure_mode",
        "description": "Repeating the same reasoning loop without progress",
        "keywords": [
            "as I said before", "as mentioned earlier",
            "going in circles", "I keep getting", "same result",
            "back to where", "this again", "repeating",
            "I already tried", "we already know", "as before",
        ],
    },
    "overthinking": {
        "category": "failure_mode",
        "description": "Excessive deliberation on a simple sub-problem",
        "keywords": [
            "let me think about this more", "to be thorough",
            "let me consider all", "exhaustively", "every possibility",
            "just to be safe", "one more check", "let me also verify",
            "additionally", "furthermore", "moreover", "also,",
            "let me also consider", "we should also",
        ],
    },
    # ── Termination ────────────────────────────────────────────
    "answer_crystallisation": {
        "category": "termination",
        "description": "Converging on a final answer; readiness to stop thinking",
        "keywords": [
            "the answer is", "final answer", "the result is",
            "in conclusion", "therefore the answer",
            "so the answer", "thus the answer",
            "we get", "we find that", "we obtain",
            "hence,", "which gives us",
            "\\boxed", "####",
        ],
    },
}


# ==============================
#   Reasoning Phase Detection
# ==============================
REASONING_PHASES = {
    "pre_reasoning": {
        "description": "Prompt encoding, before reasoning begins",
        "markers_start": [],  # assigned by position (before <think>)
        "markers_end": ["<think>", "Let me", "I need to"],
    },
    "early_reasoning": {
        "description": "Initial approach formation, first ~15% of thinking",
        "fraction_range": (0.0, 0.15),
    },
    "mid_reasoning": {
        "description": "Main computation, middle ~55% of thinking",
        "fraction_range": (0.15, 0.70),
    },
    "late_reasoning": {
        "description": "Verification and answer formation, last ~25% of thinking",
        "fraction_range": (0.70, 0.95),
    },
    "post_reasoning": {
        "description": "Final answer output, after thinking ends",
        "fraction_range": (0.95, 1.0),
    },
}


def assign_reasoning_phase(
    token_idx: int,
    n_total_tokens: int,
    think_start_idx: int = 0,
    think_end_idx: Optional[int] = None,
) -> str:
    """
    Assign a reasoning phase label to a token based on its position
    within the chain-of-thought trace.

    Args:
        token_idx: Index of the token within the generated sequence.
        n_total_tokens: Total number of generated tokens.
        think_start_idx: Token index where <think> begins (0 if no marker).
        think_end_idx: Token index where </think> ends (n_total if no marker).

    Returns:
        One of: "pre_reasoning", "early_reasoning", "mid_reasoning",
                "late_reasoning", "post_reasoning"
    """
    if think_end_idx is None:
        think_end_idx = n_total_tokens

    if token_idx < think_start_idx:
        return "pre_reasoning"
    if token_idx >= think_end_idx:
        return "post_reasoning"

    # Compute fractional position within thinking span
    think_length = max(think_end_idx - think_start_idx, 1)
    frac = (token_idx - think_start_idx) / think_length

    if frac < 0.15:
        return "early_reasoning"
    elif frac < 0.70:
        return "mid_reasoning"
    elif frac < 0.95:
        return "late_reasoning"
    else:
        return "post_reasoning"


def assign_all_phases(
    n_tokens: int,
    token_strings: Optional[list[str]] = None,
) -> list[str]:
    """
    Assign reasoning phases to all tokens in a CoT trace.

    If token_strings are provided, detects <think>...</think> boundaries.
    Otherwise assigns purely by fractional position.
    """
    think_start = 0
    think_end = n_tokens

    if token_strings is not None:
        full_text = "".join(token_strings)
        # Detect <think>...</think> boundaries
        think_start_match = re.search(r"<think>", full_text, re.IGNORECASE)
        think_end_match = re.search(r"</think>", full_text, re.IGNORECASE)

        if think_start_match:
            # Find which token index corresponds to the <think> position
            char_pos = think_start_match.end()
            cumulative = 0
            for i, ts in enumerate(token_strings):
                cumulative += len(ts)
                if cumulative >= char_pos:
                    think_start = i + 1
                    break

        if think_end_match:
            char_pos = think_end_match.start()
            cumulative = 0
            for i, ts in enumerate(token_strings):
                cumulative += len(ts)
                if cumulative >= char_pos:
                    think_end = i
                    break

    return [
        assign_reasoning_phase(i, n_tokens, think_start, think_end)
        for i in range(n_tokens)
    ]


# ==============================
#   Cognitive Labeler
# ==============================
@dataclass
class CognitiveLabeler:
    """
    Labels tokens in a CoT trace with binary cognitive operation indicators
    following the Reasoning Behaviour Ontology.

    Each token receives binary labels based on whether the surrounding text
    window (centered on that token) matches any trigger for that cognitive
    operation.

    Parameters:
        label_specs: Dict mapping label names to their trigger specifications.
        window_size: Number of chars around each token boundary to check.
    """
    label_specs: dict = field(default_factory=lambda: DEFAULT_LABEL_SPECS)
    window_size: int = 80

    def __post_init__(self):
        # Pre-compile regex patterns for efficiency
        self._compiled = {}
        for label_name, spec in self.label_specs.items():
            patterns = []
            for kw in spec.get("keywords", []):
                patterns.append(re.compile(re.escape(kw), re.IGNORECASE))
            for pat in spec.get("regex_patterns", []):
                patterns.append(re.compile(pat, re.IGNORECASE))
            self._compiled[label_name] = patterns

    def label_text(self, text: str) -> dict[str, bool]:
        """
        Label an entire text span with cognitive operation indicators.

        Returns a dict mapping each label name to a boolean indicating
        whether the text contains any trigger for that operation.
        """
        labels = {}
        for label_name, patterns in self._compiled.items():
            labels[label_name] = any(pat.search(text) for pat in patterns)
        return labels

    def label_tokens(
        self,
        token_strings: list[str],
        token_offsets: Optional[list[int]] = None,
    ) -> dict[str, np.ndarray]:
        """
        Label each token in a CoT trace with cognitive operation indicators.

        Args:
            token_strings: List of decoded token strings.
            token_offsets: Character offsets of each token in the full text.
                If None, offsets are computed by concatenating token_strings.

        Returns:
            Dict mapping label names to boolean numpy arrays of shape (n_tokens,).
        """
        if token_offsets is None:
            full_text = "".join(token_strings)
            offsets = []
            pos = 0
            for ts in token_strings:
                offsets.append(pos)
                pos += len(ts)
            token_offsets = offsets
        else:
            full_text = "".join(token_strings)

        n_tokens = len(token_strings)
        labels = {name: np.zeros(n_tokens, dtype=bool) for name in self._compiled}

        for i, offset in enumerate(token_offsets):
            start = max(0, offset - self.window_size)
            end = min(len(full_text), offset + len(token_strings[i]) + self.window_size)
            window = full_text[start:end]

            for label_name, patterns in self._compiled.items():
                if any(pat.search(window) for pat in patterns):
                    labels[label_name][i] = True

        return labels

    def label_tokens_with_phases(
        self,
        token_strings: list[str],
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """
        Label tokens with both cognitive operations and reasoning phases.

        Returns:
            (cognitive_labels, phase_labels) where cognitive_labels maps
            label names to boolean arrays and phase_labels is a list of
            phase strings per token.
        """
        cognitive = self.label_tokens(token_strings)
        phases = assign_all_phases(len(token_strings), token_strings)
        return cognitive, phases

    @property
    def label_names(self) -> list[str]:
        """Return the list of all cognitive operation label names."""
        return list(self.label_specs.keys())

    @property
    def categories(self) -> dict[str, list[str]]:
        """Return a mapping from category to label names."""
        cats = {}
        for name, spec in self.label_specs.items():
            cat = spec.get("category", "other")
            cats.setdefault(cat, []).append(name)
        return cats


def summarize_labels(labels: dict[str, np.ndarray]) -> dict[str, dict]:
    """
    Summarize cognitive label statistics for a CoT trace.

    Returns dict with per-label count and fraction of positive tokens.
    """
    summary = {}
    for name, arr in labels.items():
        n_total = len(arr)
        n_pos = int(arr.sum()) if hasattr(arr, 'sum') else sum(arr)
        summary[name] = {
            "count": n_pos,
            "total": n_total,
            "fraction": n_pos / n_total if n_total > 0 else 0.0,
        }
    return summary
