import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from glp.utils_acts import MemmapReader, MemmapWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LoadReasoningActsConfig:
    acts_dir: str = "~/reasoning_acts"
    save_recovered_artifacts: bool = False
    output_dir: Optional[str] = None
    dtype: str = "float32"
    validate: bool = True
    preview_traces: int = 3
    export_glp_dataset: bool = False
    export_selector: str = "start_think"


class ReasoningActsDataset(Dataset):
    def __init__(
        self,
        reader: MemmapReader,
        trace_metadata: list[dict[str, Any]],
        hidden_dim: int,
        correctness_labels: Optional[list[int]] = None,
        phase_labels: Optional[list[int]] = None,
    ):
        self.reader = reader
        self.trace_metadata = trace_metadata
        self.hidden_dim = hidden_dim
        self.correctness_labels = correctness_labels
        self.phase_labels = phase_labels
        self.trace_offsets = self._build_trace_offsets(trace_metadata)

    @staticmethod
    def _build_trace_offsets(trace_metadata: list[dict[str, Any]]) -> list[tuple[int, int]]:
        offsets = []
        cursor = 0
        for trace in trace_metadata:
            n_tokens = int(trace.get("n_saved_tokens", 0))
            offsets.append((cursor, cursor + n_tokens))
            cursor += n_tokens
        return offsets

    def __len__(self) -> int:
        return len(self.trace_metadata)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        trace = self.trace_metadata[idx]
        start, end = self.trace_offsets[idx]
        token_rows = [self.reader[token_idx] for token_idx in range(start, end)]
        if token_rows:
            acts = torch.from_numpy(np.stack(token_rows, axis=0)).float()
        else:
            acts = torch.empty((0, self.hidden_dim), dtype=torch.float32)

        item = {
            "activations": acts,
            "trace_metadata": trace,
        }
        if self.correctness_labels is not None:
            item["correctness"] = torch.tensor(self.correctness_labels[start:end], dtype=torch.long)
        if self.phase_labels is not None:
            item["phase_labels"] = torch.tensor(self.phase_labels[start:end], dtype=torch.long)
        return item


def align_trace_metadata_to_reader(
    trace_metadata: list[dict[str, Any]],
    n_memmap_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    aligned = []
    consumed = 0
    dropped_traces = 0
    dropped_tokens = 0

    for trace in trace_metadata:
        n_tokens = int(trace.get("n_saved_tokens", 0))
        next_consumed = consumed + n_tokens
        if next_consumed <= n_memmap_rows:
            aligned.append(trace)
            consumed = next_consumed
            continue
        dropped_traces += 1
        dropped_tokens += max(0, next_consumed - n_memmap_rows)
        break

    if len(aligned) < len(trace_metadata):
        for trace in trace_metadata[len(aligned) + 1 :]:
            dropped_traces += 1
            dropped_tokens += int(trace.get("n_saved_tokens", 0))

    return aligned, {
        "aligned_saved_tokens": consumed,
        "dropped_traces": dropped_traces,
        "dropped_claimed_tokens": dropped_tokens,
    }


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_hidden_dim(indices: np.ndarray) -> int:
    chunk_lengths = indices[:, 2] - indices[:, 1]
    unique_lengths = np.unique(chunk_lengths)
    if len(unique_lengths) != 1:
        raise ValueError(
            "Expected one activation vector per memmap span with a constant hidden size, "
            f"but found span lengths {unique_lengths.tolist()}."
        )
    return int(unique_lengths[0])


def infer_trace_statistics(trace_metadata: list[dict[str, Any]]) -> dict[str, Any]:
    n_traces = len(trace_metadata)
    n_saved_tokens = sum(int(trace.get("n_saved_tokens", 0)) for trace in trace_metadata)
    n_marker_tokens = sum(int(trace.get("n_reasoning_marker_tokens", 0)) for trace in trace_metadata)
    n_generated_tokens = sum(int(trace.get("n_generated_tokens", 0)) for trace in trace_metadata)
    n_empty_generations = sum(int(trace.get("n_generated_tokens", 0)) == 0 for trace in trace_metadata)
    traces_with_single_token = sum(int(trace.get("n_saved_tokens", 0)) == 1 for trace in trace_metadata)
    traces_with_two_tokens = sum(int(trace.get("n_saved_tokens", 0)) == 2 for trace in trace_metadata)
    return {
        "n_traces": n_traces,
        "n_saved_tokens": n_saved_tokens,
        "n_marker_tokens": n_marker_tokens,
        "n_generated_tokens_total": n_generated_tokens,
        "n_empty_generations": n_empty_generations,
        "traces_with_single_saved_token": traces_with_single_token,
        "traces_with_two_saved_tokens": traces_with_two_tokens,
        "avg_saved_tokens_per_trace": (n_saved_tokens / max(n_traces, 1)),
    }


def build_recovered_metadata(
    acts_dir: Path,
    trace_metadata: list[dict[str, Any]],
    hidden_dim: int,
    indices: np.ndarray,
) -> dict[str, Any]:
    example_ids = [int(trace.get("example_idx", -1)) for trace in trace_metadata]
    trace_ids = [int(trace.get("trace_idx", -1)) for trace in trace_metadata]
    return {
        "source_dir": str(acts_dir),
        "hidden_dim": hidden_dim,
        "n_tokens": int(len(indices)),
        "n_traces_total": len(trace_metadata),
        "n_examples": len({idx for idx in example_ids if idx >= 0}),
        "saved_token_selector": "reasoning_markers_only",
        "trace_metadata_jsonl": str(acts_dir / "trace_metadata.jsonl"),
        "token_to_example": [
            ex_idx
            for trace in trace_metadata
            for ex_idx in [int(trace.get("example_idx", -1))] * int(trace.get("n_saved_tokens", 0))
        ],
        "token_to_trace": [
            trace_idx
            for trace in trace_metadata
            for trace_idx in [int(trace.get("trace_idx", -1))] * int(trace.get("n_saved_tokens", 0))
        ],
        "examples": trace_metadata,
    }


def maybe_recover_artifacts(
    acts_dir: Path,
    output_dir: Path,
    trace_metadata: list[dict[str, Any]],
    reader: MemmapReader,
    hidden_dim: int,
) -> None:
    logger.info(f"Recovering missing dataset artifacts into {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "dtype.txt").write_text(str(reader.dtype))
    metadata = build_recovered_metadata(acts_dir, trace_metadata, hidden_dim, reader.indices)
    with (output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    all_acts = np.stack([reader[idx] for idx in range(len(reader))], axis=0).astype(np.float32)
    mean = torch.from_numpy(all_acts.mean(axis=0, keepdims=True))
    var = torch.from_numpy(all_acts.var(axis=0, keepdims=True).clip(min=1e-8))
    torch.save({"mean": mean, "var": var}, output_dir / "rep_statistics.pt")


def get_selected_token_positions(
    trace_metadata: list[dict[str, Any]],
    selector: str,
) -> list[tuple[int, int]]:
    selected_positions = []
    cursor = 0

    for trace_idx, trace in enumerate(trace_metadata):
        saved_tokens = list(trace.get("saved_tokens", []))
        n_tokens = int(trace.get("n_saved_tokens", len(saved_tokens)))
        trace_positions = list(range(cursor, cursor + n_tokens))

        if selector == "all":
            selected_positions.extend((trace_idx, pos) for pos in trace_positions)
        elif selector == "start_think":
            if saved_tokens and saved_tokens[0] == "<think>" and trace_positions:
                selected_positions.append((trace_idx, trace_positions[0]))
        elif selector == "end_think":
            if saved_tokens and saved_tokens[-1] == "</think>" and trace_positions:
                selected_positions.append((trace_idx, trace_positions[-1]))
        else:
            raise ValueError(f"Unknown selector '{selector}'")

        cursor += n_tokens

    return selected_positions


def export_glp_dataset(
    dataset: ReasoningActsDataset,
    acts_dir: Path,
    output_dir: Path,
    selector: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_positions = get_selected_token_positions(dataset.trace_metadata, selector)
    if not selected_positions:
        raise ValueError(f"No activations matched selector='{selector}'")

    writer = MemmapWriter(
        output_dir=output_dir,
        file_size=max(len(selected_positions) * dataset.hidden_dim, dataset.hidden_dim),
        dtype=np.dtype("float32"),
    )

    selected_acts = []
    token_to_example = []
    token_to_trace = []
    exported_examples = []

    for trace_idx, token_idx in selected_positions:
        act = np.asarray(dataset.reader[token_idx], dtype=np.float32)
        writer.write(act)
        selected_acts.append(act)

        trace = dataset.trace_metadata[trace_idx]
        token_to_example.append(int(trace.get("example_idx", -1)))
        token_to_trace.append(int(trace.get("trace_idx", -1)))
        exported_examples.append(
            {
                "example_idx": int(trace.get("example_idx", -1)),
                "trace_idx": int(trace.get("trace_idx", -1)),
                "selector": selector,
                "question": trace.get("question"),
                "saved_tokens": trace.get("saved_tokens", []),
                "correct": trace.get("correct", -1),
            }
        )

    writer.flush()
    (output_dir / "dtype.txt").write_text("float32")

    stacked = np.stack(selected_acts, axis=0)
    mean = torch.from_numpy(stacked.mean(axis=0, keepdims=True))
    var = torch.from_numpy(stacked.var(axis=0, keepdims=True).clip(min=1e-8))
    torch.save({"mean": mean, "var": var}, output_dir / "rep_statistics.pt")

    metadata = {
        "source_dir": str(acts_dir),
        "selector": selector,
        "hidden_dim": dataset.hidden_dim,
        "n_tokens": len(selected_positions),
        "n_traces_total": len(exported_examples),
        "saved_token_selector": selector,
        "token_to_example": token_to_example,
        "token_to_trace": token_to_trace,
        "examples": exported_examples,
    }
    with (output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "export_dir": str(output_dir),
        "selector": selector,
        "n_exported_tokens": len(selected_positions),
        "hidden_dim": dataset.hidden_dim,
    }


def validate_dataset(
    acts_dir: Path,
    trace_metadata: list[dict[str, Any]],
    reader: MemmapReader,
) -> dict[str, Any]:
    stats = infer_trace_statistics(trace_metadata)
    stats["n_memmap_rows"] = len(reader)
    stats["n_trace_rows"] = len(trace_metadata)
    stats["token_row_match"] = len(reader) == stats["n_saved_tokens"]
    stats["is_marker_only_dataset"] = stats["avg_saved_tokens_per_trace"] <= 2.1
    stats["looks_complete_for_full_reasoning_glp"] = (
        stats["token_row_match"]
        and not stats["is_marker_only_dataset"]
        and (acts_dir / "metadata.json").exists()
        and (acts_dir / "rep_statistics.pt").exists()
        and (acts_dir / "dtype.txt").exists()
    )
    return stats


def load_reasoning_acts_dataset(
    acts_dir: str | Path,
    *,
    dtype: str = "float32",
    validate: bool = True,
) -> tuple[ReasoningActsDataset, dict[str, Any]]:
    acts_dir = Path(acts_dir).expanduser().resolve()
    trace_metadata_path = acts_dir / "trace_metadata.jsonl"
    data_indices_path = acts_dir / "data_indices.npy"

    if not trace_metadata_path.exists():
        raise FileNotFoundError(f"Missing trace metadata: {trace_metadata_path}")
    if not data_indices_path.exists():
        raise FileNotFoundError(f"Missing memmap indices: {data_indices_path}")

    np_dtype = np.dtype(dtype)
    reader = MemmapReader(acts_dir, np_dtype)
    trace_metadata = load_jsonl(trace_metadata_path)
    aligned_trace_metadata, alignment_stats = align_trace_metadata_to_reader(
        trace_metadata,
        len(reader),
    )
    hidden_dim = infer_hidden_dim(reader.indices)

    correctness_labels = None
    phase_labels = None
    correctness_path = acts_dir / "correctness_labels.json"
    phase_path = acts_dir / "phase_labels.json"
    if correctness_path.exists():
        correctness_labels = load_json(correctness_path)
    if phase_path.exists():
        phase_labels = load_json(phase_path)

    dataset = ReasoningActsDataset(
        reader=reader,
        trace_metadata=aligned_trace_metadata,
        hidden_dim=hidden_dim,
        correctness_labels=correctness_labels,
        phase_labels=phase_labels,
    )

    summary = {
        "acts_dir": str(acts_dir),
        "hidden_dim": hidden_dim,
        "has_metadata_json": (acts_dir / "metadata.json").exists(),
        "has_dtype_txt": (acts_dir / "dtype.txt").exists(),
        "has_rep_statistics": (acts_dir / "rep_statistics.pt").exists(),
    }
    if validate:
        summary.update(validate_dataset(acts_dir, trace_metadata, reader))
    summary.update(alignment_stats)
    summary["n_trace_rows_loaded"] = len(aligned_trace_metadata)
    return dataset, summary


def main() -> None:
    config_base = OmegaConf.structured(LoadReasoningActsConfig())
    OmegaConf.set_struct(config_base, False)
    config_cli = OmegaConf.from_cli()
    config = OmegaConf.merge(config_base, config_cli)

    dataset, summary = load_reasoning_acts_dataset(
        config.acts_dir,
        dtype=config.dtype,
        validate=config.validate,
    )

    logger.info("Reasoning activation summary:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")

    for idx in range(min(config.preview_traces, len(dataset))):
        row = dataset[idx]
        logger.info(
            "Preview trace %d: example_idx=%s trace_idx=%s acts_shape=%s saved_tokens=%s",
            idx,
            row["trace_metadata"].get("example_idx"),
            row["trace_metadata"].get("trace_idx"),
            tuple(row["activations"].shape),
            row["trace_metadata"].get("saved_tokens"),
        )

    if config.save_recovered_artifacts:
        output_dir = Path(config.output_dir or config.acts_dir).expanduser().resolve()
        maybe_recover_artifacts(
            acts_dir=Path(config.acts_dir).expanduser().resolve(),
            output_dir=output_dir,
            trace_metadata=dataset.trace_metadata,
            reader=dataset.reader,
            hidden_dim=dataset.hidden_dim,
        )

    if config.export_glp_dataset:
        export_dir = Path(config.output_dir or config.acts_dir).expanduser().resolve()
        export_summary = export_glp_dataset(
            dataset=dataset,
            acts_dir=Path(config.acts_dir).expanduser().resolve(),
            output_dir=export_dir,
            selector=config.export_selector,
        )
        logger.info("Exported GLP dataset:")
        for key, value in export_summary.items():
            logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()
