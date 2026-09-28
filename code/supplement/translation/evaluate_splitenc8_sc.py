#!/usr/bin/env python3
"""One-shot formal test for the validation-selected SplitEnc8 S-C checkpoint."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler

from common_raw10s import (
    CommonBackend, DATASET_ORDER, SAMPLING_RATE, SCTranslationDataset,
    guard_server_path, sha256_file, sha256_indices,
)
from splitenc8_sc_model import SplitEnc8SC
from build_common_protocol import verify as verify_protocol


FIELDS = [
    "global_index", "dataset", "dataset_id", "record_id", "group_id",
    "mae", "mse", "rmse", "pearson", "derivative_mae",
    "foot_mae_ms", "pat_mae_ms", "valid_feet",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multi-cache-index", required=True)
    parser.add_argument("--protocol-dir", required=True)
    parser.add_argument("--legacy-train-package", required=True)
    parser.add_argument("--module-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-beats", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=3)
    parser.add_argument("--expected-world-size", type=int, default=4)
    parser.add_argument("--ddp-timeout-seconds", type=int, default=6000)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=831)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class StridedSampler(Sampler[int]):
    def __init__(self, size: int, rank: int, world: int) -> None:
        self.values = range(rank, size, world)

    def __iter__(self):
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)


def loader_kwargs(args: argparse.Namespace) -> dict:
    output = {
        "batch_size": args.batch_size, "num_workers": args.num_workers,
        "pin_memory": True, "drop_last": False, "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        output["prefetch_factor"] = args.prefetch_factor
    return output


def forward(model, batch: dict, device: torch.device) -> torch.Tensor:
    return model(
        batch["ecg"].to(device, non_blocking=True),
        batch["time_sec"].to(device, non_blocking=True),
        batch["beat_mask"].to(device, non_blocking=True),
        batch["raw_features"].to(device, non_blocking=True),
        batch["raw_beat_index"].to(device, non_blocking=True),
        batch["raw_position"].to(device, non_blocking=True),
    )


def foot_index(signal: np.ndarray, r_peak: int) -> int | None:
    lo = int(r_peak) + int(round(0.08 * SAMPLING_RATE))
    hi = min(int(r_peak) + int(round(0.55 * SAMPLING_RATE)), len(signal) - 1)
    if lo < 0 or hi <= lo + 2:
        return None
    window = signal[lo:hi]
    if not np.isfinite(window).all() or float(np.std(window)) < 1e-6:
        return None
    peak = lo + int(np.argmax(window))
    if peak <= lo + 1:
        return None
    return lo + int(np.argmin(signal[lo:peak + 1]))


def sample_rows(prediction: np.ndarray, target: np.ndarray, batch: dict) -> list[dict]:
    rows = []
    for index in range(len(prediction)):
        pred = prediction[index, 0].astype(np.float64, copy=False)
        true = target[index, 0].astype(np.float64, copy=False)
        difference = pred - true
        pred_center = pred - pred.mean()
        true_center = true - true.mean()
        denominator = math.sqrt(float(np.sum(pred_center**2) * np.sum(true_center**2)))
        pearson = float(np.sum(pred_center * true_center) / denominator) if denominator > 1e-12 else float("nan")
        feet = []
        peaks = batch["peaks"][index].cpu().numpy()
        for peak in peaks[peaks >= 0]:
            pred_foot, true_foot = foot_index(pred, int(peak)), foot_index(true, int(peak))
            if pred_foot is not None and true_foot is not None:
                feet.append(abs(pred_foot - true_foot) * 1000.0 / SAMPLING_RATE)
        dataset = str(batch["dataset"][index])
        record_id = str(batch["record_id"][index])
        rows.append({
            "global_index": int(batch["global_index"][index]),
            "dataset": dataset,
            "dataset_id": int(batch["dataset_id"][index]),
            "record_id": record_id,
            "group_id": str(batch["group_id"][index]),
            "mae": float(np.mean(np.abs(difference))),
            "mse": float(np.mean(difference**2)),
            "rmse": float(np.sqrt(np.mean(difference**2))),
            "pearson": pearson,
            "derivative_mae": float(np.mean(np.abs(np.diff(pred) - np.diff(true)))),
            "foot_mae_ms": float(np.mean(feet)) if feet else float("nan"),
            # With one shared ECG-R reference, PAT error is exactly the PPG-foot
            # timing error.  We export both names so comparison tables are clear.
            "pat_mae_ms": float(np.mean(feet)) if feet else float("nan"),
            "valid_feet": len(feet),
        })
    return rows


def mean_finite(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else float("nan")


def summarize_rows(paths: list[Path], bootstrap_samples: int, seed: int) -> dict:
    metrics = ("mae", "mse", "rmse", "pearson", "derivative_mae", "foot_mae_ms", "pat_mae_ms")
    dataset_values = {dataset: {metric: [] for metric in metrics} for dataset in DATASET_ORDER}
    group_sums: dict[str, dict[str, float]] = defaultdict(
        lambda: ({metric: 0.0 for metric in metrics} | {f"{metric}_count": 0.0 for metric in metrics})
    )
    all_values = {metric: [] for metric in metrics}
    total_rows = 0
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                total_rows += 1
                dataset, group = row["dataset"], row["group_id"]
                for metric in metrics:
                    value = float(row[metric])
                    if math.isfinite(value):
                        dataset_values[dataset][metric].append(value)
                        all_values[metric].append(value)
                        group_sums[group][metric] += value
                        group_sums[group][f"{metric}_count"] += 1.0
    group_means = {metric: [] for metric in metrics}
    for values in group_sums.values():
        for metric in metrics:
            count = values[f"{metric}_count"]
            group_means[metric].append(values[metric] / count if count else float("nan"))
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    # Exact cluster bootstrap.  One shared resampling of source-record groups is
    # reused for every metric, and draws are chunked so large UCI group counts
    # do not allocate a bootstrap_samples x groups matrix.
    matrix = np.column_stack([np.asarray(group_means[metric], dtype=np.float64) for metric in metrics])
    finite = np.isfinite(matrix)
    filled = np.where(finite, matrix, 0.0)
    groups = len(matrix)
    probabilities = np.full(groups, 1.0 / groups, dtype=np.float64)
    bytes_per_row = max(groups * np.dtype(np.int64).itemsize, 1)
    chunk = max(1, min(32, (64 * 1024 * 1024) // bytes_per_row))
    rng = np.random.default_rng(seed)
    draws = np.full((bootstrap_samples, len(metrics)), np.nan, dtype=np.float64)
    for left in range(0, bootstrap_samples, chunk):
        right = min(left + chunk, bootstrap_samples)
        weights = rng.multinomial(groups, probabilities, size=right - left)
        numerator = weights @ filled
        denominator = weights @ finite
        draws[left:right] = np.divide(
            numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0,
        )
    ci = {
        metric: [float(np.nanpercentile(draws[:, index], 2.5)), float(np.nanpercentile(draws[:, index], 97.5))]
        for index, metric in enumerate(metrics)
    }
    per_dataset = {
        dataset: {metric: mean_finite(values[metric]) for metric in metrics} | {"samples": len(values["mae"])}
        for dataset, values in dataset_values.items()
    }
    equal_dataset_macro = {
        metric: mean_finite([per_dataset[dataset][metric] for dataset in DATASET_ORDER]) for metric in metrics
    }
    sample_macro = {metric: mean_finite(all_values[metric]) for metric in metrics}
    group_macro = {metric: mean_finite(group_means[metric]) for metric in metrics}
    sample_macro["rmse_from_all_point_mse"] = math.sqrt(max(sample_macro["mse"], 0.0))
    return {
        "samples": total_rows,
        "groups": len(group_sums),
        "sample_macro": sample_macro,
        "group_macro": group_macro,
        "group_bootstrap_95ci": ci,
        "group_bootstrap_method": "exact_nonparametric_resample_source_record_groups",
        "per_dataset": per_dataset,
        "four_dataset_equal_macro": equal_dataset_macro,
    }


def main() -> None:
    args = parse_args()
    if args.max_beats < 2:
        raise ValueError("max-beats must be at least 2")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != args.expected_world_size or not torch.cuda.is_available():
        raise RuntimeError(f"expected {args.expected_world_size} CUDA ranks, got {world}")
    dist.init_process_group("nccl", timeout=timedelta(seconds=args.ddp_timeout_seconds))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    output = guard_server_path(args.output_dir, "evaluation output")
    output_exists = torch.tensor(int(output.exists()), dtype=torch.int32, device=device)
    dist.all_reduce(output_exists, op=dist.ReduceOp.MAX)
    if int(output_exists.item()):
        raise FileExistsError(f"refusing to overwrite: {output}")
    dist.barrier()
    if rank == 0:
        output.mkdir(parents=True)
    dist.barrier()

    protocol = guard_server_path(args.protocol_dir, "protocol")
    protocol_manifest_path = protocol / "split_manifest.json"
    protocol_manifest = verify_protocol(protocol)
    protocol_max_beats = int(protocol_manifest.get("model_max_beats", 25))
    if protocol_max_beats != args.max_beats:
        raise ValueError(f"protocol max beats {protocol_max_beats} != evaluation max beats {args.max_beats}")
    with np.load(protocol / "split_indices.npz") as values:
        test_indices = np.asarray(values["test"], dtype=np.int64)
    backend = CommonBackend(args.multi_cache_index)
    dataset = SCTranslationDataset(backend, test_indices, max_beats=args.max_beats)
    loader = DataLoader(dataset, sampler=StridedSampler(len(dataset), rank, world), **loader_kwargs(args))

    legacy_root = guard_server_path(args.legacy_train_package, "legacy train package")
    sys.path.insert(0, str(legacy_root))
    import trainer_core_ecg2ppg as legacy
    model_module = load_module(guard_server_path(args.module_path, "model module"), "formal_test_splitenc8_backbone")
    architecture = SimpleNamespace(beat_len=128, d_model=768, nhead=12, num_layers=9, dim_feedforward=2048, dropout=0.1, phase_tokens=8)
    backbone = legacy.build_backbone(model_module, architecture)
    checkpoint_path = guard_server_path(args.checkpoint, "translation checkpoint")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    saved_args = payload.get("args", {})
    if int(saved_args.get("max_beats", 25)) != args.max_beats:
        raise ValueError("checkpoint max-beats does not match formal evaluation")
    system = SplitEnc8SC(
        backbone, d_model=768, phase_tokens=8, beat_len=128,
        gradient_checkpointing=False, backbone_mode=str(saved_args.get("backbone_mode", "fullfinetune")),
    )
    system.load_state_dict(payload["model_state"], strict=True)
    system.to(device).eval()
    ddp = DistributedDataParallel(system, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    rank_path = output / f"per_sample_metrics_rank{rank:02d}.csv"
    with rank_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        with torch.inference_mode():
            for step, batch in enumerate(loader, start=1):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp):
                    prediction = forward(ddp, batch, device)
                target = batch["target_ppg"].numpy()
                writer.writerows(sample_rows(prediction.float().cpu().numpy(), target, batch))
                if rank == 0 and (step == 1 or step % 200 == 0):
                    print(f"test_step={step}/{len(loader)}", flush=True)
    dist.barrier()

    if rank == 0:
        rank_paths = [output / f"per_sample_metrics_rank{value:02d}.csv" for value in range(world)]
        merged = output / "per_sample_metrics.csv"
        with merged.open("wb") as destination:
            for index, path in enumerate(rank_paths):
                with path.open("rb") as source:
                    if index:
                        source.readline()
                    shutil.copyfileobj(source, destination)
        summary = summarize_rows(rank_paths, args.bootstrap_samples, args.seed + 991)
        if int(summary["samples"]) != int(len(test_indices)):
            raise RuntimeError(f"formal test sample count mismatch: {summary['samples']} != {len(test_indices)}")
        summary.update({
            "status": "passed",
            "protocol": protocol_manifest["protocol"],
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "protocol_manifest_sha256": sha256_file(protocol_manifest_path),
            "protocol_counts": protocol_manifest["counts"],
            "test_indices_sha256_int64": sha256_indices(test_indices),
            "selected_by": "validation_loss_only",
            "test_used_for_selection": False,
            "source": "ECG only",
            "target_in_forward": False,
            "coordinate": "per-window-zscore PPG [1,2500]",
            "max_beats": args.max_beats,
            "foot_definition": "minimum before systolic maximum in ECG-R + 80..550 ms",
            "pat_definition": "PPG-foot minus shared ECG-R; therefore absolute PAT error equals foot_mae_ms",
        })
        (output / "test_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output / "RUN_COMPLETE.txt").write_text(json.dumps({"status": "complete", "samples": summary["samples"]}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
