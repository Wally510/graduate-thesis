#!/usr/bin/env python3
"""在 VTaC canonical test 上评价 epoch014 + phase8 adapter。

VTaC 只作外部测试。每个 window 固定选择 target beat，并在同一 target
beat 上依次遮挡1至8个等宽 token；其他 beat 保持可见。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_PHASE_TRAIN_ROOT = (
    "/data-ai/sl20200894/Code/"
    "splitenc8_phase8_maskadapter_frozen_20260724"
)
DEFAULT_BEAT_EVAL_ROOT = (
    "/data-ai/sl20200894/Code/"
    "splitenc8_teacher_beat_reconstruction_20260724"
)
DEFAULT_COMMON_ROOT = (
    "/data-ai/sl20200894/Code/"
    "splitenc8_timepres_decoderonly_longrun_20260724"
)

PHASE_TRAIN_ROOT = Path(
    os.environ.get("PHASE_TRAIN_ROOT", DEFAULT_PHASE_TRAIN_ROOT)
)
BEAT_EVAL_ROOT = Path(
    os.environ.get("BEAT_EVAL_ROOT", DEFAULT_BEAT_EVAL_ROOT)
)
COMMON_ROOT = Path(os.environ.get("COMMON_ROOT", DEFAULT_COMMON_ROOT))
for dependency_root in (
    PHASE_TRAIN_ROOT,
    BEAT_EVAL_ROOT,
    COMMON_ROOT,
):
    sys.path.insert(0, str(dependency_root))

from evaluate_real_beat_reconstruction import (  # noqa: E402
    BeatSample,
    load_module,
    load_vtac_samples,
)
from train_phase8_maskadapter_frozen import (  # noqa: E402
    PhaseMaskAdapter,
    forward_phase_mask_adapter,
    load_common,
)


DEFAULT_CHECKPOINT = (
    "/data-ai/sl20200894/Code/foundation_stage2_ppg2ecg_aux_bundle_20260701/"
    "stage2_recon_modality_flexible_recon_mf_2gpu_periodic/"
    "model_only_epoch014_step001562066_20260716_163316.pt"
)
DEFAULT_MODEL_MODULE = (
    "/data-ai/sl20200894/Code/foundation_stage2_ppg2ecg_aux_bundle_20260701/"
    "ecg_ppg_multitask_pretrain_orthogonal_modality_flexible_merged.py"
)
DEFAULT_DOWNSTREAM_PACKAGE = (
    "/data-ai/sl20200894/Code/"
    "downstream_stage2_ppg2ecg_splitenc8_unified_20260723"
)
DEFAULT_PREPROCESS_MODULE = (
    f"{DEFAULT_DOWNSTREAM_PACKAGE}/"
    "downstream_epoch_save_packed_amp_all_ckpts_flat_20260701/"
    "direct_bp_orthogonal_foundation_singlefile.py"
)
DEFAULT_VTAC_ROOT = (
    "/data-ai/sl20200894/downstream_datasets/VTaC/extracted_1.0_20260621/"
    "vtac-a-benchmark-dataset-of-ventricular-tachycardia-alarms-from-"
    "icu-monitors-1.0"
)
DEFAULT_VTAC_MANIFEST = (
    f"{DEFAULT_DOWNSTREAM_PACKAGE}/"
    "virtual_r_vtac_ppgdalia_fixed_protocol_20260704/manifests/vtac/"
    "canonical_vtac_window_split_seed42.csv"
)

CANONICAL_TEST_RECORD_COUNT = 2658


def audit_canonical_test_manifest(path: Path) -> dict[str, Any]:
    """拒绝误用非 canonical VTaC test manifest 或重复 window。"""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    test_rows = [row for row in rows if row.get("split") == "test"]
    record_ids = [row.get("record_id", "") for row in test_rows]
    if any(not record_id for record_id in record_ids):
        raise RuntimeError("canonical VTaC test manifest含空record_id")
    unique_count = len(set(record_ids))
    report = {
        "manifest": str(path),
        "test_row_count": len(test_rows),
        "unique_record_id_count": unique_count,
        "expected_canonical_test_record_count": CANONICAL_TEST_RECORD_COUNT,
        "all_test_record_ids_unique": unique_count == len(test_rows),
    }
    if len(test_rows) != CANONICAL_TEST_RECORD_COUNT or unique_count != len(test_rows):
        raise RuntimeError(
            "VTaC canonical test审计失败："
            f"test_rows={len(test_rows)}, unique_record_id={unique_count}, "
            f"expected={CANONICAL_TEST_RECORD_COUNT}"
        )
    return report


class MetricAccumulator:
    def __init__(self) -> None:
        self.n = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_xx = 0.0
        self.sum_yy = 0.0
        self.sum_xy = 0.0

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        x = np.asarray(prediction, dtype=np.float64).reshape(-1)
        y = np.asarray(target, dtype=np.float64).reshape(-1)
        if x.size != y.size:
            raise ValueError(f"metric shape mismatch: {x.shape} vs {y.shape}")
        if x.size == 0:
            return
        delta = x - y
        self.n += int(x.size)
        self.sum_abs += float(np.abs(delta).sum())
        self.sum_sq += float(np.square(delta).sum())
        self.sum_x += float(x.sum())
        self.sum_y += float(y.sum())
        self.sum_xx += float(np.square(x).sum())
        self.sum_yy += float(np.square(y).sum())
        self.sum_xy += float((x * y).sum())

    def report(self) -> dict[str, Any]:
        if self.n == 0:
            return {
                "n": 0,
                "mae": None,
                "rmse": None,
                "pearson": None,
            }
        numerator = self.n * self.sum_xy - self.sum_x * self.sum_y
        denominator = (
            (self.n * self.sum_xx - self.sum_x**2)
            * (self.n * self.sum_yy - self.sum_y**2)
        )
        pearson = (
            numerator / math.sqrt(denominator)
            if denominator > 1e-20
            else float("nan")
        )
        return {
            "n": self.n,
            "mae": self.sum_abs / self.n,
            "rmse": math.sqrt(self.sum_sq / self.n),
            "pearson": pearson,
        }


def stable_rng(*parts: object) -> np.random.Generator:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    seed = int.from_bytes(
        hashlib.sha256(payload).digest()[:8],
        "little",
        signed=False,
    )
    return np.random.default_rng(seed)


def distribution_group(masked_phase_count: int) -> str:
    if masked_phase_count <= 4:
        return "trained_mask_range_1to4"
    if masked_phase_count <= 7:
        return "high_mask_stress_5to7"
    return "whole_target_beat_bridge_8"


def condition_strategies(masked_phase_count: int) -> tuple[str, ...]:
    return (
        ("all",)
        if masked_phase_count == 8
        else ("random", "contiguous")
    )


def choose_target_beats(
    sample: BeatSample,
    *,
    count: int,
    seed: int,
) -> list[int]:
    valid = np.flatnonzero(sample.beat_mask)
    if valid.size < 2:
        return []
    number = min(max(int(count), 1), int(valid.size))
    rng = stable_rng(seed, sample.sample_id, "target_beats")
    return sorted(
        int(value)
        for value in rng.choice(valid, size=number, replace=False)
    )


def make_condition_phase_mask(
    sample: BeatSample,
    *,
    target_beats: list[int],
    masked_phase_count: int,
    strategy: str,
    phase_tokens: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, list[int]]]:
    if not (1 <= masked_phase_count <= phase_tokens):
        raise ValueError(masked_phase_count)
    mask = np.zeros(
        (sample.beat_mask.shape[0], phase_tokens), dtype=bool
    )
    positions: dict[str, list[int]] = {}
    for beat_index in target_beats:
        if masked_phase_count == phase_tokens:
            selected = np.arange(phase_tokens, dtype=np.int64)
        else:
            rng = stable_rng(
                seed,
                sample.sample_id,
                beat_index,
                masked_phase_count,
                strategy,
            )
            if strategy == "random":
                selected = np.sort(
                    rng.choice(
                        phase_tokens,
                        size=masked_phase_count,
                        replace=False,
                    )
                )
            elif strategy == "contiguous":
                start = int(
                    rng.integers(
                        0, phase_tokens - masked_phase_count + 1
                    )
                )
                selected = np.arange(
                    start, start + masked_phase_count
                )
            else:
                raise ValueError(strategy)
        mask[beat_index, selected] = True
        positions[str(beat_index)] = [
            int(value) for value in selected.tolist()
        ]
    return mask, positions


def load_adapter_checkpoint(
    adapter: PhaseMaskAdapter,
    path: Path,
    *,
    expected_base_checkpoint: Path,
) -> dict[str, Any]:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, dict):
        raise TypeError("phase adapter checkpoint顶层必须是dict")
    state = raw.get("phase_adapter_state")
    if not isinstance(state, dict):
        raise KeyError("checkpoint缺少phase_adapter_state")
    incompatible = adapter.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "phase adapter加载不完整："
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    checkpoint_base = raw.get("base_checkpoint")
    base_name_match = (
        Path(str(checkpoint_base)).name
        == expected_base_checkpoint.name
        if checkpoint_base
        else None
    )
    if checkpoint_base and not base_name_match:
        raise RuntimeError(
            "adapter记录的base checkpoint与当前epoch014不一致："
            f"{checkpoint_base} vs {expected_base_checkpoint}"
        )
    return {
        "checkpoint": str(path),
        "step": raw.get("step"),
        "epoch": raw.get("epoch"),
        "kind": raw.get("kind"),
        "validation": raw.get("validation"),
        "base_checkpoint": checkpoint_base,
        "base_checkpoint_name_match": base_name_match,
        "policy": raw.get("policy"),
        "mask_scope": raw.get("mask_scope"),
        "phase_tokens": raw.get("phase_tokens"),
        "segment_len": raw.get("segment_len"),
        "matched_keys": sorted(state),
    }


def scalar_metrics(
    prediction: np.ndarray, target: np.ndarray
) -> dict[str, float | int | None]:
    accumulator = MetricAccumulator()
    accumulator.update(prediction, target)
    return accumulator.report()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def batched(items: list[BeatSample], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def evaluate(
    model: torch.nn.Module,
    adapter: PhaseMaskAdapter,
    samples: list[BeatSample],
    *,
    device: torch.device,
    batch_size: int,
    target_beats_per_sample: int,
    seed: int,
    output_dir: Path,
    examples_per_condition: int,
    use_amp: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    eligible = [
        sample for sample in samples if int(sample.beat_mask.sum()) >= 2
    ]
    if not eligible:
        raise RuntimeError("VTaC test没有>=2 beats的可评价window")

    target_beats_by_id = {
        sample.sample_id: choose_target_beats(
            sample,
            count=target_beats_per_sample,
            seed=seed,
        )
        for sample in eligible
    }
    metric_accumulators: dict[
        tuple[int, str, str, str], MetricAccumulator
    ] = {}
    condition_sample_counts: dict[str, int] = {}
    manifest_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []

    def accumulator(
        phase_count: int,
        strategy: str,
        channel: str,
        scope: str,
    ) -> MetricAccumulator:
        key = (phase_count, strategy, channel, scope)
        metric_accumulators.setdefault(key, MetricAccumulator())
        return metric_accumulators[key]

    for phase_count in range(1, 9):
        for strategy in condition_strategies(phase_count):
            condition_id = f"bins_{phase_count}_{strategy}"
            masks: dict[str, np.ndarray] = {}
            positions_by_id: dict[str, dict[str, list[int]]] = {}
            for sample in eligible:
                phase_mask, positions = make_condition_phase_mask(
                    sample,
                    target_beats=target_beats_by_id[sample.sample_id],
                    masked_phase_count=phase_count,
                    strategy=strategy,
                    phase_tokens=8,
                    seed=seed,
                )
                masks[sample.sample_id] = phase_mask
                positions_by_id[sample.sample_id] = positions
                manifest_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        **sample.metadata,
                        "target_beats": target_beats_by_id[
                            sample.sample_id
                        ],
                        "masked_phase_count": phase_count,
                        "mask_ratio": phase_count / 8.0,
                        "strategy": strategy,
                        "masked_phase_indices_by_beat": positions,
                        "distribution_group": distribution_group(
                            phase_count
                        ),
                        "corruption_seed": seed,
                    }
                )
            condition_sample_counts[condition_id] = len(masks)

            example_count = 0
            for group in batched(eligible, batch_size):
                beats = torch.from_numpy(
                    np.stack([sample.beats for sample in group])
                ).float().to(device)
                time_sec = torch.from_numpy(
                    np.stack([sample.time_sec for sample in group])
                ).float().to(device)
                beat_mask = torch.from_numpy(
                    np.stack([sample.beat_mask for sample in group])
                ).bool().to(device)
                phase_mask = torch.from_numpy(
                    np.stack([masks[sample.sample_id] for sample in group])
                ).bool().to(device)
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_amp,
                ):
                    prediction, sample_mask = (
                        forward_phase_mask_adapter(
                            model,
                            adapter,
                            beats,
                            time_sec,
                            beat_mask,
                            phase_mask,
                        )
                    )

                prediction_np = prediction.float().cpu().numpy()
                target_np = beats.float().cpu().numpy()
                sample_mask_np = sample_mask.cpu().numpy()
                valid_mask_np = (
                    beat_mask[:, :, None, None]
                    .expand_as(beats)
                    .cpu()
                    .numpy()
                )
                masked_np = sample_mask_np & valid_mask_np
                composite_np = np.where(
                    sample_mask_np, prediction_np, target_np
                )

                for channel_index, channel in (
                    (None, "both"),
                    (0, "ecg"),
                    (1, "ppg"),
                ):
                    if channel_index is None:
                        prediction_view = prediction_np
                        target_view = target_np
                        composite_view = composite_np
                        masked_view = masked_np
                    else:
                        prediction_view = prediction_np[
                            :, :, channel_index
                        ]
                        target_view = target_np[:, :, channel_index]
                        composite_view = composite_np[
                            :, :, channel_index
                        ]
                        masked_view = masked_np[:, :, channel_index]
                    accumulator(
                        phase_count,
                        strategy,
                        channel,
                        "masked_samples",
                    ).update(
                        prediction_view[masked_view],
                        target_view[masked_view],
                    )

                    composite_diff = np.diff(
                        composite_view, axis=-1
                    )
                    target_diff = np.diff(target_view, axis=-1)
                    within_mask = (
                        masked_view[..., 1:] & masked_view[..., :-1]
                    )
                    boundary_mask = (
                        masked_view[..., 1:] ^ masked_view[..., :-1]
                    )
                    accumulator(
                        phase_count,
                        strategy,
                        channel,
                        "within_mask_derivative",
                    ).update(
                        composite_diff[within_mask],
                        target_diff[within_mask],
                    )
                    accumulator(
                        phase_count,
                        strategy,
                        channel,
                        "boundary_derivative",
                    ).update(
                        composite_diff[boundary_mask],
                        target_diff[boundary_mask],
                    )

                for index, sample in enumerate(group):
                    mask_i = masked_np[index]
                    for channel_index, channel in (
                        (None, "both"),
                        (0, "ecg"),
                        (1, "ppg"),
                    ):
                        if channel_index is None:
                            pred_i = prediction_np[index][mask_i]
                            target_i = target_np[index][mask_i]
                        else:
                            channel_mask = mask_i[:, channel_index]
                            pred_i = prediction_np[
                                index, :, channel_index
                            ][channel_mask]
                            target_i = target_np[
                                index, :, channel_index
                            ][channel_mask]
                        report = scalar_metrics(pred_i, target_i)
                        per_sample_rows.append(
                            {
                                "sample_id": sample.sample_id,
                                "record_id": sample.metadata.get(
                                    "record_id"
                                ),
                                "subject_id": sample.metadata.get(
                                    "subject_id"
                                ),
                                "label": sample.metadata.get("label"),
                                "masked_phase_count": phase_count,
                                "mask_ratio": phase_count / 8.0,
                                "strategy": strategy,
                                "distribution_group": (
                                    distribution_group(phase_count)
                                ),
                                "channel": channel,
                                **report,
                            }
                        )

                    if example_count < examples_per_condition:
                        target_beat = target_beats_by_id[
                            sample.sample_id
                        ][0]
                        examples.append(
                            {
                                "sample_id": sample.sample_id,
                                "metadata": sample.metadata,
                                "masked_phase_count": phase_count,
                                "mask_ratio": phase_count / 8.0,
                                "strategy": strategy,
                                "distribution_group": (
                                    distribution_group(phase_count)
                                ),
                                "target_beat": target_beat,
                                "masked_phase_indices": (
                                    positions_by_id[sample.sample_id][
                                        str(target_beat)
                                    ]
                                ),
                                "original": target_np[
                                    index, target_beat
                                ].tolist(),
                                "corrupted": np.where(
                                    sample_mask_np[
                                        index, target_beat
                                    ],
                                    0.0,
                                    target_np[index, target_beat],
                                ).tolist(),
                                "raw_prediction": prediction_np[
                                    index, target_beat
                                ].tolist(),
                                "recovered": composite_np[
                                    index, target_beat
                                ].tolist(),
                            }
                        )
                        example_count += 1

    expected_condition_ids = {
        f"bins_{phase_count}_{strategy}"
        for phase_count in range(1, 9)
        for strategy in condition_strategies(phase_count)
    }
    if len(expected_condition_ids) != 15 or set(condition_sample_counts) != expected_condition_ids:
        raise RuntimeError(
            "base条件审计失败："
            f"expected={len(expected_condition_ids)}, actual={len(condition_sample_counts)}"
        )
    if "bins_8_all" not in condition_sample_counts or any(
        key.startswith("bins_8_") and key != "bins_8_all"
        for key in condition_sample_counts
    ):
        raise RuntimeError("base条件审计失败：8-bin必须仅出现一次all策略")
    if any(count != len(eligible) for count in condition_sample_counts.values()):
        raise RuntimeError(
            "base条件审计失败：某个条件没有覆盖全部eligible VTaC window"
        )

    manifest_path = output_dir / "fixed_vtac_phase_mask_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metric_rows: list[dict[str, Any]] = []
    for (phase_count, strategy, channel, scope), metric in sorted(
        metric_accumulators.items()
    ):
        metric_rows.append(
            {
                "masked_phase_count": phase_count,
                "mask_ratio": phase_count / 8.0,
                "strategy": strategy,
                "condition_id": f"bins_{phase_count}_{strategy}",
                "evaluated_sample_count": condition_sample_counts[
                    f"bins_{phase_count}_{strategy}"
                ],
                "distribution_group": distribution_group(phase_count),
                "channel": channel,
                "scope": scope,
                **metric.report(),
            }
        )
    write_csv(output_dir / "metrics_by_condition.csv", metric_rows)
    write_csv(output_dir / "per_sample_metrics.csv", per_sample_rows)
    (output_dir / "prediction_examples.json").write_text(
        json.dumps(examples, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "dataset": "VTaC canonical test",
        "loaded_sample_count": len(samples),
        "eligible_sample_count_ge_2_beats": len(eligible),
        "target_beats_per_sample": target_beats_per_sample,
        "phase_mask_counts": list(range(1, 9)),
        "strategies": {
            "1_to_7": ["random", "contiguous"],
            "8": ["all"],
        },
        "base_condition_count": len(condition_sample_counts),
        "evaluated_sample_count_by_condition": condition_sample_counts,
        "primary_metric_scope": "masked_samples",
        "mask_manifest": str(manifest_path),
        "metrics": metric_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--module-path", default=DEFAULT_MODEL_MODULE)
    parser.add_argument(
        "--preprocess-module", default=DEFAULT_PREPROCESS_MODULE
    )
    parser.add_argument("--vtac-root", default=DEFAULT_VTAC_ROOT)
    parser.add_argument(
        "--vtac-manifest", default=DEFAULT_VTAC_MANIFEST
    )
    parser.add_argument("--vtac-split", default="test", choices=("test",))
    parser.add_argument("--vtac-source-fs", type=int, default=250)
    parser.add_argument(
        "--vtac-context-start-sec", type=float, default=240.0
    )
    parser.add_argument("--vtac-window-sec", type=float, default=10.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0表示canonical test全部window",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-beats-per-sample", type=int, default=1)
    parser.add_argument("--examples-per-condition", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-beats", type=int, default=25)
    parser.add_argument("--beat-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--nhead", type=int, default=12)
    parser.add_argument("--num-layers", type=int, default=9)
    parser.add_argument("--dim-feedforward", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--phase-tokens", type=int, default=8)
    parser.add_argument("--decoder-hidden", type=int, default=256)
    parser.add_argument(
        "--skip-long-gap-suite",
        action="store_true",
        help="只运行原始15条件；正式外部测试默认同时运行独立长缺失压力套件",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase_tokens != 8 or args.beat_len != 128:
        raise ValueError("本评测固定用于splitenc8/beat_len128")
    if Path(args.checkpoint) != Path(DEFAULT_CHECKPOINT):
        raise ValueError(
            "VTaC正式参考实现固定使用epoch014 checkpoint："
            f"{DEFAULT_CHECKPOINT}"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求CUDA，但torch.cuda.is_available()为False")
    device = torch.device(args.device)
    use_amp = bool(
        args.amp
        and device.type == "cuda"
        and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    )

    audit, _ = load_common(COMMON_ROOT)
    model_module = audit.load_module(Path(args.module_path))
    model = audit.build_model(model_module, args).to(device)
    checkpoint_report = audit.load_checkpoint_exact(
        model, Path(args.checkpoint)
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    adapter = PhaseMaskAdapter(
        d_model=args.d_model,
        phase_tokens=args.phase_tokens,
        beat_len=args.beat_len,
        initial_mask_token=model.mask_token,
        hidden_dim=args.decoder_hidden,
    ).to(device)
    adapter_report = load_adapter_checkpoint(
        adapter,
        Path(args.adapter_checkpoint),
        expected_base_checkpoint=Path(args.checkpoint),
    )
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    adapter.eval()

    manifest_audit = audit_canonical_test_manifest(Path(args.vtac_manifest))
    preprocess_module = load_module(Path(args.preprocess_module))
    samples = load_vtac_samples(
        Path(args.vtac_root),
        Path(args.vtac_manifest),
        args.vtac_split,
        args.max_samples,
        args.seed,
        args.max_beats,
        args.beat_len,
        args.vtac_source_fs,
        args.vtac_context_start_sec,
        args.vtac_window_sec,
        preprocess_module,
    )
    if not samples:
        raise RuntimeError("没有加载到VTaC canonical test样本")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "base_checkpoint_report.json").write_text(
        json.dumps(checkpoint_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "adapter_checkpoint_report.json").write_text(
        json.dumps(adapter_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "evaluation_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "canonical_manifest_audit.json").write_text(
        json.dumps(manifest_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = evaluate(
        model,
        adapter,
        samples,
        device=device,
        batch_size=args.batch_size,
        target_beats_per_sample=args.target_beats_per_sample,
        seed=args.seed,
        output_dir=output_dir,
        examples_per_condition=args.examples_per_condition,
        use_amp=use_amp,
    )
    summary["canonical_manifest_audit"] = manifest_audit
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    long_gap_summary = None
    if not args.skip_long_gap_suite:
        from evaluate_phase8_vtac_long_gaps import evaluate_long_gaps

        long_gap_summary = evaluate_long_gaps(
            model,
            adapter,
            samples,
            device=device,
            batch_size=args.batch_size,
            seed=args.seed,
            output_dir=output_dir / "long_gap_stress",
            examples_per_condition=args.examples_per_condition,
            use_amp=use_amp,
        )
    print(
        f"vtac_phase8_complete samples={summary['eligible_sample_count_ge_2_beats']} "
        f"base_conditions=15 "
        f"long_gap_conditions="
        f"{0 if long_gap_summary is None else long_gap_summary['condition_count']} "
        f"output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
