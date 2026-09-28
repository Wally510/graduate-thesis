#!/usr/bin/env python3
"""VTaC phase8 的跨 beat、连续多 beat 与 oracle 秒级长缺失压力测试。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from train_phase8_maskadapter_frozen import (
    PhaseMaskAdapter,
    fixed_phase_position,
)


@dataclass(frozen=True)
class LongGapCondition:
    condition_id: str
    family: str
    requested_tokens: int | None = None
    requested_whole_beats: int | None = None
    requested_gap_sec: float | None = None
    left_bins: int | None = None
    full_beats: int | None = None
    right_bins: int | None = None


def long_gap_conditions() -> list[LongGapCondition]:
    conditions = [
        LongGapCondition(
            f"cross_boundary_{tokens:02d}tokens",
            "cross_beat_boundary",
            requested_tokens=tokens,
        )
        for tokens in (4, 8, 12, 16, 24)
    ]
    conditions += [
        LongGapCondition(
            f"whole_beats_{beats}",
            "consecutive_whole_beats",
            requested_tokens=beats * 8,
            requested_whole_beats=beats,
        )
        for beats in (1, 2, 3, 4)
    ]
    conditions += [
        LongGapCondition(
            f"mixed_{left}+{full * 8}+{right}",
            "partial_full_partial",
            requested_tokens=left + full * 8 + right,
            left_bins=left,
            full_beats=full,
            right_bins=right,
        )
        for left, full, right in ((2, 1, 2), (4, 1, 4), (4, 2, 4))
    ]
    conditions += [
        LongGapCondition(
            f"oracle_time_gap_{str(seconds).replace('.', 'p')}sec",
            "oracle_time_gap",
            requested_gap_sec=seconds,
        )
        for seconds in (0.5, 1.0, 2.0, 4.0)
    ]
    return conditions


def stable_rng(*parts: object) -> np.random.Generator:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    seed = int.from_bytes(
        hashlib.sha256(payload).digest()[:8], "little", signed=False
    )
    return np.random.default_rng(seed)


def _valid_consecutive_runs(beat_mask: np.ndarray) -> list[np.ndarray]:
    valid = np.flatnonzero(beat_mask)
    if valid.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(valid) != 1) + 1
    return [run for run in np.split(valid, breaks) if run.size]


def _token_mask_to_sample_mask(
    phase_mask: np.ndarray,
    *,
    beat_len: int = 128,
    phase_tokens: int = 8,
) -> np.ndarray:
    segment_len = beat_len // phase_tokens
    one_channel = np.repeat(phase_mask, segment_len, axis=-1)
    return np.repeat(one_channel[:, None, :], 2, axis=1)


def _choose_candidate(
    candidates: list[np.ndarray],
    *,
    seed: int,
    sample_id: str,
    condition_id: str,
) -> np.ndarray | None:
    if not candidates:
        return None
    rng = stable_rng(seed, sample_id, condition_id, "long_gap_location")
    return candidates[int(rng.integers(0, len(candidates)))]


def _cross_boundary_phase_mask(
    beat_mask: np.ndarray,
    tokens: int,
    *,
    seed: int,
    sample_id: str,
    condition_id: str,
) -> np.ndarray | None:
    phase_mask = np.zeros((beat_mask.size, 8), dtype=bool)
    candidates: list[np.ndarray] = []
    for run in _valid_consecutive_runs(beat_mask):
        flat = np.concatenate(
            [np.arange(int(beat) * 8, (int(beat) + 1) * 8) for beat in run]
        )
        if flat.size <= tokens:
            continue
        for start in range(0, flat.size - tokens + 1):
            selected = flat[start : start + tokens]
            if selected[0] // 8 == selected[-1] // 8:
                continue
            candidates.append(selected)
    selected = _choose_candidate(
        candidates,
        seed=seed,
        sample_id=sample_id,
        condition_id=condition_id,
    )
    if selected is None:
        return None
    phase_mask.reshape(-1)[selected] = True
    return phase_mask


def _whole_beats_phase_mask(
    beat_mask: np.ndarray,
    count: int,
    *,
    seed: int,
    sample_id: str,
    condition_id: str,
) -> np.ndarray | None:
    candidates: list[np.ndarray] = []
    for run in _valid_consecutive_runs(beat_mask):
        if run.size <= count:
            continue
        candidates.extend(
            run[start : start + count]
            for start in range(0, run.size - count + 1)
        )
    selected = _choose_candidate(
        candidates,
        seed=seed,
        sample_id=sample_id,
        condition_id=condition_id,
    )
    if selected is None:
        return None
    phase_mask = np.zeros((beat_mask.size, 8), dtype=bool)
    phase_mask[selected] = True
    return phase_mask


def _mixed_phase_mask(
    beat_mask: np.ndarray,
    *,
    left_bins: int,
    full_beats: int,
    right_bins: int,
    seed: int,
    sample_id: str,
    condition_id: str,
) -> np.ndarray | None:
    width = full_beats + 2
    candidates: list[np.ndarray] = []
    for run in _valid_consecutive_runs(beat_mask):
        if run.size <= width:
            continue
        candidates.extend(
            run[start : start + width]
            for start in range(0, run.size - width + 1)
        )
    selected = _choose_candidate(
        candidates,
        seed=seed,
        sample_id=sample_id,
        condition_id=condition_id,
    )
    if selected is None:
        return None
    phase_mask = np.zeros((beat_mask.size, 8), dtype=bool)
    phase_mask[int(selected[0]), 8 - left_bins :] = True
    phase_mask[selected[1:-1]] = True
    phase_mask[int(selected[-1]), :right_bins] = True
    return phase_mask


def _estimated_beat_edges(
    time_sec: np.ndarray, beat_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    valid = np.flatnonzero(beat_mask)
    if valid.size < 2:
        return None
    centers = np.asarray(time_sec[valid], dtype=np.float64)
    if not np.all(np.diff(centers) > 0):
        return None
    midpoint = 0.5 * (centers[:-1] + centers[1:])
    left = np.concatenate(
        [[centers[0] - 0.5 * (centers[1] - centers[0])], midpoint]
    )
    right = np.concatenate(
        [midpoint, [centers[-1] + 0.5 * (centers[-1] - centers[-2])]]
    )
    return left, right


def _oracle_time_gap_masks(
    time_sec: np.ndarray,
    beat_mask: np.ndarray,
    gap_sec: float,
    *,
    seed: int,
    sample_id: str,
    condition_id: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]] | None:
    valid = np.flatnonzero(beat_mask)
    edges = _estimated_beat_edges(time_sec, beat_mask)
    if edges is None:
        return None
    left, right = edges
    total_start = float(left[0])
    total_end = float(right[-1])
    if total_end - total_start <= gap_sec:
        return None

    # 尽量在缺失区间两侧都保留上下文；窗口太短时退化为只保证仍有可见点。
    context = min(0.25, max((total_end - total_start - gap_sec) / 4.0, 0.0))
    low = total_start + context
    high = total_end - gap_sec - context
    if high < low:
        low, high = total_start, total_end - gap_sec
    rng = stable_rng(seed, sample_id, condition_id, "oracle_time_gap_start")
    start = float(rng.uniform(low, high)) if high > low else float(low)
    end = start + float(gap_sec)

    sample_mask = np.zeros((beat_mask.size, 2, 128), dtype=bool)
    for local_index, beat_index in enumerate(valid):
        positions = left[local_index] + (
            np.arange(128, dtype=np.float64) + 0.5
        ) / 128.0 * (right[local_index] - left[local_index])
        masked = (positions >= start) & (positions < end)
        sample_mask[int(beat_index), :, masked] = True
    if not sample_mask.any():
        return None
    one_channel = sample_mask[:, 0].reshape(beat_mask.size, 8, 16)
    phase_mask = one_channel.any(axis=-1)
    if phase_mask.all():
        return None
    return phase_mask, sample_mask, {
        "requested_gap_start_sec_relative": start,
        "requested_gap_end_sec_relative": end,
        "requested_gap_sec": float(gap_sec),
    }


def make_long_gap_masks(
    sample,
    condition: LongGapCondition,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    beat_mask = np.asarray(sample.beat_mask, dtype=bool)
    extra: dict[str, Any] = {
        "requested_gap_start_sec_relative": None,
        "requested_gap_end_sec_relative": None,
    }
    if condition.family == "cross_beat_boundary":
        phase_mask = _cross_boundary_phase_mask(
            beat_mask,
            int(condition.requested_tokens),
            seed=seed,
            sample_id=sample.sample_id,
            condition_id=condition.condition_id,
        )
    elif condition.family == "consecutive_whole_beats":
        phase_mask = _whole_beats_phase_mask(
            beat_mask,
            int(condition.requested_whole_beats),
            seed=seed,
            sample_id=sample.sample_id,
            condition_id=condition.condition_id,
        )
    elif condition.family == "partial_full_partial":
        phase_mask = _mixed_phase_mask(
            beat_mask,
            left_bins=int(condition.left_bins),
            full_beats=int(condition.full_beats),
            right_bins=int(condition.right_bins),
            seed=seed,
            sample_id=sample.sample_id,
            condition_id=condition.condition_id,
        )
    elif condition.family == "oracle_time_gap":
        result = _oracle_time_gap_masks(
            sample.time_sec,
            beat_mask,
            float(condition.requested_gap_sec),
            seed=seed,
            sample_id=sample.sample_id,
            condition_id=condition.condition_id,
        )
        if result is None:
            return None
        phase_mask, sample_mask, raw_extra = result
        extra.update(raw_extra)
        return phase_mask, sample_mask, extra
    else:
        raise ValueError(condition.family)
    if phase_mask is None:
        return None
    return phase_mask, _token_mask_to_sample_mask(phase_mask), extra


def forward_with_corruption_mask(
    model: torch.nn.Module,
    adapter: PhaseMaskAdapter,
    beats: torch.Tensor,
    time_sec: torch.Tensor,
    beat_mask: torch.Tensor,
    phase_missing_mask: torch.Tensor,
    sample_corruption_mask: torch.Tensor,
) -> torch.Tensor:
    """允许秒级缺失只破坏真实缺失采样点，同时替换所有受影响 token。"""
    b, k, channels, length = beats.shape
    corrupted = beats.masked_fill(sample_corruption_mask, 0.0)
    with torch.no_grad():
        ecg_x = corrupted[:, :, 0:1].reshape(b * k, 1, length)
        ppg_x = corrupted[:, :, 1:2].reshape(b * k, 1, length)
        ecg_z = model.ecg_encoder.forward_phase(ecg_x).reshape(
            b, k, adapter.phase_tokens, adapter.d_model
        )
        ppg_z = model.ppg_encoder.forward_phase(ppg_x).reshape(
            b, k, adapter.phase_tokens, adapter.d_model
        )
        stacked = torch.stack([ecg_z, ppg_z], dim=2)
        gate_logits = model.fusion_gate(stacked.mean(dim=3)).squeeze(-1)
        alpha = torch.softmax(gate_logits, dim=2)
        fused = (alpha[:, :, :, None, None] * stacked).sum(dim=2)
        phase_position = fixed_phase_position(model).view(
            1, 1, adapter.phase_tokens, adapter.d_model
        )
        input_mode = torch.zeros(b, dtype=torch.long, device=beats.device)
        mode_embedding = model.input_mode_embed(input_mode).view(
            b, 1, 1, adapter.d_model
        )
        time_embedding = model.time_encoding(time_sec).unsqueeze(2)
        masked_token = adapter.phase_mask_token + phase_position
        phase_z = torch.where(
            phase_missing_mask.unsqueeze(-1), masked_token, fused
        )
        phase_z = phase_z + mode_embedding + time_embedding
        for block in model.blocks:
            phase_z = block(phase_z, beat_mask)
        phase_z = model.norm(phase_z)
        return adapter.decode(phase_z)


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
            return {"n": 0, "mae": None, "rmse": None, "pearson": None}
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


def _scalar_metrics(
    prediction: np.ndarray, target: np.ndarray
) -> dict[str, Any]:
    metric = MetricAccumulator()
    metric.update(prediction, target)
    return metric.report()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _batched(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _true_ranges(mask: np.ndarray) -> list[list[int]]:
    values = np.flatnonzero(mask)
    if values.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(values) != 1) + 1
    return [
        [int(group[0]), int(group[-1]) + 1]
        for group in np.split(values, breaks)
    ]


def _mask_description(
    phase_mask: np.ndarray, sample_mask: np.ndarray
) -> dict[str, Any]:
    phase_by_beat = {
        str(beat): np.flatnonzero(phase_mask[beat]).astype(int).tolist()
        for beat in np.flatnonzero(phase_mask.any(axis=1))
    }
    sample_ranges = {
        str(beat): _true_ranges(sample_mask[beat, 0])
        for beat in np.flatnonzero(sample_mask[:, 0].any(axis=1))
    }
    return {
        "masked_phase_indices_by_beat": phase_by_beat,
        "masked_sample_ranges_by_beat_end_exclusive": sample_ranges,
        "actual_masked_phase_token_count": int(phase_mask.sum()),
        "actual_masked_samples_per_channel": int(sample_mask[:, 0].sum()),
    }


def evaluate_long_gaps(
    model: torch.nn.Module,
    adapter: PhaseMaskAdapter,
    samples: list[Any],
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
    output_dir: Path,
    examples_per_condition: int,
    use_amp: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    base_eligible = [
        sample for sample in samples if int(sample.beat_mask.sum()) >= 2
    ]
    conditions = long_gap_conditions()
    accumulators: dict[tuple[str, str, str], MetricAccumulator] = {}
    manifest_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    condition_counts: dict[str, int] = {}

    def accumulator(
        condition_id: str, channel: str, scope: str
    ) -> MetricAccumulator:
        key = (condition_id, channel, scope)
        accumulators.setdefault(key, MetricAccumulator())
        return accumulators[key]

    for condition in conditions:
        prepared: list[tuple[Any, np.ndarray, np.ndarray, dict[str, Any]]] = []
        for sample in base_eligible:
            result = make_long_gap_masks(sample, condition, seed=seed)
            if result is None:
                continue
            phase_mask, sample_mask, extra = result
            description = _mask_description(phase_mask, sample_mask)
            prepared.append((sample, phase_mask, sample_mask, extra))
            manifest_rows.append(
                {
                    "sample_id": sample.sample_id,
                    **sample.metadata,
                    **asdict(condition),
                    **description,
                    **extra,
                    "corruption_seed": seed,
                    "segmentation_policy": "oracle_complete_signal_rr_segmentation",
                }
            )
        condition_counts[condition.condition_id] = len(prepared)
        if not prepared:
            raise RuntimeError(
                "long-gap条件没有可评价VTaC window："
                f"{condition.condition_id}; 不应静默写出空指标。"
            )

        example_count = 0
        for group in _batched(prepared, batch_size):
            beats = torch.from_numpy(
                np.stack([item[0].beats for item in group])
            ).float().to(device)
            time_sec = torch.from_numpy(
                np.stack([item[0].time_sec for item in group])
            ).float().to(device)
            beat_mask = torch.from_numpy(
                np.stack([item[0].beat_mask for item in group])
            ).bool().to(device)
            phase_mask = torch.from_numpy(
                np.stack([item[1] for item in group])
            ).bool().to(device)
            sample_mask = torch.from_numpy(
                np.stack([item[2] for item in group])
            ).bool().to(device)
            valid = beat_mask[:, :, None, None].expand_as(beats)
            evaluation_mask = sample_mask & valid
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                prediction = forward_with_corruption_mask(
                    model,
                    adapter,
                    beats,
                    time_sec,
                    beat_mask,
                    phase_mask,
                    sample_mask,
                )

            prediction_np = prediction.float().cpu().numpy()
            target_np = beats.float().cpu().numpy()
            mask_np = evaluation_mask.cpu().numpy()
            composite_np = np.where(mask_np, prediction_np, target_np)

            for channel_index, channel in (
                (None, "both"),
                (0, "ecg"),
                (1, "ppg"),
            ):
                if channel_index is None:
                    pred_view, target_view = prediction_np, target_np
                    composite_view, mask_view = composite_np, mask_np
                else:
                    pred_view = prediction_np[:, :, channel_index]
                    target_view = target_np[:, :, channel_index]
                    composite_view = composite_np[:, :, channel_index]
                    mask_view = mask_np[:, :, channel_index]
                accumulator(
                    condition.condition_id, channel, "masked_samples"
                ).update(pred_view[mask_view], target_view[mask_view])
                composite_diff = np.diff(composite_view, axis=-1)
                target_diff = np.diff(target_view, axis=-1)
                within = mask_view[..., 1:] & mask_view[..., :-1]
                boundary = mask_view[..., 1:] ^ mask_view[..., :-1]
                accumulator(
                    condition.condition_id,
                    channel,
                    "within_mask_derivative",
                ).update(composite_diff[within], target_diff[within])
                accumulator(
                    condition.condition_id,
                    channel,
                    "boundary_derivative",
                ).update(composite_diff[boundary], target_diff[boundary])

            for index, (sample, phase_i, sample_i, extra) in enumerate(group):
                for channel_index, channel in (
                    (None, "both"),
                    (0, "ecg"),
                    (1, "ppg"),
                ):
                    if channel_index is None:
                        local_mask = mask_np[index]
                        pred_i = prediction_np[index][local_mask]
                        target_i = target_np[index][local_mask]
                    else:
                        local_mask = mask_np[index, :, channel_index]
                        pred_i = prediction_np[index, :, channel_index][local_mask]
                        target_i = target_np[index, :, channel_index][local_mask]
                    per_sample_rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "record_id": sample.metadata.get("record_id"),
                            **asdict(condition),
                            **_mask_description(phase_i, sample_i),
                            **extra,
                            "channel": channel,
                            **_scalar_metrics(pred_i, target_i),
                        }
                    )
                if example_count < examples_per_condition:
                    examples.append(
                        {
                            "sample_id": sample.sample_id,
                            "metadata": sample.metadata,
                            **asdict(condition),
                            **_mask_description(phase_i, sample_i),
                            **extra,
                            "original": target_np[index].tolist(),
                            "corrupted": np.where(
                                sample_i, 0.0, target_np[index]
                            ).tolist(),
                            "raw_prediction": prediction_np[index].tolist(),
                            "recovered": composite_np[index].tolist(),
                        }
                    )
                    example_count += 1

    manifest_path = output_dir / "fixed_vtac_long_gap_mask_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_id = {condition.condition_id: condition for condition in conditions}
    if len(by_id) != 16 or len(condition_counts) != 16:
        raise RuntimeError(
            "long-gap条件审计失败："
            f"defined={len(by_id)}, evaluated={len(condition_counts)}"
        )
    metric_rows: list[dict[str, Any]] = []
    for (condition_id, channel, scope), metric in sorted(accumulators.items()):
        metric_rows.append(
            {
                **asdict(by_id[condition_id]),
                "evaluated_sample_count": condition_counts[condition_id],
                "channel": channel,
                "scope": scope,
                **metric.report(),
            }
        )
    _write_csv(output_dir / "metrics_by_condition.csv", metric_rows)
    _write_csv(output_dir / "per_sample_metrics.csv", per_sample_rows)
    (output_dir / "prediction_examples.json").write_text(
        json.dumps(examples, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = {
        "dataset": "VTaC canonical test",
        "suite": "phase8_long_gap_stress",
        "base_eligible_sample_count_ge_2_beats": len(base_eligible),
        "condition_count": len(conditions),
        "conditions": [asdict(condition) for condition in conditions],
        "evaluated_sample_count_by_condition": condition_counts,
        "primary_metric_scope": "masked_samples",
        "oracle_time_gap_note": (
            "秒级区间按完整信号得到的R峰中心时间映射到重采样beat；"
            "不是先破坏原始信号再重新检测R峰的end-to-end测试。"
        ),
        "mask_manifest": str(manifest_path),
        "metrics": metric_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
