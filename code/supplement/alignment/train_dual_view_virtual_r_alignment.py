#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-1 embedding alignment with true PPG-only virtual-R beats.

Teacher target is always the dual-modal CSFM embedding.  The student is trained
with two views from the same record:
  1) ECG+PPG cached beats.
  2) PPG-only beats re-segmented by a frozen PPG-anchor distance network.

This stage only aligns embeddings.  It does not train the foundation
reconstruction/prediction heads.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ecg_ppg_multitask_pretrain_orthogonal_modality_flexible import (
    ECGPPGMultiTaskTransformer,
)


def log_stage(msg: str, t0: Optional[float] = None) -> float:
    now = time.time()
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    if t0 is None:
        print(f"[{stamp}] {msg}", flush=True)
    else:
        print(f"[{stamp}] {msg} elapsed={now - t0:.1f}s", flush=True)
    return now


def zscore_1d(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - float(x.mean())) / (float(x.std()) + eps)


def resample_1d(x: np.ndarray, out_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size <= 1:
        return np.zeros(out_len, dtype=np.float32)
    src = np.linspace(0.0, 1.0, x.size, dtype=np.float32)
    dst = np.linspace(0.0, 1.0, out_len, dtype=np.float32)
    return np.interp(dst, src, x).astype(np.float32)


def pad_beats(
    beats: np.ndarray,
    time_sec: np.ndarray,
    max_beats: int,
    beat_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = np.zeros((max_beats, 2, beat_len), dtype=np.float32)
    t_out = np.zeros((max_beats,), dtype=np.float32)
    mask = np.zeros((max_beats,), dtype=np.bool_)
    n = min(int(beats.shape[0]), max_beats)
    if n > 0:
        out[:n] = beats[:n]
        t_out[:n] = time_sec[:n]
        mask[:n] = True
    return out, t_out, mask


def normalize_cached_beats(beats: np.ndarray) -> np.ndarray:
    beats = np.asarray(beats, dtype=np.float32)
    mean = beats.mean(axis=-1, keepdims=True)
    std = beats.std(axis=-1, keepdims=True)
    return (beats - mean) / (std + 1e-6)


def extract_embedding_array(value: object) -> Optional[np.ndarray]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float32).reshape(-1)
    if isinstance(value, (list, tuple)) and value:
        try:
            return np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return None
    if isinstance(value, dict):
        preferred_keys = (
            "embedding",
            "emb",
            "csfm_emb",
            "teacher_emb",
            "z",
            "feat",
            "features",
        )
        for key in preferred_keys:
            if key in value:
                out = extract_embedding_array(value[key])
                if out is not None:
                    return out
        for nested in value.values():
            out = extract_embedding_array(nested)
            if out is not None:
                return out
    return None


def load_teacher_memmap(matrix_path: str, rel_paths_path: str) -> Tuple[List[str], np.ndarray]:
    t0 = log_stage(f"[teacher-memmap] loading matrix={matrix_path} rel_paths={rel_paths_path}")
    emb_matrix = np.load(matrix_path, mmap_mode="r")
    with open(rel_paths_path, "r", encoding="utf-8") as f:
        keys = [ln.strip() for ln in f if ln.strip()]
    if emb_matrix.ndim != 2:
        raise ValueError(f"teacher matrix must be 2D, got {emb_matrix.shape}")
    if len(keys) > emb_matrix.shape[0]:
        raise ValueError(f"rel_paths has {len(keys)} rows but matrix has {emb_matrix.shape[0]}")
    emb_matrix = emb_matrix[: len(keys)]
    log_stage(
        f"[teacher-memmap] loaded rows={len(keys)} shape={emb_matrix.shape} "
        f"size={emb_matrix.nbytes / 1e9:.2f}GB",
        t0,
    )
    return keys, emb_matrix


def load_teacher_matrix(path: str) -> Tuple[List[str], np.ndarray]:
    t0 = log_stage(f"[teacher-table] loading {path}")
    obj = torch.load(path, map_location="cpu")
    log_stage(f"[teacher-table] torch.load done type={type(obj).__name__}", t0)
    if isinstance(obj, dict) and "table" in obj:
        log_stage(f"[teacher-table] using obj['table'] entries={len(obj['table'])}")
        obj = obj["table"]
    elif isinstance(obj, dict) and "embeddings" in obj:
        log_stage(f"[teacher-table] using obj['embeddings'] entries={len(obj['embeddings'])}")
        obj = obj["embeddings"]
    if not isinstance(obj, dict):
        raise TypeError(f"teacher table must be a dict, got {type(obj)!r}")

    first_key = None
    first_emb = None
    for k, v in obj.items():
        first_emb = extract_embedding_array(v)
        if first_emb is not None:
            first_key = str(k)
            break
    if first_emb is None:
        raise RuntimeError(f"no usable teacher embeddings found in {path}")

    emb_dim = int(first_emb.size)
    nrec = len(obj)
    keys: List[str] = []
    emb_matrix = np.empty((nrec, emb_dim), dtype=np.float32)
    skipped = 0
    row = 0
    t1 = log_stage(f"[teacher-table] converting entries={len(obj)} emb_dim={emb_dim} to dense matrix")
    for i, (k, v) in enumerate(obj.items(), start=1):
        emb = extract_embedding_array(v)
        if emb is None:
            skipped += 1
            continue
        if emb.size != emb_dim:
            skipped += 1
            continue
        keys.append(str(k))
        emb_matrix[row] = emb
        row += 1
        if i % 500000 == 0:
            gb = emb_matrix.nbytes / 1e9
            log_stage(f"[teacher-table] converted {i}/{len(obj)} rows={row} skipped={skipped} matrix={gb:.2f}GB", t1)
    if row == 0:
        raise RuntimeError(f"no usable teacher embeddings found in {path}")
    if row < nrec:
        emb_matrix = emb_matrix[:row]
    del obj
    log_stage(
        f"[teacher-table] loaded rows={len(keys)} skipped={skipped} "
        f"shape={emb_matrix.shape} size={emb_matrix.nbytes / 1e9:.2f}GB from {path}",
        t0,
    )
    return keys, emb_matrix


def lookup_teacher_embedding(
    table: Dict[str, torch.Tensor],
    npz_path: Path,
    cache_root: Path,
) -> Optional[torch.Tensor]:
    keys = [
        str(npz_path),
        npz_path.as_posix(),
        str(npz_path.resolve()),
        npz_path.resolve().as_posix(),
        str(npz_path.relative_to(cache_root)),
        npz_path.relative_to(cache_root).as_posix(),
        npz_path.name,
    ]
    for key in keys:
        if key in table:
            return table[key]
    return None


class RawAnchorAlignmentDataset(Dataset):
    def __init__(
        self,
        cache_root: str,
        teacher_table_path: str,
        teacher_matrix_path: str = "",
        teacher_rel_paths_path: str = "",
        max_beats: int = 25,
        beat_len: int = 128,
        limit: int = 0,
        verify_paths: bool = False,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.max_beats = int(max_beats)
        self.beat_len = int(beat_len)
        ds_t0 = log_stage(f"[dataset] init cache_root={self.cache_root}")
        if teacher_matrix_path and teacher_rel_paths_path:
            self.teacher_keys, self.emb_matrix = load_teacher_memmap(
                teacher_matrix_path, teacher_rel_paths_path
            )
        else:
            self.teacher_keys, self.emb_matrix = load_teacher_matrix(teacher_table_path)

        build_t0 = log_stage(
            f"[dataset] building item list from teacher keys count={len(self.teacher_keys)} "
            f"verify_paths={int(verify_paths)}"
        )
        self.items: List[Tuple[Path, int]] = []
        missing_files = 0
        for i, key in enumerate(self.teacher_keys, start=1):
            p = Path(key)
            if not p.is_absolute():
                p = self.cache_root / key
            if verify_paths and not p.exists():
                missing_files += 1
                if missing_files <= 20:
                    log_stage(f"[dataset] missing file skipped: {p}")
                if i % 500000 == 0:
                    log_stage(
                        f"[dataset] checked {i}/{len(self.teacher_keys)} "
                        f"items={len(self.items)} missing_files={missing_files}",
                        build_t0,
                    )
                continue
            self.items.append((p, i - 1))
            if limit and limit > 0 and len(self.items) >= int(limit):
                break
            if i % 500000 == 0:
                log_stage(
                    f"[dataset] checked {i}/{len(self.teacher_keys)} "
                    f"items={len(self.items)} missing_files={missing_files}",
                    build_t0,
                )
        if not self.items:
            raise RuntimeError(
                f"no teacher-table paths available for cache_root={self.cache_root}; "
                f"teacher_count={len(self.teacher_keys)}"
            )
        print(
            f"[dataset] usable={len(self.items)} from_teacher_keys=1 "
            f"missing_files={missing_files} "
            f"emb_matrix_shape={self.emb_matrix.shape} "
            f"cache_root={self.cache_root}"
            f"{' limit=' + str(limit) if limit and limit > 0 else ''}",
            flush=True,
        )
        log_stage("[dataset] init done", ds_t0)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        path, emb_row = self.items[idx]
        with np.load(path, allow_pickle=False) as data:
            beats = normalize_cached_beats(data["beats"])
            time_sec = np.asarray(data["time_sec"], dtype=np.float32)
            beats_pad, time_pad, beat_mask = pad_beats(
                beats, time_sec, self.max_beats, self.beat_len
            )

            raw_ppg = np.asarray(data["raw_ppg"], dtype=np.float32)
            fs = float(np.asarray(data["fs"]).reshape(-1)[0])
            ppg_maxslope_idx = np.asarray(data["ppg_maxslope_idx"], dtype=np.int64)

        return {
            "path": str(path),
            "beats_both": torch.from_numpy(beats_pad),
            "time_both": torch.from_numpy(time_pad),
            "mask_both": torch.from_numpy(beat_mask),
            "teacher_emb": torch.from_numpy(self.emb_matrix[emb_row].copy()),
            "raw_ppg": raw_ppg,
            "fs": fs,
            "ppg_maxslope_idx": ppg_maxslope_idx,
        }


def collate_alignment(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "path": [b["path"] for b in batch],
        "beats_both": torch.stack([b["beats_both"] for b in batch], dim=0),
        "time_both": torch.stack([b["time_both"] for b in batch], dim=0),
        "mask_both": torch.stack([b["mask_both"] for b in batch], dim=0),
        "teacher_emb": torch.stack([b["teacher_emb"] for b in batch], dim=0),
        "raw_ppg": [b["raw_ppg"] for b in batch],
        "fs": [b["fs"] for b in batch],
        "ppg_maxslope_idx": [b["ppg_maxslope_idx"] for b in batch],
    }


def build_anchor_windows(
    raw_ppg: np.ndarray,
    fs: float,
    ppg_idx: np.ndarray,
    target_fs: float,
    w_left_sec: float,
    w_right_sec: float,
) -> Tuple[np.ndarray, np.ndarray]:
    raw_ppg = np.asarray(raw_ppg, dtype=np.float32)
    ppg_idx = np.asarray(ppg_idx, dtype=np.int64)
    out_len = int(round((w_left_sec + w_right_sec) * target_fs))
    wins: List[np.ndarray] = []
    valid_idx: List[int] = []
    left = int(round(w_left_sec * fs))
    right = int(round(w_right_sec * fs))
    for idx in ppg_idx:
        lo = int(idx) - left
        hi = int(idx) + right
        if lo < 0 or hi > raw_ppg.size or hi <= lo + 4:
            continue
        seg = zscore_1d(resample_1d(raw_ppg[lo:hi], out_len))
        der = np.gradient(seg).astype(np.float32)
        wins.append(np.stack([seg, der], axis=0))
        valid_idx.append(int(idx))
    if not wins:
        return np.zeros((0, 2, out_len), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(wins, axis=0).astype(np.float32), np.asarray(valid_idx, dtype=np.int64)


@torch.no_grad()
def predict_virtual_r_idx(
    anchor_net: nn.Module,
    raw_ppg: np.ndarray,
    fs: float,
    ppg_idx: np.ndarray,
    device: torch.device,
    target_fs: float,
    w_left_sec: float,
    w_right_sec: float,
    jitter_prob: float,
    jitter_std_sec: float,
    jitter_max_sec: float,
) -> np.ndarray:
    wins, valid_ppg_idx = build_anchor_windows(
        raw_ppg, fs, ppg_idx, target_fs, w_left_sec, w_right_sec
    )
    if wins.shape[0] < 2:
        return np.zeros((0,), dtype=np.float32)

    x = torch.from_numpy(wins).to(device=device, dtype=torch.float32)
    delta = anchor_net(x)["delta"].detach().cpu().numpy().astype(np.float32)
    if jitter_prob > 0.0 and jitter_std_sec > 0.0:
        use = np.random.rand(delta.size) < float(jitter_prob)
        eps = np.random.normal(0.0, float(jitter_std_sec), size=delta.size).astype(np.float32)
        eps = np.clip(eps, -float(jitter_max_sec), float(jitter_max_sec))
        delta = delta + eps * use.astype(np.float32)

    virtual_r = valid_ppg_idx.astype(np.float32) - delta * float(fs)
    virtual_r = virtual_r[np.isfinite(virtual_r)]
    virtual_r = virtual_r[(virtual_r > 1) & (virtual_r < len(raw_ppg) - 2)]
    return np.unique(np.round(np.sort(virtual_r)).astype(np.int64)).astype(np.float32)


def segment_ppg_by_virtual_r(
    raw_ppg: np.ndarray,
    fs: float,
    virtual_r_idx: np.ndarray,
    max_beats: int,
    beat_len: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    anchors = np.asarray(virtual_r_idx, dtype=np.float32)
    anchors = anchors[np.isfinite(anchors)]
    anchors = np.sort(anchors)
    if anchors.size < 3:
        return None

    beats: List[np.ndarray] = []
    times: List[float] = []
    for i in range(1, anchors.size - 1):
        left = int(round((anchors[i - 1] + anchors[i]) * 0.5))
        right = int(round((anchors[i] + anchors[i + 1]) * 0.5))
        if left < 0 or right > len(raw_ppg) or right <= left + 4:
            continue
        ppg = zscore_1d(resample_1d(raw_ppg[left:right], beat_len))
        beat = np.zeros((2, beat_len), dtype=np.float32)
        beat[1] = ppg
        beats.append(beat)
        times.append(float(anchors[i] / fs))

    if len(beats) == 0:
        return None
    if len(beats) > max_beats:
        start = random.randint(0, len(beats) - max_beats)
        beats = beats[start : start + max_beats]
        times = times[start : start + max_beats]
    return pad_beats(
        np.stack(beats, axis=0).astype(np.float32),
        np.asarray(times, dtype=np.float32),
        max_beats,
        beat_len,
    )


def fallback_ppg_from_cached(
    beats_both: torch.Tensor,
    time_both: torch.Tensor,
    mask_both: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = beats_both.detach().cpu().numpy().copy()
    b[:, 0, :] = 0.0
    return (
        b.astype(np.float32),
        time_both.detach().cpu().numpy().astype(np.float32),
        mask_both.detach().cpu().numpy().astype(np.bool_),
    )


def build_ppg_virtual_batch(
    batch: Dict[str, object],
    anchor_net: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    beats_out: List[np.ndarray] = []
    time_out: List[np.ndarray] = []
    mask_out: List[np.ndarray] = []
    fallback_count = 0

    raw_list = batch["raw_ppg"]
    fs_list = batch["fs"]
    ppg_idx_list = batch["ppg_maxslope_idx"]
    beats_both = batch["beats_both"]
    time_both = batch["time_both"]
    mask_both = batch["mask_both"]

    for i, (raw_ppg, fs, ppg_idx) in enumerate(zip(raw_list, fs_list, ppg_idx_list)):
        virtual_r = predict_virtual_r_idx(
            anchor_net=anchor_net,
            raw_ppg=raw_ppg,
            fs=float(fs),
            ppg_idx=ppg_idx,
            device=device,
            target_fs=args.anchor_target_fs,
            w_left_sec=args.anchor_w_left_sec,
            w_right_sec=args.anchor_w_right_sec,
            jitter_prob=args.virtual_r_jitter_prob,
            jitter_std_sec=args.virtual_r_jitter_std_sec,
            jitter_max_sec=args.virtual_r_jitter_max_sec,
        )
        packed = segment_ppg_by_virtual_r(
            raw_ppg=raw_ppg,
            fs=float(fs),
            virtual_r_idx=virtual_r,
            max_beats=args.max_beats,
            beat_len=args.beat_len,
        )
        if packed is None:
            packed = fallback_ppg_from_cached(beats_both[i], time_both[i], mask_both[i])
            fallback_count += 1
        b, t, m = packed
        beats_out.append(b)
        time_out.append(t)
        mask_out.append(m)

    return (
        torch.from_numpy(np.stack(beats_out, axis=0)).to(device),
        torch.from_numpy(np.stack(time_out, axis=0)).to(device),
        torch.from_numpy(np.stack(mask_out, axis=0)).to(device),
        fallback_count,
    )


def load_anchor_net(args: argparse.Namespace, device: torch.device) -> nn.Module:
    t0 = log_stage(f"[anchor-net] loading code_dir={args.anchor_code_dir} ckpt={args.anchor_ckpt}")
    sys.path.insert(0, args.anchor_code_dir)
    from ppg_anchor_distance_net import PPGAnchorDistanceNet  # type: ignore

    model = PPGAnchorDistanceNet(
        in_channels=2,
        width=args.anchor_width,
        kernel_size=args.anchor_kernel_size,
        dilations=tuple(args.anchor_dilations),
        dropout=args.anchor_dropout,
        predict_logvar=True,
    )
    ckpt = torch.load(args.anchor_ckpt, map_location="cpu")
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log_stage("[anchor-net] ready", t0)
    return model


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def cosine_distill_loss(student_z: torch.Tensor, teacher_z: torch.Tensor) -> torch.Tensor:
    student_z = F.normalize(student_z, dim=-1)
    teacher_z = F.normalize(teacher_z, dim=-1)
    return (1.0 - (student_z * teacher_z).sum(dim=-1)).mean()


def make_student(args: argparse.Namespace, teacher_dim: int) -> Tuple[nn.Module, nn.Module]:
    model = ECGPPGMultiTaskTransformer(
        in_channels=2,
        beat_len=args.beat_len,
        morph_dim=6,
        hrv_dim=3,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        phase_tokens=args.phase_tokens,
        num_input_modes=2,
    )
    proj = ProjectionHead(args.d_model, teacher_dim)
    return model, proj


def count_parameters(module: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def save_alignment_checkpoint(
    path: str,
    model: nn.Module,
    proj: nn.Module,
    args: argparse.Namespace,
    teacher_dim: int,
    epoch: int,
    step: int,
    global_step: int,
    loss: float,
    best_loss: float,
    tag: str,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "proj_state": proj.state_dict(),
            "args": vars(args),
            "teacher_dim": teacher_dim,
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "loss": loss,
            "best_loss": best_loss,
            "tag": tag,
        },
        path,
    )
    print(
        f"[save-{tag}] {path} epoch={epoch} step={step} "
        f"global_step={global_step} loss={loss:.6f}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", default="/data-ai/sl20200894/prepared_pretrain_npz/ecg_ppg_foundation_raw_anchor")
    p.add_argument("--teacher-table", default="/data-ai/sl20200894/CSFM/csfm_base_emb_table.pt")
    p.add_argument("--teacher-matrix", default="")
    p.add_argument("--teacher-rel-paths", default="")
    p.add_argument("--anchor-code-dir", default="/data-ai/sl20200894/Code/ppg_anchor_distance_split_bundle_20260627")
    p.add_argument("--anchor-ckpt", default="/data-ai/sl20200894/Code/ppg_anchor_distance_split_bundle_20260627/ppg_anchor_distance_net_subject_split.pt")
    p.add_argument("--save-path", default="/data-ai/sl20200894/Code/foundation_embedding_alignment_warmup_bundle_20260627/dual_view_virtual_r_alignment.pt")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--verify-paths", action="store_true")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--w-both", type=float, default=0.5)
    p.add_argument("--w-ppg", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument(
        "--save-every-steps",
        type=int,
        default=0,
        help="Save an additional checkpoint every N global optimization steps; 0 disables.",
    )

    p.add_argument("--beat-len", type=int, default=128)
    p.add_argument("--max-beats", type=int, default=25)
    # Keep the warmup student at the same scale as the original foundation
    # training script; single-modality support should not shrink the backbone.
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--nhead", type=int, default=12)
    p.add_argument("--num-layers", type=int, default=9)
    p.add_argument("--dim-feedforward", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--phase-tokens", type=int, default=8)

    p.add_argument("--anchor-target-fs", type=float, default=125.0)
    p.add_argument("--anchor-w-left-sec", type=float, default=0.6)
    p.add_argument("--anchor-w-right-sec", type=float, default=0.4)
    p.add_argument("--anchor-width", type=int, default=64)
    p.add_argument("--anchor-kernel-size", type=int, default=7)
    p.add_argument("--anchor-dilations", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--anchor-dropout", type=float, default=0.1)

    p.add_argument("--virtual-r-jitter-prob", type=float, default=0.8)
    p.add_argument("--virtual-r-jitter-std-sec", type=float, default=0.02)
    p.add_argument("--virtual-r-jitter-max-sec", type=float, default=0.06)
    return p.parse_args()


def main() -> None:
    main_t0 = log_stage("[main] start")
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    print(
        "virtual_r: anchor_net=true "
        f"jitter_prob={args.virtual_r_jitter_prob} "
        f"jitter_std={args.virtual_r_jitter_std_sec}s "
        f"jitter_max={args.virtual_r_jitter_max_sec}s",
        flush=True,
    )

    log_stage("[main] creating dataset")
    ds = RawAnchorAlignmentDataset(
        cache_root=args.cache_root,
        teacher_table_path=args.teacher_table,
        teacher_matrix_path=args.teacher_matrix,
        teacher_rel_paths_path=args.teacher_rel_paths,
        max_beats=args.max_beats,
        beat_len=args.beat_len,
        limit=args.limit,
        verify_paths=args.verify_paths,
    )
    first_teacher_dim = int(ds.emb_matrix.shape[1])
    log_stage(f"[main] creating dataloader samples={len(ds)} batch_size={args.batch_size} workers={args.num_workers}")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_alignment,
    )

    log_stage(f"[main] creating student teacher_dim={first_teacher_dim}")
    model, proj = make_student(args, first_teacher_dim)
    model_total, model_trainable = count_parameters(model)
    proj_total, proj_trainable = count_parameters(proj)
    print(
        "student_config: "
        f"d_model={args.d_model} nhead={args.nhead} "
        f"num_layers={args.num_layers} dim_feedforward={args.dim_feedforward} "
        f"phase_tokens={args.phase_tokens}",
        flush=True,
    )
    print(
        f"student parameters: total={model_total:,} trainable={model_trainable:,}; "
        f"projection parameters: total={proj_total:,} trainable={proj_trainable:,}",
        flush=True,
    )
    model.to(device)
    proj.to(device)
    anchor_net = load_anchor_net(args, device)

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(proj.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    log_stage("[main] training loop ready", main_t0)

    best_loss = float("inf")
    steps_per_epoch = len(loader)
    for epoch in range(1, args.epochs + 1):
        epoch_t0 = log_stage(f"[epoch {epoch:03d}] start")
        model.train()
        proj.train()
        total = 0.0
        n = 0
        fallback_total = 0

        wait_t0 = time.time()
        for step, batch in enumerate(loader, start=1):
            if step == 1:
                log_stage(f"[epoch {epoch:03d}] first batch received", wait_t0)
            beats_both = batch["beats_both"].to(device, non_blocking=True)
            time_both = batch["time_both"].to(device, non_blocking=True)
            mask_both = batch["mask_both"].to(device, non_blocking=True)
            teacher = batch["teacher_emb"].to(device, non_blocking=True)

            beats_ppg, time_ppg, mask_ppg, fallback_count = build_ppg_virtual_batch(
                batch, anchor_net, device, args
            )
            fallback_total += fallback_count

            bsz = beats_both.shape[0]
            both_modality_mask = torch.ones((bsz, 2), dtype=torch.bool, device=device)
            ppg_modality_mask = torch.tensor([[False, True]], device=device).repeat(bsz, 1)
            both_input_mode = torch.zeros((bsz,), dtype=torch.long, device=device)
            ppg_input_mode = torch.ones((bsz,), dtype=torch.long, device=device)

            out_both = model(
                beats_both,
                beat_mask=mask_both,
                time_sec=time_both,
                modality_mask=both_modality_mask,
                input_mode=both_input_mode,
            )
            out_ppg = model(
                beats_ppg,
                beat_mask=mask_ppg,
                time_sec=time_ppg,
                modality_mask=ppg_modality_mask,
                input_mode=ppg_input_mode,
            )

            loss_both = cosine_distill_loss(proj(out_both["cls"]), teacher)
            loss_ppg = cosine_distill_loss(proj(out_ppg["cls"]), teacher)
            loss = args.w_both * loss_both + args.w_ppg * loss_ppg

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(proj.parameters()), 1.0)
            opt.step()

            total += float(loss.item())
            n += 1
            global_step = (epoch - 1) * steps_per_epoch + step
            if step % args.log_every == 0:
                elapsed = time.time() - epoch_t0
                sec_per_step = elapsed / max(step, 1)
                eta = sec_per_step * max(steps_per_epoch - step, 0)
                pct = 100.0 * step / max(steps_per_epoch, 1)
                print(
                    f"epoch={epoch:03d} step={step:06d}/{steps_per_epoch:06d} "
                    f"pct={pct:5.1f}% eta={eta/3600:.2f}h "
                    f"loss={total / max(n, 1):.5f} "
                    f"both={float(loss_both.item()):.5f} "
                    f"ppg={float(loss_ppg.item()):.5f} "
                    f"fallback={fallback_total}",
                    flush=True,
                )
            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                save_base = Path(args.save_path)
                step_path = save_base.with_name(
                    f"{save_base.stem}_step{global_step:08d}{save_base.suffix}"
                )
                save_alignment_checkpoint(
                    str(step_path),
                    model,
                    proj,
                    args,
                    first_teacher_dim,
                    epoch,
                    step,
                    global_step,
                    float(loss.item()),
                    best_loss,
                    tag="step",
                )

        epoch_loss = total / max(n, 1)
        log_stage(f"epoch={epoch:03d} train_loss={epoch_loss:.6f} fallback={fallback_total}", epoch_t0)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_alignment_checkpoint(
                args.save_path,
                model,
                proj,
                args,
                first_teacher_dim,
                epoch,
                steps_per_epoch,
                epoch * steps_per_epoch,
                epoch_loss,
                best_loss,
                tag="best",
            )


if __name__ == "__main__":
    main()
