# -*- coding: utf-8 -*-
"""
ECG+PPG beat-level Transformer pretraining starter.

This file implements a phase-aware ECG+PPG beat Transformer:
  beat waveform -> intra-beat phase tokens -> phase attention + beat attention
  -> reliability-gated pooling -> reconstruction / PAT / morphology / HRV heads.

Expected prepared sample format for real data:
  beats:      (K, C, L) float32, C=2 for ECG+PPG or C=4 for ECG+PPG_IR/RD/OT
  time_sec:   (K,) float32 cumulative R-peak or beat-center timestamps
  pat:        (K,) float32, ECG R-peak to PPG foot/peak delay in seconds
  morph:      (K, M) float32 PPG morphology targets
  hrv:        (H,) float32 sequence-level HRV/HR targets

If --data-dir is omitted, the script runs a synthetic smoke test.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class Batch:
    # DataLoader 取出的 batch 会先被整理成这个结构。
    # 这样训练代码里不用反复写 items["xxx"]，读起来更清楚。
    beats: torch.Tensor
    time_sec: torch.Tensor
    beat_mask: torch.Tensor
    pretrain_mask: torch.Tensor
    pat: torch.Tensor
    pat_mask: torch.Tensor
    morph: torch.Tensor
    morph_mask: torch.Tensor
    hrv: torch.Tensor
    hrv_mask: torch.Tensor
    recon_target: Optional[torch.Tensor] = None
    modality_mask: Optional[torch.Tensor] = None
    # Backward-compatible name: modality_status now means input_mode
    # (0=both, 1=ppg_only), not ECG/PPG token identity.
    modality_status: Optional[torch.Tensor] = None


def zscore_per_beat(beats: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize each channel within each beat."""
    # 对每一个 beat、每一个通道分别做标准化。
    # 这样模型更关注 ECG/PPG 的形状，而不是原始幅值尺度。
    mean = beats.mean(axis=-1, keepdims=True)
    std = beats.std(axis=-1, keepdims=True)
    return ((beats - mean) / (std + eps)).astype(np.float32)


def pad_or_crop_1d(x: np.ndarray, length: int, value: float = 0.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    out = np.full((length,), value, dtype=np.float32)
    n = min(len(x), length)
    out[:n] = x[:n]
    return out


def pad_or_crop_2d(x: np.ndarray, rows: int, cols: int, value: float = 0.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    out = np.full((rows, cols), value, dtype=np.float32)
    r = min(x.shape[0], rows)
    c = min(x.shape[1], cols)
    out[:r, :c] = x[:r, :c]
    return out


def sample_pretrain_mask(
    beat_mask: np.ndarray,
    mask_ratio: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    # 从有效 beat 中随机挑一部分作为 [MASK]。
    # 之后模型需要根据上下文重建这些被遮住的 beat。
    valid = np.where(beat_mask)[0]
    out = np.zeros_like(beat_mask, dtype=bool)
    if len(valid) == 0:
        return out
    n_mask = max(1, int(round(len(valid) * mask_ratio)))
    chosen = rng.choice(valid, size=min(n_mask, len(valid)), replace=False)
    out[chosen] = True
    return out


class PreparedNPZBeatDataset(Dataset):
    """
    Loads prepared .npz files.

    Required arrays per file:
      beats, time_sec, pat, morph, hrv

    This keeps raw ECG/PPG fiducial detection outside the model file. You can
    connect your existing extract_cycles_4ch-style preprocessing by exporting
    one .npz per record/window.
    """

    def __init__(
        self,
        data_dir: str | os.PathLike,
        max_beats: int = 64,
        in_channels: int = 2,
        beat_len: int = 256,
        morph_dim: int = 6,
        hrv_dim: int = 3,
        mask_ratio: float = 0.15,
        seed: int = 0,
        normalize: bool = True,
    ):
        # 递归读取，方便把不同数据源放在同一个 data_dir 的子目录里。
        self.files = sorted(Path(data_dir).rglob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")
        self.max_beats = max_beats
        self.in_channels = in_channels
        self.beat_len = beat_len
        self.morph_dim = morph_dim
        self.hrv_dim = hrv_dim
        self.mask_ratio = mask_ratio
        self.rng = np.random.RandomState(seed)
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 一个 .npz 对应一条记录/一个窗口。
        # 这里假设你已经在预处理阶段完成了 R 峰检测、beat 切分、
        # PAT 和形态学/HRV 辅助标签计算。
        with np.load(self.files[idx], allow_pickle=False) as item:
            beats = item["beats"].astype(np.float32)
            time_sec = item["time_sec"].astype(np.float32)
            pat = item["pat"].astype(np.float32)
            morph = item["morph"].astype(np.float32)
            hrv = item["hrv"].astype(np.float32)

        beats = beats[: self.max_beats, : self.in_channels, : self.beat_len]
        if self.normalize:
            beats = zscore_per_beat(beats)

        k = beats.shape[0]
        # beat_mask=True 表示这个位置是真实 beat；
        # False 表示为了凑成固定长度 max_beats 而补出来的 padding。
        beat_mask = np.zeros(self.max_beats, dtype=bool)
        beat_mask[:k] = True

        # Transformer 训练时通常要求 batch 内 shape 一致。
        # 所以短序列补 0，长序列截断到 max_beats。
        beats_pad = np.zeros((self.max_beats, self.in_channels, self.beat_len), dtype=np.float32)
        beats_pad[:k, : beats.shape[1], : beats.shape[2]] = beats

        time_pad = pad_or_crop_1d(time_sec, self.max_beats)
        pat_pad = pad_or_crop_1d(pat, self.max_beats)
        morph_pad = pad_or_crop_2d(morph, self.max_beats, self.morph_dim)
        hrv_pad = pad_or_crop_1d(hrv, self.hrv_dim)

        # 有些辅助标签可能算不出来，用 NaN 表示。
        # loss 里只会计算 mask=True 的位置，NaN 标签会被自动跳过。
        pat_mask = beat_mask & np.isfinite(pat_pad)
        morph_mask = beat_mask[:, None] & np.isfinite(morph_pad)
        hrv_mask = np.isfinite(hrv_pad)

        pat_pad = np.nan_to_num(pat_pad, nan=0.0)
        morph_pad = np.nan_to_num(morph_pad, nan=0.0)
        hrv_pad = np.nan_to_num(hrv_pad, nan=0.0)

        pretrain_mask = sample_pretrain_mask(beat_mask, self.mask_ratio, self.rng)

        return {
            "beats": torch.from_numpy(beats_pad),
            "time_sec": torch.from_numpy(time_pad),
            "beat_mask": torch.from_numpy(beat_mask),
            "pretrain_mask": torch.from_numpy(pretrain_mask),
            "pat": torch.from_numpy(pat_pad[:, None]),
            "pat_mask": torch.from_numpy(pat_mask[:, None]),
            "morph": torch.from_numpy(morph_pad),
            "morph_mask": torch.from_numpy(morph_mask),
            "hrv": torch.from_numpy(hrv_pad),
            "hrv_mask": torch.from_numpy(hrv_mask),
        }


class SyntheticECGPPGDataset(Dataset):
    """Small synthetic dataset used only to smoke-test the code path."""

    def __init__(
        self,
        n: int = 128,
        max_beats: int = 32,
        in_channels: int = 2,
        beat_len: int = 256,
        morph_dim: int = 6,
        hrv_dim: int = 3,
        mask_ratio: float = 0.15,
        seed: int = 0,
    ):
        self.n = n
        self.max_beats = max_beats
        self.in_channels = in_channels
        self.beat_len = beat_len
        self.morph_dim = morph_dim
        self.hrv_dim = hrv_dim
        self.mask_ratio = mask_ratio
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 合成数据只用于检查代码能不能跑通。
        # 它不是生理真实数据，不应该用它评估模型好坏。
        rng = np.random.RandomState(idx)
        k = rng.randint(self.max_beats // 2, self.max_beats + 1)
        rr = rng.normal(0.82, 0.06, size=k).clip(0.55, 1.25).astype(np.float32)
        time_sec = np.cumsum(rr) - rr[0]

        t = np.linspace(0.0, 1.0, self.beat_len, endpoint=False, dtype=np.float32)
        beats = np.zeros((k, self.in_channels, self.beat_len), dtype=np.float32)
        pat = rng.normal(0.22, 0.025, size=k).astype(np.float32)
        amp = rng.normal(1.0, 0.10, size=k).astype(np.float32)

        for i in range(k):
            # 简单造一个 ECG R 波和一个延迟出现的 PPG 波。
            # PAT 越大，PPG 主峰越晚出现。
            ecg = np.exp(-0.5 * ((t - 0.24) / 0.018) ** 2)
            ecg -= 0.25 * np.exp(-0.5 * ((t - 0.20) / 0.014) ** 2)
            ppg_center = min(0.24 + pat[i], 0.88)
            ppg = amp[i] * np.exp(-0.5 * ((t - ppg_center) / 0.08) ** 2)
            ppg += 0.25 * amp[i] * np.exp(-0.5 * ((t - ppg_center - 0.16) / 0.11) ** 2)
            beats[i, 0] = ecg + rng.normal(0.0, 0.015, size=self.beat_len)
            beats[i, 1] = ppg + rng.normal(0.0, 0.015, size=self.beat_len)
            for ch in range(2, self.in_channels):
                beats[i, ch] = ppg * (1.0 + 0.05 * ch) + rng.normal(0.0, 0.02, size=self.beat_len)

        beats = zscore_per_beat(beats)
        beat_mask = np.zeros(self.max_beats, dtype=bool)
        beat_mask[:k] = True

        beats_pad = np.zeros((self.max_beats, self.in_channels, self.beat_len), dtype=np.float32)
        beats_pad[:k] = beats
        time_pad = pad_or_crop_1d(time_sec, self.max_beats)
        pat_pad = pad_or_crop_1d(pat, self.max_beats)

        morph = np.zeros((k, self.morph_dim), dtype=np.float32)
        morph[:, 0] = amp
        morph[:, 1] = pat
        morph[:, 2] = rr
        morph[:, 3:] = rng.normal(0.0, 0.1, size=(k, max(self.morph_dim - 3, 0)))
        morph_pad = pad_or_crop_2d(morph, self.max_beats, self.morph_dim)

        hrv = np.array(
            [
                rr.std(),
                np.sqrt(np.mean(np.diff(rr) ** 2)) if k > 1 else 0.0,
                60.0 / rr.mean(),
            ],
            dtype=np.float32,
        )
        hrv = pad_or_crop_1d(hrv, self.hrv_dim)

        pretrain_mask = sample_pretrain_mask(beat_mask, self.mask_ratio, self.rng)

        return {
            "beats": torch.from_numpy(beats_pad),
            "time_sec": torch.from_numpy(time_pad),
            "beat_mask": torch.from_numpy(beat_mask),
            "pretrain_mask": torch.from_numpy(pretrain_mask),
            "pat": torch.from_numpy(pat_pad[:, None]),
            "pat_mask": torch.from_numpy(beat_mask[:, None]),
            "morph": torch.from_numpy(morph_pad),
            "morph_mask": torch.from_numpy(beat_mask[:, None].repeat(self.morph_dim, axis=1)),
            "hrv": torch.from_numpy(hrv),
            "hrv_mask": torch.ones(self.hrv_dim, dtype=torch.bool),
        }


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 2):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BeatEncoder(nn.Module):
    def __init__(self, in_channels: int, d_model: int):
        super().__init__()
        # 每个 beat 先经过 1D CNN，压缩成一个 d_model 维 token。
        # 输入:  (B*K, C, L)
        # 输出:  (B*K, d_model)
        self.net = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=9, stride=2),
            ConvBlock(32, 64, kernel_size=7, stride=2),
            ConvBlock(64, 128, kernel_size=5, stride=2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(128, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x).squeeze(-1)
        return self.proj(x)


class PhaseAwareBeatEncoder(nn.Module):
    def __init__(self, in_channels: int, d_model: int, phase_tokens: int = 8):
        super().__init__()
        self.phase_tokens = int(phase_tokens)
        if self.phase_tokens < 1:
            raise ValueError("phase_tokens must be >= 1")
        self.net = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=9, stride=2),
            ConvBlock(32, 64, kernel_size=7, stride=2),
            ConvBlock(64, 128, kernel_size=5, stride=2),
        )
        self.phase_pool = nn.AdaptiveAvgPool1d(self.phase_tokens)
        self.proj = nn.Linear(128, d_model)
        self.phase_embed = nn.Parameter(torch.zeros(1, self.phase_tokens, d_model))
        nn.init.normal_(self.phase_embed, std=0.02)

    def forward_phase(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        phase_feat = self.phase_pool(feat).transpose(1, 2)
        return self.proj(phase_feat) + self.phase_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_phase(x).mean(dim=1)


class OrthogonalBeatPhaseBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.phase_norm = nn.LayerNorm(d_model)
        self.phase_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.beat_norm = nn.LayerNorm(d_model)
        self.beat_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, beat_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, K, P, D), where K is beat index and P is intra-beat phase index.
        b, k, p, d = x.shape

        phase_x = self.phase_norm(x).reshape(b * k, p, d)
        phase_y, _ = self.phase_attn(phase_x, phase_x, phase_x, need_weights=False)
        x = x + phase_y.reshape(b, k, p, d)

        beat_x = self.beat_norm(x).permute(0, 2, 1, 3).reshape(b * p, k, d)
        beat_pad = (~beat_mask).unsqueeze(1).expand(b, p, k).reshape(b * p, k)
        beat_y, _ = self.beat_attn(beat_x, beat_x, beat_x, key_padding_mask=beat_pad, need_weights=False)
        beat_y = beat_y.reshape(b, p, k, d).permute(0, 2, 1, 3)
        x = x + beat_y

        return x + self.ffn(self.ffn_norm(x))


class ReliabilityGatedPooling(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        hidden = max(d_model // 2, 32)
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, beat_z: torch.Tensor, beat_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(beat_z).squeeze(-1)
        logits = logits.masked_fill(~beat_mask, -1e4)
        weights = torch.softmax(logits, dim=1) * beat_mask.to(dtype=beat_z.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = (beat_z * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights


class CumulativeTimeEncoding(nn.Module):
    """Continuous sinusoidal encoding for real cumulative beat times."""

    def __init__(self, d_model: int, num_bands: int = 32, max_period: float = 10000.0):
        super().__init__()
        self.num_bands = num_bands
        freq = torch.exp(-math.log(max_period) * torch.arange(num_bands).float() / max(num_bands - 1, 1))
        self.register_buffer("freq", freq)
        self.proj = nn.Linear(num_bands * 2 + 1, d_model)

    def forward(self, time_sec: torch.Tensor) -> torch.Tensor:
        # time_sec: (B, K)
        # 这里不用简单的第几个 beat 作为位置，而是用真实累计时间。
        # 这样模型能看到 RR 间期不均匀带来的 HR/HRV 信息。
        angles = time_sec.unsqueeze(-1) * self.freq.view(1, 1, -1)
        feats = torch.cat([time_sec.unsqueeze(-1), torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.proj(feats)


class ECGPPGMultiTaskTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        beat_len: int = 256,
        morph_dim: int = 6,
        hrv_dim: int = 3,
        d_model: int = 320,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 640,
        dropout: float = 0.1,
        phase_tokens: int = 8,
        num_input_modes: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.beat_len = beat_len
        self.morph_dim = morph_dim
        self.hrv_dim = hrv_dim
        self.d_model = d_model
        self.phase_tokens = int(phase_tokens)

        if in_channels < 2:
            raise ValueError("modality-flexible model expects at least 2 channels: ECG + PPG")
        self.ecg_encoder = PhaseAwareBeatEncoder(in_channels=1, d_model=d_model, phase_tokens=self.phase_tokens)
        self.ppg_encoder = PhaseAwareBeatEncoder(in_channels=1, d_model=d_model, phase_tokens=self.phase_tokens)
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # 2) 给 beat token 加上真实时间位置编码
        self.time_encoding = CumulativeTimeEncoding(d_model=d_model)

        # 3) CLS 汇总整段序列；mask token 替换被遮住的 beat
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        self.input_mode_embed = nn.Embedding(num_input_modes, d_model)
        nn.init.zeros_(self.input_mode_embed.weight)

        # 4) Transformer 建模 beat 与 beat 之间的上下文关系
        self.blocks = nn.ModuleList(
            [
                OrthogonalBeatPhaseBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.reliability_pool = ReliabilityGatedPooling(d_model=d_model, dropout=dropout)

        # 5) 四个预训练任务 head
        recon_dim = in_channels * beat_len
        self.recon_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, recon_dim),
        )
        self.pat_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        self.morph_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, morph_dim),
        )
        self.hrv_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, hrv_dim))

        nn.init.normal_(self.mask_token, std=0.02)

    def encode_context(
        self,
        beats: torch.Tensor,
        time_sec: torch.Tensor,
        beat_mask: torch.Tensor,
        pretrain_mask: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
        input_mode: Optional[torch.Tensor] = None,
        modality_status: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b, k, c, l = beats.shape
        if c != self.in_channels or l != self.beat_len:
            raise ValueError(f"Expected beats (B,K,{self.in_channels},{self.beat_len}), got {tuple(beats.shape)}")

        ecg_x = beats[:, :, 0:1, :].reshape(b * k, 1, l)
        ppg_x = beats[:, :, 1:2, :].reshape(b * k, 1, l)
        ecg_z = self.ecg_encoder.forward_phase(ecg_x).reshape(b, k, self.phase_tokens, self.d_model)
        ppg_z = self.ppg_encoder.forward_phase(ppg_x).reshape(b, k, self.phase_tokens, self.d_model)
        z = torch.stack([ecg_z, ppg_z], dim=2)  # [B,K,2,P,D]

        if modality_mask is None:
            modality_mask = torch.ones(b, 2, dtype=torch.bool, device=beats.device)
        else:
            modality_mask = modality_mask.to(device=beats.device, dtype=torch.bool)
        gate_logits = self.fusion_gate(z.mean(dim=3)).squeeze(-1)  # [B,K,2]
        gate_logits = gate_logits.masked_fill(~modality_mask[:, None, :].expand(b, k, 2), -1e4)
        alpha = torch.softmax(gate_logits, dim=2)
        alpha = alpha.masked_fill(~modality_mask[:, None, :].expand(b, k, 2), 0.0)
        denom = alpha.sum(dim=2, keepdim=True).clamp_min(1e-6)
        alpha = alpha / denom
        phase_z = (alpha[:, :, :, None, None] * z).sum(dim=2)
        if input_mode is None:
            # Deprecated alias: modality_status is now semantically input_mode
            # (0=both, 1=ppg_only), applied after mask-aware fusion.
            input_mode = modality_status
        if input_mode is not None:
            input_mode = input_mode.to(device=beats.device, dtype=torch.long).clamp(
                0, self.input_mode_embed.num_embeddings - 1
            )
            phase_z = phase_z + self.input_mode_embed(input_mode).view(b, 1, 1, self.d_model)
        phase_z = phase_z + self.time_encoding(time_sec).unsqueeze(2)

        if pretrain_mask is not None:
            mask_token = self.mask_token.expand(b, k, self.phase_tokens, -1)
            phase_z = torch.where(pretrain_mask.unsqueeze(-1).unsqueeze(-1), mask_token, phase_z)

        for block in self.blocks:
            phase_z = block(phase_z, beat_mask)
        phase_z = self.norm(phase_z)

        beat_z = phase_z.mean(dim=2)
        cls_z, reliability = self.reliability_pool(beat_z, beat_mask)
        return {
            "cls": cls_z,
            "beat_z": beat_z,
            "phase_z": phase_z,
            "reliability": reliability,
        }

    def forward(
        self,
        beats: torch.Tensor,
        time_sec: torch.Tensor,
        beat_mask: torch.Tensor,
        pretrain_mask: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
        input_mode: Optional[torch.Tensor] = None,
        modality_status: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # beats: (B, K, C, L), beat_mask/pretrain_mask: (B, K), True means valid/masked.
        b, k, c, l = beats.shape
        if c != self.in_channels or l != self.beat_len:
            raise ValueError(f"Expected beats (B,K,{self.in_channels},{self.beat_len}), got {tuple(beats.shape)}")

        ctx = self.encode_context(
            beats,
            time_sec,
            beat_mask,
            pretrain_mask,
            modality_mask=modality_mask,
            input_mode=input_mode,
            modality_status=modality_status,
        )
        beat_z = ctx["beat_z"]
        cls_z = ctx["cls"]
        recon = self.recon_head(beat_z).reshape(b, k, self.in_channels, self.beat_len)
        return {
            "cls": cls_z,
            "beat_z": beat_z,
            "phase_z": ctx["phase_z"],
            "reliability": ctx["reliability"],
            "recon": recon,
            "pat": self.pat_head(beat_z),
            "morph": self.morph_head(beat_z),
            "hrv": self.hrv_head(cls_z),
        }

            # 被 mask 的 beat 不让模型直接看到原始 token。
            # 模型只能靠前后 beat 和另一个模态学会重建。

        # 在序列最前面拼一个 CLS token。
        # CLS 输出用于 HRV/HR 这类整段序列级任务。

        # PyTorch Transformer 的 key_padding_mask=True 表示“这个位置要忽略”。
        # CLS 永远有效；padding beat 会被注意力机制忽略。

        # recon/pat/morph 是 per-beat 输出；hrv 是 sequence-level 输出。


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # 只在 mask=True 的位置计算 MSE。
    # 这个函数同时用于 padding mask 和“某些标签缺失”的情况。
    mask = mask.to(dtype=torch.bool, device=pred.device)
    target = target.to(device=pred.device, dtype=pred.dtype)
    if mask.ndim < pred.ndim:
        for _ in range(pred.ndim - mask.ndim):
            mask = mask.unsqueeze(-1)
    mask = mask.expand_as(pred)
    if not mask.any():
        return pred.new_tensor(0.0)
    diff = pred[mask] - target[mask]
    return (diff * diff).mean()


def multitask_loss(
    out: Dict[str, torch.Tensor],
    batch: Batch,
    w_recon: float = 1.0,
    w_pat: float = 0.5,
    w_morph: float = 0.2,
    w_hrv: float = 0.2,
) -> Dict[str, torch.Tensor]:
    # Reconstruction: 只重建被随机 mask 的 beat。
    # PAT: 让每个 beat 的表示学习 ECG->PPG 的延迟。
    # Morphology: 让表示学习 PPG 波形形态。
    # HRV/HR: 用 CLS 表示学习整段序列的心率/变异性。
    recon_mask = batch.beat_mask & batch.pretrain_mask
    recon_target = batch.recon_target if batch.recon_target is not None else batch.beats
    loss_recon = masked_mse(out["recon"], recon_target, recon_mask)
    loss_pat = masked_mse(out["pat"], batch.pat, batch.pat_mask & batch.beat_mask.unsqueeze(-1))
    loss_morph = masked_mse(out["morph"], batch.morph, batch.morph_mask)
    loss_hrv = masked_mse(out["hrv"], batch.hrv, batch.hrv_mask)
    total = w_recon * loss_recon + w_pat * loss_pat + w_morph * loss_morph + w_hrv * loss_hrv
    return {
        "loss": total,
        "recon": loss_recon.detach(),
        "pat": loss_pat.detach(),
        "morph": loss_morph.detach(),
        "hrv": loss_hrv.detach(),
    }


def move_batch(items: Dict[str, torch.Tensor], device: torch.device) -> Batch:
    # 把 DataLoader 给出的 dict 移到 CPU/GPU，并整理成 Batch。
    recon_target = items.get("recon_target", items["beats"])
    modality_mask = items.get("modality_mask")
    modality_status = items.get("input_mode", items.get("modality_status"))
    return Batch(
        beats=items["beats"].to(device).float(),
        time_sec=items["time_sec"].to(device).float(),
        beat_mask=items["beat_mask"].to(device).bool(),
        pretrain_mask=items["pretrain_mask"].to(device).bool(),
        pat=items["pat"].to(device).float(),
        pat_mask=items["pat_mask"].to(device).bool(),
        morph=items["morph"].to(device).float(),
        morph_mask=items["morph_mask"].to(device).bool(),
        hrv=items["hrv"].to(device).float(),
        hrv_mask=items["hrv_mask"].to(device).bool(),
        recon_target=recon_target.to(device).float(),
        modality_mask=None if modality_mask is None else modality_mask.to(device).bool(),
        modality_status=None if modality_status is None else modality_status.to(device).long(),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()
    sums = {"loss": 0.0, "recon": 0.0, "pat": 0.0, "morph": 0.0, "hrv": 0.0}
    n = 0
    for items in loader:
        # 标准训练流程：forward -> loss -> backward -> update。
        batch = move_batch(items, device)
        out = model(batch.beats, batch.time_sec, batch.beat_mask, batch.pretrain_mask)
        losses = multitask_loss(out, batch)

        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = batch.beats.shape[0]
        for key in sums:
            sums[key] += float(losses[key].detach().cpu()) * bs
        n += bs
    return {key: val / max(n, 1) for key, val in sums.items()}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    sums = {"loss": 0.0, "recon": 0.0, "pat": 0.0, "morph": 0.0, "hrv": 0.0}
    n = 0
    for items in loader:
        batch = move_batch(items, device)
        out = model(batch.beats, batch.time_sec, batch.beat_mask, batch.pretrain_mask)
        losses = multitask_loss(out, batch)
        bs = batch.beats.shape[0]
        for key in sums:
            sums[key] += float(losses[key].detach().cpu()) * bs
        n += bs
    return {key: val / max(n, 1) for key, val in sums.items()}


def count_parameters(model: nn.Module) -> Dict[str, int]:
    # 统计模型参数量，方便确认当前配置是否和 AnyPPG 接近。
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def make_loader(args: argparse.Namespace, train: bool) -> DataLoader:
    if args.data_dir:
        # 真实训练：读取你预处理后导出的 .npz 文件。
        ds: Dataset = PreparedNPZBeatDataset(
            data_dir=args.data_dir,
            max_beats=args.max_beats,
            in_channels=args.in_channels,
            beat_len=args.beat_len,
            morph_dim=args.morph_dim,
            hrv_dim=args.hrv_dim,
            mask_ratio=args.mask_ratio,
            seed=args.seed + (0 if train else 1000),
        )
    else:
        # 不传 --data-dir 时，默认用假数据跑 smoke test。
        # 这可以先检查模型结构、loss 和保存逻辑是否正常。
        ds = SyntheticECGPPGDataset(
            n=args.synthetic_n,
            max_beats=args.max_beats,
            in_channels=args.in_channels,
            beat_len=args.beat_len,
            morph_dim=args.morph_dim,
            hrv_dim=args.hrv_dim,
            mask_ratio=args.mask_ratio,
            seed=args.seed + (0 if train else 1000),
        )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ECG+PPG multitask Transformer pretraining starter")
    parser.add_argument("--data-dir", default="", help="Directory of prepared .npz files. Empty uses synthetic data.")
    parser.add_argument("--save-path", default="ecg_ppg_multitask_pretrain.pt")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--synthetic-n", type=int, default=64)
    parser.add_argument("--in-channels", type=int, default=2)
    parser.add_argument("--beat-len", type=int, default=256)
    parser.add_argument("--max-beats", type=int, default=32)
    parser.add_argument("--morph-dim", type=int, default=6)
    parser.add_argument("--hrv-dim", type=int, default=3)
    parser.add_argument("--mask-ratio", type=float, default=0.15)

    # Orthogonal beat-phase preset. Default 2-channel model is about 5.65M params.
    parser.add_argument("--d-model", type=int, default=320)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=640)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--phase-tokens", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 有 GPU 就用 GPU，没有就用 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = make_loader(args, train=True)
    val_loader = make_loader(args, train=False)

    model = ECGPPGMultiTaskTransformer(
        in_channels=args.in_channels,
        beat_len=args.beat_len,
        morph_dim=args.morph_dim,
        hrv_dim=args.hrv_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        phase_tokens=args.phase_tokens,
    ).to(device)

    param_info = count_parameters(model)
    print(
        f"model parameters: total={param_info['total']:,} "
        f"trainable={param_info['trainable']:,}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 这里只保存验证 loss 最低的权重，作为后续微调用的 encoder 初始化。
    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device)
        va = evaluate(model, val_loader, device)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} "
            f"val_recon={va['recon']:.4f} val_pat={va['pat']:.4f} "
            f"val_morph={va['morph']:.4f} val_hrv={va['hrv']:.4f}"
        )
        if va["loss"] < best_val:
            best_val = va["loss"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "model_state": model.state_dict(),
            "args": vars(args),
            "best_val_loss": best_val,
        },
        args.save_path,
    )
    print(f"saved: {args.save_path}")


if __name__ == "__main__":
    main()
