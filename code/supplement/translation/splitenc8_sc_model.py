#!/usr/bin/env python3
"""SplitEnc8 ECG-only Segment/Compose model for a 2500-point PPG target."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as gradient_checkpoint


class ConvRefineBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels), nn.GELU(), nn.Conv1d(channels, channels, 5, padding=2),
            nn.GroupNorm(groups, channels), nn.GELU(), nn.Conv1d(channels, channels, 3, padding=1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.net(values)


class TokenDenseWaveformHead(nn.Module):
    def __init__(self, d_model: int, phase_tokens: int = 8, beat_len: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        if phase_tokens != 8 or beat_len != 128:
            raise ValueError("SplitEnc8 S-C fixes phase_tokens=8 and beat_len=128")
        self.phase_tokens = phase_tokens
        self.coarse = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 16))
        self.token_projection = nn.Linear(d_model, 128)
        self.upsampler = nn.Sequential(
            nn.ConvTranspose1d(128, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.GELU(), ConvRefineBlock(128),
            nn.ConvTranspose1d(128, 96, 4, 2, 1), nn.GroupNorm(8, 96), nn.GELU(), ConvRefineBlock(96),
            nn.ConvTranspose1d(96, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.GELU(), ConvRefineBlock(64),
            nn.ConvTranspose1d(64, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.GELU(), ConvRefineBlock(64),
        )
        self.output = nn.Conv1d(64, 1, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, phase_z: torch.Tensor) -> torch.Tensor:
        batch, beats, phases, _ = phase_z.shape
        if phases != self.phase_tokens:
            raise ValueError(f"phase token count mismatch: {phases}")
        coarse = self.coarse(phase_z).reshape(batch, beats, phases, 1, 16)
        coarse = coarse.permute(0, 1, 3, 2, 4).contiguous().reshape(batch, beats, 1, 128)
        values = self.token_projection(phase_z).reshape(batch * beats, phases, 128).transpose(1, 2)
        residual = self.output(self.upsampler(values)).reshape(batch, beats, 1, 128)
        return coarse + residual


class SplitEnc8SC(nn.Module):
    """ECG-only encoder, beat predictor, and parameter-free raw-grid compose."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        d_model: int = 768,
        phase_tokens: int = 8,
        beat_len: int = 128,
        head_hidden_dim: int = 256,
        gradient_checkpointing: bool = True,
        backbone_mode: str = "fullfinetune",
    ) -> None:
        super().__init__()
        if backbone_mode not in {"frozen", "fullfinetune"}:
            raise ValueError(backbone_mode)
        self.backbone = backbone
        self.backbone_mode = backbone_mode
        self.d_model = d_model
        self.phase_tokens = phase_tokens
        self.beat_len = beat_len
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.raw_feature_norm = nn.LayerNorm(3)
        self.raw_feature_projection = nn.Linear(3, d_model)
        self.ppg_head = TokenDenseWaveformHead(d_model, phase_tokens, beat_len, head_hidden_dim)
        self.affine_head = nn.Sequential(
            nn.Linear(d_model + 3, head_hidden_dim), nn.GELU(), nn.Linear(head_hidden_dim, 2),
        )
        nn.init.zeros_(self.raw_feature_projection.weight)
        nn.init.zeros_(self.raw_feature_projection.bias)
        nn.init.zeros_(self.affine_head[-1].weight)
        nn.init.zeros_(self.affine_head[-1].bias)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        if backbone_mode == "fullfinetune":
            for module in self.active_backbone_modules():
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        self.backbone.eval()

    def active_backbone_modules(self) -> tuple[nn.Module, ...]:
        return (self.backbone.ecg_encoder, self.backbone.time_encoding, self.backbone.blocks, self.backbone.norm)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        if self.backbone_mode == "fullfinetune":
            for module in self.active_backbone_modules():
                module.train(mode)
        return self

    def encode_beats(
        self,
        ecg: torch.Tensor,
        time_sec: torch.Tensor,
        beat_mask: torch.Tensor,
        raw_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ecg.ndim != 4 or tuple(ecg.shape[2:]) != (1, self.beat_len):
            raise ValueError(f"ECG shape mismatch: {tuple(ecg.shape)}")
        batch, beats, _, length = ecg.shape
        phase_z = self.backbone.ecg_encoder.forward_phase(ecg.reshape(batch * beats, 1, length))
        phase_z = phase_z.reshape(batch, beats, self.phase_tokens, self.d_model)
        feature_z = self.raw_feature_projection(self.raw_feature_norm(raw_features)).unsqueeze(2)
        phase_z = phase_z + feature_z + self.backbone.time_encoding(time_sec).unsqueeze(2)
        for block in self.backbone.blocks:
            if self.gradient_checkpointing and torch.is_grad_enabled():
                phase_z = gradient_checkpoint(
                    lambda value, module=block: module(value, beat_mask), phase_z, use_reentrant=False,
                )
            else:
                phase_z = block(phase_z, beat_mask)
        phase_z = self.backbone.norm(phase_z)
        beat_context = phase_z.mean(dim=2)
        return phase_z, beat_context

    def predict_beats(
        self,
        ecg: torch.Tensor,
        time_sec: torch.Tensor,
        beat_mask: torch.Tensor,
        raw_features: torch.Tensor,
    ) -> torch.Tensor:
        phase_z, beat_context = self.encode_beats(ecg, time_sec, beat_mask, raw_features)
        shape = self.ppg_head(phase_z)
        affine = self.affine_head(torch.cat([beat_context, raw_features], dim=-1))
        scale = torch.exp(affine[..., 0].clamp(-3.0, 3.0)).unsqueeze(-1).unsqueeze(-1)
        shift = affine[..., 1].unsqueeze(-1).unsqueeze(-1)
        prediction = shape * scale + shift
        return prediction * beat_mask.unsqueeze(-1).unsqueeze(-1).to(prediction.dtype)

    def compose(
        self,
        beat_prediction: torch.Tensor,
        raw_beat_index: torch.Tensor,
        raw_position: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable linear interpolation from beat coordinates to raw10s."""
        batch, beats, channels, length = beat_prediction.shape
        if channels != 1 or length != self.beat_len:
            raise ValueError(f"beat prediction shape mismatch: {tuple(beat_prediction.shape)}")
        if raw_beat_index.ndim != 2 or raw_position.shape != raw_beat_index.shape:
            raise ValueError("compose map shape mismatch")
        position0 = torch.floor(raw_position).long().clamp(0, length - 1)
        position1 = (position0 + 1).clamp(max=length - 1)
        weight = (raw_position - position0.to(raw_position.dtype)).clamp(0.0, 1.0)
        beat_index = raw_beat_index.long().clamp(0, beats - 1)
        flat = beat_prediction[:, :, 0, :].reshape(batch, beats * length)
        index0 = beat_index * length + position0
        index1 = beat_index * length + position1
        value0 = torch.gather(flat, 1, index0)
        value1 = torch.gather(flat, 1, index1)
        return ((1.0 - weight) * value0 + weight * value1).unsqueeze(1)

    def forward(
        self,
        ecg: torch.Tensor,
        time_sec: torch.Tensor,
        beat_mask: torch.Tensor,
        raw_features: torch.Tensor,
        raw_beat_index: torch.Tensor,
        raw_position: torch.Tensor,
    ) -> torch.Tensor:
        beats = self.predict_beats(ecg, time_sec, beat_mask, raw_features)
        return self.compose(beats, raw_beat_index, raw_position)


__all__ = ["SplitEnc8SC"]
