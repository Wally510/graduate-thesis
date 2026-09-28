#!/usr/bin/env python3
"""epoch014 SplitEnc8 + 随机Token-Dense的纯训练去噪任务。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from ddp_train_core import add_common_args, train_ddp


CORRUPTION_FAMILIES = (
    "clean_identity",
    "gaussian",
    "baseline_wander",
    "narrowband_sinusoid",
    "motion_burst",
    "mixed",
)
CORRUPTION_PROBABILITIES = (0.10, 0.20, 0.15, 0.15, 0.20, 0.20)
SEVERITY_NAMES = ("mild", "moderate", "severe")

DEFAULT_PHASE_BASE_ROOT = (
    "/data-ai/sl20200894/Code/"
    "splitenc8_phase8_maskadapter_frozen_20260724"
)
DEFAULT_REFINER_ROOT = (
    "/data-ai/sl20200894/Code/"
    "splitenc8_phase8_maskadapter_refiner_frozen_20260726"
)
DEFAULT_DENSE_ROOT = (
    "/data-ai/sl20200894/Code/"
    "splitenc8_phase8_token_dense_frozen_20260726"
)
PHASE_BASE_ROOT = Path(
    os.environ.get("PHASE_BASE_ROOT", DEFAULT_PHASE_BASE_ROOT)
)
REFINER_ROOT = Path(
    os.environ.get("REFINER_ROOT", DEFAULT_REFINER_ROOT)
)
DENSE_ROOT = Path(os.environ.get("DENSE_ROOT", DEFAULT_DENSE_ROOT))

# 只复用结构定义，禁止依赖模块加载旧重构权重。
os.environ["REFINER_SKIP_INITIAL_LOAD"] = "1"
CAPTURED_MODEL: nn.Module | None = None


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DENSE = load_module(
    DENSE_ROOT / "train_phase8_token_dense_frozen.py",
    "splitenc8_dense_structure_for_denoising",
)
BASE = DENSE.BASE


def random_initialize_adapter(
    adapter: nn.Module, *, seed: int
) -> dict[str, Any]:
    first_parameter = next(adapter.parameters())
    devices = (
        [int(first_parameter.device.index)]
        if first_parameter.is_cuda
        else []
    )
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        for module in adapter.phase_recon_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(adapter.phase_mask_token, mean=0.0, std=0.02)
    return {
        "method": "fresh_random_initialization",
        "seed": int(seed),
        "loaded_adapter_checkpoint": False,
        "phase_mask_token_used": False,
        "phase_mask_token_trainable": False,
    }


def strict_backbone_report(model: nn.Module, report: dict[str, Any]) -> None:
    if report.get("missing_backbone"):
        raise RuntimeError("epoch014 backbone存在随机缺失")
    keys = set(model.state_dict())
    required = (
        "ecg_encoder.",
        "ppg_encoder.",
        "fusion_gate.",
        "time_encoding.",
        "input_mode_embed.",
        "blocks.",
        "norm.",
    )
    missing = [
        prefix
        for prefix in required
        if not any(key.startswith(prefix) for key in keys)
    ]
    if missing:
        raise RuntimeError(f"模型结构缺少关键模块：{missing}")
    block_ids = sorted(
        {
            int(key.split(".")[1])
            for key in keys
            if key.startswith("blocks.")
            and len(key.split(".")) > 2
            and key.split(".")[1].isdigit()
        }
    )
    if block_ids != list(range(9)):
        raise RuntimeError(f"Transformer block_ids错误：{block_ids}")
    state = model.state_dict()
    for prefix in ("ecg_encoder.phase_embed", "ppg_encoder.phase_embed"):
        candidates = [
            (key, tensor)
            for key, tensor in state.items()
            if key == prefix or key.startswith(prefix + ".")
        ]
        if len(candidates) != 1 or 8 not in candidates[0][1].shape:
            raise RuntimeError(
                f"{prefix}不是split-encoder-8："
                f"{[(key, tuple(tensor.shape)) for key, tensor in candidates]}"
            )


def make_training_targets(
    beat_mask: torch.Tensor,
    *,
    phase_tokens: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    output = np.zeros((*beat_mask.shape, phase_tokens), dtype=bool)
    for sample_index, valid in enumerate(beat_mask.cpu().numpy()):
        candidates = np.flatnonzero(valid)
        if candidates.size == 0:
            raise ValueError("样本没有有效beat")
        output[sample_index, int(rng.choice(candidates)), :] = True
    return torch.from_numpy(output)


def target_beat_from_phase(phase_target_mask: torch.Tensor) -> torch.Tensor:
    target = phase_target_mask.any(dim=-1)
    if not torch.all(target.sum(dim=1) == 1):
        raise RuntimeError("每个样本必须且只能选择一个target beat")
    return target


def draw_family(batch_size: int, device: torch.device) -> torch.Tensor:
    draw = torch.rand(batch_size, device=device)
    thresholds = torch.tensor(
        np.cumsum(CORRUPTION_PROBABILITIES)[:-1],
        dtype=draw.dtype,
        device=device,
    )
    return torch.bucketize(draw, thresholds)


def corrupt_target_beats(
    beats: torch.Tensor, target_beat: torch.Tensor
) -> tuple[torch.Tensor, list[str], list[str]]:
    """在逐beat z-score空间对一个target beat加合成噪声。"""
    b, _, channels, length = beats.shape
    device, dtype = beats.device, beats.dtype
    family = draw_family(b, device)
    severity = torch.randint(0, 3, (b,), device=device)
    scale = torch.tensor(
        [0.65, 1.0, 1.55], device=device, dtype=dtype
    )[severity].view(b, 1, 1, 1)
    position = torch.linspace(
        0.0, 1.0, length, device=device, dtype=dtype
    ).view(1, 1, 1, length)
    noise = torch.zeros_like(beats)

    gaussian_on = (family == 1) | (family == 5)
    gaussian_sigma = 0.16 * scale * (
        0.75
        + 0.50
        * torch.rand(b, 1, channels, 1, device=device, dtype=dtype)
    )
    noise = noise + (
        torch.randn_like(beats)
        * gaussian_sigma
        * gaussian_on.view(b, 1, 1, 1)
    )

    baseline_on = (family == 2) | (family == 5)
    baseline_amp = 0.30 * scale * (
        0.70
        + 0.60
        * torch.rand(b, 1, channels, 1, device=device, dtype=dtype)
    )
    baseline_cycles = 0.20 + 1.30 * torch.rand(
        b, 1, channels, 1, device=device, dtype=dtype
    )
    baseline_phase = 2.0 * math.pi * torch.rand(
        b, 1, channels, 1, device=device, dtype=dtype
    )
    baseline = baseline_amp * torch.sin(
        2.0 * math.pi * baseline_cycles * position + baseline_phase
    )
    noise = noise + baseline * baseline_on.view(b, 1, 1, 1)

    narrow_on = (family == 3) | (family == 5)
    narrow_amp = 0.10 * scale * (
        0.70
        + 0.60
        * torch.rand(b, 1, channels, 1, device=device, dtype=dtype)
    )
    narrow_cycles = torch.randint(
        4, 19, (b, 1, channels, 1), device=device
    ).to(dtype)
    narrow_phase = 2.0 * math.pi * torch.rand(
        b, 1, channels, 1, device=device, dtype=dtype
    )
    narrow = narrow_amp * torch.sin(
        2.0 * math.pi * narrow_cycles * position + narrow_phase
    )
    noise = noise + narrow * narrow_on.view(b, 1, 1, 1)

    motion_on = (family == 4) | (family == 5)
    center = 0.10 + 0.80 * torch.rand(
        b, 1, channels, 1, device=device, dtype=dtype
    )
    width = 0.035 + 0.12 * torch.rand(
        b, 1, channels, 1, device=device, dtype=dtype
    )
    envelope = torch.exp(-0.5 * ((position - center) / width).square())
    motion_amp = 0.65 * scale * (
        0.60
        + 0.80
        * torch.rand(b, 1, channels, 1, device=device, dtype=dtype)
    )
    motion_texture = 0.55 * torch.randn_like(beats) + torch.empty(
        b, 1, channels, 1, device=device, dtype=dtype
    ).uniform_(-1.0, 1.0)
    noise = noise + (
        motion_amp
        * envelope
        * motion_texture
        * motion_on.view(b, 1, 1, 1)
    )

    # mixed温和叠加，避免幅度简单累加到远离训练分布。
    noise = torch.where(
        (family == 5).view(b, 1, 1, 1), 0.55 * noise, noise
    )
    corrupted = torch.where(
        target_beat[:, :, None, None], beats + noise, beats
    )
    return (
        corrupted,
        [CORRUPTION_FAMILIES[int(value)] for value in family.tolist()],
        [SEVERITY_NAMES[int(value)] for value in severity.tolist()],
    )


def encode_backbone(
    model: nn.Module,
    adapter: nn.Module,
    corrupted: torch.Tensor,
    time_sec: torch.Tensor,
    beat_mask: torch.Tensor,
    *,
    gradient_checkpointing: bool,
) -> torch.Tensor:
    b, k, _, length = corrupted.shape
    ecg_z = model.ecg_encoder.forward_phase(
        corrupted[:, :, 0:1].reshape(b * k, 1, length)
    ).reshape(b, k, 8, adapter.d_model)
    ppg_z = model.ppg_encoder.forward_phase(
        corrupted[:, :, 1:2].reshape(b * k, 1, length)
    ).reshape(b, k, 8, adapter.d_model)
    stacked = torch.stack([ecg_z, ppg_z], dim=2)
    logits = model.fusion_gate(stacked.mean(dim=3)).squeeze(-1)
    alpha = torch.softmax(logits, dim=2)
    phase_z = (alpha[:, :, :, None, None] * stacked).sum(dim=2)
    input_mode = torch.zeros(
        b, dtype=torch.long, device=corrupted.device
    )
    phase_z = (
        phase_z
        + model.input_mode_embed(input_mode).view(
            b, 1, 1, adapter.d_model
        )
        + model.time_encoding(time_sec).unsqueeze(2)
    )
    for block in model.blocks:
        if gradient_checkpointing:
            phase_z = gradient_checkpoint(
                lambda value, module=block: module(value, beat_mask),
                phase_z,
                use_reentrant=False,
            )
        else:
            phase_z = block(phase_z, beat_mask)
    return model.norm(phase_z)


def denoise_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_sample_mask: torch.Tensor,
    phase_target_mask: torch.Tensor,
    beat_mask: torch.Tensor,
    args,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_beat = target_beat_from_phase(phase_target_mask)
    primary = F.smooth_l1_loss(
        prediction[target_sample_mask], target[target_sample_mask]
    )
    pred_diff = prediction[..., 1:] - prediction[..., :-1]
    target_diff = target[..., 1:] - target[..., :-1]
    derivative_mask = (
        target_beat[:, :, None, None]
        & beat_mask[:, :, None, None]
    ).expand_as(pred_diff)
    derivative = F.smooth_l1_loss(
        pred_diff[derivative_mask], target_diff[derivative_mask]
    )
    context_mask = (
        (~target_beat)[:, :, None, None]
        & beat_mask[:, :, None, None]
    ).expand_as(target)
    context_anchor = (
        F.smooth_l1_loss(
            prediction[context_mask], target[context_mask]
        )
        if args.context_anchor_weight > 0 and context_mask.any()
        else prediction.new_zeros(())
    )
    total = (
        primary
        + args.derivative_weight * derivative
        + args.context_anchor_weight * context_anchor
    )
    return total, {
        "total": float(total.detach()),
        "target_beat_smoothl1": float(primary.detach()),
        "target_beat_derivative": float(derivative.detach()),
        "clean_context_anchor": float(context_anchor.detach()),
    }


class DenoisingHooks:
    task_name = "splitenc8_token_dense_denoising"
    loss_names = (
        "total",
        "target_beat_smoothl1",
        "target_beat_derivative",
        "clean_context_anchor",
    )

    @property
    def policy(self) -> str:
        return (
            "epoch014_"
            + self.backbone_mode
            + "_random_token_dense_targetbeat_denoising_ddp2_v1"
        )

    def build(self, args, device: torch.device) -> dict[str, Any]:
        global CAPTURED_MODEL
        self.backbone_mode = args.backbone_mode
        if Path(args.source_checkpoint) != Path(args.checkpoint):
            raise ValueError(
                "去噪任务不加载旧重构权重；source-checkpoint必须等于epoch014"
            )
        audit, common = BASE.load_common(Path(args.common_root))
        model_module = audit.load_module(Path(args.module_path))
        model = audit.build_model(model_module, args).to(device)
        checkpoint_report = audit.load_checkpoint_exact(
            model, Path(args.checkpoint)
        )
        strict_backbone_report(model, checkpoint_report)
        adapter = DENSE.PhaseMaskAdapterTokenDense(
            d_model=args.d_model,
            phase_tokens=args.phase_tokens,
            beat_len=args.beat_len,
            initial_mask_token=torch.zeros_like(model.mask_token),
            hidden_dim=args.decoder_hidden,
        ).to(device)
        init_report = random_initialize_adapter(
            adapter, seed=args.seed + 9201
        )
        CAPTURED_MODEL = model
        return {
            "model": model,
            "adapter": adapter,
            "audit": audit,
            "common": common,
            "model_module": model_module,
            "checkpoint_report": checkpoint_report,
            "source_report": {
                "checkpoint": str(args.checkpoint),
                "policy": "stage2_epoch014_backbone_only",
                "step": 0,
                "epoch": 0,
                "parameter_warm_start": False,
                "adapter_initialization": init_report,
                "backbone_mode": args.backbone_mode,
            },
        }

    def configure_trainable(self, adapter: nn.Module) -> None:
        if CAPTURED_MODEL is None:
            raise RuntimeError("未捕获backbone")
        for parameter in CAPTURED_MODEL.parameters():
            parameter.requires_grad_(False)
        if self.backbone_mode == "fullfinetune":
            modules = (
                CAPTURED_MODEL.ecg_encoder,
                CAPTURED_MODEL.ppg_encoder,
                CAPTURED_MODEL.fusion_gate,
                CAPTURED_MODEL.time_encoding,
                CAPTURED_MODEL.input_mode_embed,
                CAPTURED_MODEL.blocks,
                CAPTURED_MODEL.norm,
            )
            for module in modules:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        for parameter in adapter.parameters():
            parameter.requires_grad_(False)
        for parameter in adapter.phase_recon_head.parameters():
            parameter.requires_grad_(True)
        # 去噪保留观测token，不做token replacement。
        adapter.phase_mask_token.requires_grad_(False)

    def optimizer_groups(self, system, args):
        groups = [
            {
                "params": system.adapter.phase_recon_head.parameters(),
                "lr": args.head_lr,
                "name": "random_token_dense_head",
            }
        ]
        backbone_parameters = [
            parameter
            for parameter in system.model.parameters()
            if parameter.requires_grad
        ]
        if self.backbone_mode == "fullfinetune":
            if not backbone_parameters:
                raise RuntimeError("解冻版没有找到backbone可训练参数")
            groups.append(
                {
                    "params": backbone_parameters,
                    "lr": args.backbone_lr,
                    "name": "used_stage2_backbone_only",
                }
            )
        elif backbone_parameters:
            raise RuntimeError("冻结版仍存在backbone可训练参数")
        return groups

    def make_train_control(self, batch, args, rng, device):
        return make_training_targets(
            batch["beat_mask"], phase_tokens=8, rng=rng
        ).to(device, non_blocking=True)

    def forward_train(
        self, model, adapter, beats, time_sec, beat_mask, control
    ):
        target_beat = target_beat_from_phase(control)
        corrupted, _, _ = corrupt_target_beats(beats, target_beat)
        if self.backbone_mode == "frozen":
            model.eval()
            with torch.no_grad():
                phase_z = encode_backbone(
                    model,
                    adapter,
                    corrupted,
                    time_sec,
                    beat_mask,
                    gradient_checkpointing=False,
                )
        else:
            phase_z = encode_backbone(
                model,
                adapter,
                corrupted,
                time_sec,
                beat_mask,
                gradient_checkpointing=args_gradient_checkpointing(
                    self
                ),
            )
        prediction = adapter.decode(phase_z)
        target_samples = (
            target_beat[:, :, None, None].expand_as(beats)
            & beat_mask[:, :, None, None]
        )
        return prediction, target_samples

    def loss(
        self,
        prediction,
        target,
        target_sample_mask,
        phase_target_mask,
        beat_mask,
        args,
    ):
        return denoise_losses(
            prediction,
            target,
            target_sample_mask,
            phase_target_mask,
            beat_mask,
            args,
        )

    def active_losses(self, args):
        return {
            "target_beat_smoothl1": 1.0,
            "target_beat_derivative": args.derivative_weight,
            "clean_context_anchor": args.context_anchor_weight,
        }

    def checkpoint_metadata(self, args):
        return {
            "backbone_mode": args.backbone_mode,
            "adapter_random_initialized": True,
            "target_scope": "one_valid_target_beat_per_window",
            "corruption_families": dict(
                zip(CORRUPTION_FAMILIES, CORRUPTION_PROBABILITIES)
            ),
            "severity_levels": list(SEVERITY_NAMES),
            "phase_token_replacement": False,
            "phase_mask_token_trainable": False,
            "input_mode": 0,
        }

    def summary_metadata(self, args, output):
        policy = {
            "task": "one_target_beat_supervised_denoising",
            "backbone_mode": args.backbone_mode,
            "families": dict(
                zip(CORRUPTION_FAMILIES, CORRUPTION_PROBABILITIES)
            ),
            "severity_levels": list(SEVERITY_NAMES),
            "normalization_space": "per_beat_zscore",
            "phase_token_replacement": False,
            "phase_mask_token_trainable": False,
            "validation_disabled": True,
            "input_mode": 0,
        }
        (output / "denoising_corruption_policy.json").write_text(
            json.dumps(policy, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "adapter_random_initialized": True,
            "validation_disabled": True,
            "corruption_policy": policy,
        }


def args_gradient_checkpointing(hooks: DenoisingHooks) -> bool:
    return bool(getattr(hooks, "gradient_checkpointing", False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--backbone-mode",
        required=True,
        choices=("frozen", "fullfinetune"),
    )
    parser.add_argument(
        "--gradient-checkpointing", action="store_true"
    )
    parser.add_argument(
        "--derivative-weight", type=float, default=0.10
    )
    parser.add_argument(
        "--context-anchor-weight", type=float, default=0.005
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    hooks = DenoisingHooks()
    hooks.gradient_checkpointing = parsed.gradient_checkpointing
    train_ddp(parsed, hooks)
