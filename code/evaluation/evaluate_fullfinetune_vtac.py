#!/usr/bin/env python3
"""兼容包装：严格加载新long-run frozen/fullfinetune epoch权重。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch.nn as nn


POLICIES = {
    "frozen": (
        "epoch014_frozen_random_token_dense_random_phase_mask_"
        "multibeat_longrun_ddp2_v1"
    ),
    "fullfinetune": (
        "epoch014_fullfinetune_random_token_dense_random_phase_mask_"
        "multibeat_longrun_ddp2_v1"
    ),
}
CURRENT_VARIANT = os.environ.get("LONGRUN_TRAINING_VARIANT", "")
if CURRENT_VARIANT not in POLICIES:
    raise RuntimeError(
        f"LONGRUN_TRAINING_VARIANT错误：{CURRENT_VARIANT}"
    )
LEGACY_ROOT = Path(
    os.environ.get(
        "LEGACY_EVAL_ROOT",
        "/data-ai/sl20200894/Code/"
        "splitenc8_phase8_fullfinetune_vtac_eval_20260726",
    )
)


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEGACY = load_module(
    LEGACY_ROOT / "evaluate_fullfinetune_vtac.py",
    "legacy_fullfinetune_vtac_for_longrun_epoch_sweep",
)
LOADER_REPORT: dict[str, Any] = {}


def load_current_checkpoint(
    adapter: nn.Module,
    path: Path,
    *,
    expected_base_checkpoint: Path,
) -> dict[str, Any]:
    global LOADER_REPORT
    if LEGACY.CAPTURED_MODEL is None:
        raise RuntimeError("尚未捕获Stage2模型")
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, dict):
        raise TypeError("checkpoint顶层不是dict")
    expected_policy = POLICIES[CURRENT_VARIANT]
    if raw.get("policy") != expected_policy:
        raise RuntimeError(
            f"policy错误：{raw.get('policy')} vs {expected_policy}"
        )
    if raw.get("validation_disabled") is not True:
        raise RuntimeError("checkpoint未声明validation_disabled")
    if raw.get("full_model_state_saved") is not True:
        raise RuntimeError("checkpoint未保存完整backbone")
    if bool(raw.get("backbone_frozen")) != (
        CURRENT_VARIANT == "frozen"
    ):
        raise RuntimeError("backbone_frozen与variant不一致")
    args = raw.get("args")
    if not isinstance(args, dict):
        raise RuntimeError("checkpoint缺少args")
    checkpoint_base = args.get("checkpoint")
    if (
        not checkpoint_base
        or Path(str(checkpoint_base)).name
        != expected_base_checkpoint.name
    ):
        raise RuntimeError("epoch014 base checkpoint不匹配")
    expected_losses = {
        "masked_smoothl1": 1.0,
        "masked_derivative": 0.1,
        "boundary_continuity": 0.1,
        "visible_context_anchor": 0.01,
    }
    if raw.get("active_losses") != expected_losses:
        raise RuntimeError("重构loss协议不匹配")
    expected_families = {
        "whole_beats": 0.6,
        "cross_boundary": 0.15,
        "mixed": 0.15,
        "time_gap": 0.1,
    }
    if raw.get("mask_families") != expected_families:
        raise RuntimeError(
            f"long-run mask family错误：{raw.get('mask_families')}"
        )
    adapter_state = raw.get("phase_adapter_state")
    full_state = raw.get("full_model_state")
    if not isinstance(adapter_state, dict) or not adapter_state:
        raise RuntimeError("缺少phase_adapter_state")
    if not isinstance(full_state, dict) or not full_state:
        raise RuntimeError("缺少full_model_state")
    adapter_report = LEGACY._strict_load_state(
        adapter, adapter_state, label="phase_adapter_state"
    )
    model_report = LEGACY._strict_load_state(
        LEGACY.CAPTURED_MODEL,
        full_state,
        label="full_model_state",
    )
    LOADER_REPORT = {
        "checkpoint": str(path),
        "epoch": int(raw.get("epoch", -1)),
        "step": int(raw.get("step", -1)),
        "kind": raw.get("kind"),
        "policy": raw.get("policy"),
        "expected_policy": expected_policy,
        "variant": CURRENT_VARIANT,
        "base_checkpoint": checkpoint_base,
        "validation_disabled": True,
        "losses": raw.get("active_losses"),
        "mask_families": raw.get("mask_families"),
        "adapter_random_initialized": raw.get(
            "adapter_random_initialized"
        ),
        "adapter_state_report": adapter_report,
        "full_model_state_report": model_report,
        "full_model_state_loaded": True,
        "backbone_frozen_during_training": (
            CURRENT_VARIANT == "frozen"
        ),
    }
    LEGACY.LOADER_REPORT = LOADER_REPORT
    return LOADER_REPORT


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=("token_dense",), required=True)
    wrapper, remaining = parser.parse_known_args()
    LEGACY.load_fullfinetune_checkpoint = load_current_checkpoint
    LEGACY.EXPECTED_POLICIES["token_dense"] = (
        POLICIES[CURRENT_VARIANT],
    )
    sys.argv = [
        sys.argv[0],
        "--variant",
        "token_dense",
        "--output-dir",
        wrapper.output_dir,
        *remaining,
    ]
    LEGACY.main()
    report = {
        "variant": "token_dense",
        "training_variant": CURRENT_VARIANT,
        "loader_report": LOADER_REPORT,
        "base_forward": "stage2_phase_tokens_then_token_dense",
        "long_gap_forward": "exact_sample_mask_then_token_dense",
        "full_model_state_loaded": True,
        "epoch014_only": False,
    }
    (
        Path(wrapper.output_dir) / "variant_loader_report.json"
    ).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
