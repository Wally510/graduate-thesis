#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED = {
    "ptt_ppg": (3189, 22),
    "mendeley": (7860, 148),
    "uq": (18553, 32),
    "ergobp": (1028, 23),
}
FORBIDDEN = "/bingding" + "/301/BP"
CRITICAL = (
    "ecg_encoder.", "ppg_encoder.", "fusion_gate.", "time_encoding.",
    "input_mode_embed.", "blocks.", "norm.", "reliability_pool.",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("splitenc8_model_contract", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_audit(checkpoint: Path, module_path: Path) -> dict:
    import torch

    module = load_module(module_path)
    model = module.ECGPPGMultiTaskTransformer(
        in_channels=2, beat_len=128, d_model=768, nhead=12,
        num_layers=9, dim_feedforward=2048, phase_tokens=8,
    )
    result = module.load_matching_foundation_state(model, str(checkpoint))
    matched, missing, skipped, missing_keys, skipped_details, unmatched = result
    missing_critical = [k for k in missing_keys if k.startswith(CRITICAL)]
    if missing_critical or skipped:
        raise RuntimeError(
            f"checkpoint关键权重未完整加载: missing={missing_critical[:20]} "
            f"skipped={skipped_details[:20]}"
        )
    state_keys = model.state_dict()
    for prefix in CRITICAL:
        if not any(k.startswith(prefix) for k in state_keys):
            raise RuntimeError(f"模型缺少关键模块: {prefix}")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "matched_tensor_count": matched,
        "non_backbone_missing_count": missing,
        "shape_skipped_count": skipped,
        "unmatched_checkpoint_key_count": len(unmatched),
        "architecture": {
            "beat_len": 128, "d_model": 768, "nhead": 12,
            "num_layers": 9, "dim_feedforward": 2048, "phase_tokens": 8,
        },
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }


def dataset_audit(root: Path) -> dict:
    sums = root / "SHA256SUMS"
    if not sums.is_file():
        raise FileNotFoundError(f"缺少 {sums}")
    declared = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(maxsplit=1)
        declared[rel.lstrip("*")] = digest
    expected_hashed = {"TRANSFER_PROVENANCE.json"}
    for dataset in EXPECTED:
        expected_hashed.add(f"data_v2_fixed_polarity/{dataset}_pat_windows.npz")
        expected_hashed.add(f"folds_v2_fixed_polarity/{dataset}/fold_manifest.csv")
    if set(declared) != expected_hashed:
        raise RuntimeError(
            f"SHA256SUMS成员不等于固定9项: "
            f"missing={sorted(expected_hashed - set(declared))}, "
            f"extra={sorted(set(declared) - expected_hashed)}"
        )
    for rel, digest in declared.items():
        path = root / rel
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"hash不一致: {rel}")

    out = {}
    for dataset, (expected_n, expected_groups) in EXPECTED.items():
        npz_path = root / "data_v2_fixed_polarity" / f"{dataset}_pat_windows.npz"
        fold_path = root / "folds_v2_fixed_polarity" / dataset / "fold_manifest.csv"
        if not npz_path.is_file() or not fold_path.is_file():
            raise FileNotFoundError(f"{dataset} 冻结资产不完整")
        with np.load(npz_path, mmap_mode="r") as z:
            ids = z["sample_id"].astype(str)
            required = {"sample_id", "ecg", "ppg", "fs"}
            if not required.issubset(z.files):
                raise RuntimeError(f"{npz_path} 缺少 {sorted(required - set(z.files))}")
        fold = pd.read_csv(fold_path, dtype={"group_id": str})
        required_cols = {
            "sample_id", "group_id", "record_id", "stratum",
            "pat_median_ms", "test_fold",
        }
        if not required_cols.issubset(fold.columns):
            raise RuntimeError(f"{fold_path} 缺少 {sorted(required_cols - set(fold.columns))}")
        if len(fold) != expected_n or fold.group_id.nunique() != expected_groups:
            raise RuntimeError(
                f"{dataset} 数量错误: windows={len(fold)}/{expected_n}, "
                f"groups={fold.group_id.nunique()}/{expected_groups}"
            )
        if set(fold.test_fold.astype(int)) != set(range(5)):
            raise RuntimeError(f"{dataset} fold不是0..4")
        if "val_fold_for_test" in fold:
            if set(fold.val_fold_for_test.astype(int)) != set(range(5)):
                raise RuntimeError(f"{dataset} val_fold_for_test不是0..4")
            expected_val_owner = (fold.test_fold.astype(int) - 1) % 5
            if not np.array_equal(fold.val_fold_for_test.astype(int), expected_val_owner):
                raise RuntimeError(f"{dataset} validation fold映射不一致")
        if fold.groupby("group_id").test_fold.nunique().max() != 1:
            raise RuntimeError(f"{dataset} 存在group跨fold")
        if set(fold.sample_id.astype(str)) != set(ids):
            raise RuntimeError(f"{dataset} NPZ与fold sample_id集合不一致")
        out[dataset] = {
            "windows": len(fold),
            "groups": int(fold.group_id.nunique()),
            "fold_windows": fold.groupby("test_fold").size().astype(int).to_dict(),
            "fold_groups": fold.groupby("test_fold").group_id.nunique().astype(int).to_dict(),
            "npz_sha256": sha256(npz_path),
            "fold_sha256": sha256(fold_path),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frozen-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-module", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    for value in vars(args).values():
        if FORBIDDEN in str(value):
            raise RuntimeError("禁止访问旧服务器路径")
    payload = {
        "dataset_audit": dataset_audit(Path(args.frozen_root)),
        "checkpoint_audit": checkpoint_audit(
            Path(args.checkpoint), Path(args.model_module)
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
