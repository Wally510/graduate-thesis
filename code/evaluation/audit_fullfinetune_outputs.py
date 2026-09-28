#!/usr/bin/env python3
"""独立审计long-run frozen/fullfinetune VTaC重构结果。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


EXPECTED_BASE_HASH = (
    "322590867881a8fc067e9bdb53c51e51a6f2c9092a440c665cde01030f40cedd"
)
EXPECTED_LONG_HASH = (
    "17c77585e5c3835b19d44bd6f6f8035e948f907f4568b4bd5f04954265e33312"
)
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


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--variant", required=True, choices=("token_dense",))
    parser.add_argument("--require-long-gap", action="store_true")
    args = parser.parse_args()
    training_variant = os.environ.get("LONGRUN_TRAINING_VARIANT", "")
    if training_variant not in POLICIES:
        raise RuntimeError(
            f"LONGRUN_TRAINING_VARIANT错误：{training_variant}"
        )
    expected_policy = POLICIES[training_variant]
    result = Path(args.result_dir)
    reference = Path(args.reference_root)

    required = [
        result / "evaluation_config.json",
        result / "base_checkpoint_report.json",
        result / "adapter_checkpoint_report.json",
        result / "variant_loader_report.json",
        result / "canonical_manifest_audit.json",
        result / "metrics_by_condition.csv",
        result / "per_sample_metrics.csv",
        result / "prediction_examples.json",
        result / "fixed_vtac_phase_mask_manifest.jsonl",
        result / "summary.json",
        result / "plot_summary.json",
        reference / "fixed_vtac_phase_mask_manifest.jsonl",
        reference / "summary.json",
    ]
    if args.require_long_gap:
        required.extend(
            [
                result / "long_gap_stress" / "metrics_by_condition.csv",
                result / "long_gap_stress" / "per_sample_metrics.csv",
                result
                / "long_gap_stress"
                / "fixed_vtac_long_gap_mask_manifest.jsonl",
                result / "long_gap_stress" / "summary.json",
                reference
                / "long_gap_stress"
                / "fixed_vtac_long_gap_mask_manifest.jsonl",
                reference / "long_gap_stress" / "summary.json",
            ]
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"审计缺少文件：{missing}")

    canonical = read_json(result / "canonical_manifest_audit.json")
    if (
        canonical.get("test_row_count") != 2658
        or canonical.get("unique_record_id_count") != 2658
        or not canonical.get("all_test_record_ids_unique")
    ):
        raise RuntimeError(f"canonical 2658检查失败：{canonical}")

    base = read_json(result / "summary.json")
    ref_base = read_json(reference / "summary.json")
    if base.get("loaded_sample_count") != 2658:
        raise RuntimeError("loaded_sample_count不是2658")
    if base.get("eligible_sample_count_ge_2_beats") != 2654:
        raise RuntimeError("eligible_sample_count不是2654")
    if base.get("base_condition_count") != 15:
        raise RuntimeError("base条件数不是15")
    if base.get("phase_mask_counts") != list(range(1, 9)):
        raise RuntimeError("没有完整覆盖1–8 token")
    if base.get("strategies", {}).get("8") != ["all"]:
        raise RuntimeError("8/8条件重复或不是all")
    if base.get("evaluated_sample_count_by_condition") != ref_base.get(
        "evaluated_sample_count_by_condition"
    ):
        raise RuntimeError("base逐条件样本数与job9996不一致")
    condition_ids = {
        row["condition_id"]
        for row in read_csv(result / "metrics_by_condition.csv")
    }
    if len(condition_ids) != 15:
        raise RuntimeError(f"metrics条件数错误：{len(condition_ids)}")
    if {
        item for item in condition_ids if item.startswith("bins_8_")
    } != {"bins_8_all"}:
        raise RuntimeError("metrics中8/8条件重复")

    actual_manifest = result / "fixed_vtac_phase_mask_manifest.jsonl"
    reference_manifest = reference / "fixed_vtac_phase_mask_manifest.jsonl"
    actual_hash = sha256(actual_manifest)
    reference_hash = sha256(reference_manifest)
    if (
        actual_hash != reference_hash
        or reference_hash != EXPECTED_BASE_HASH
    ):
        raise RuntimeError(
            f"base fixed mask不一致：actual={actual_hash}, "
            f"reference={reference_hash}"
        )

    adapter = read_json(result / "adapter_checkpoint_report.json")
    actual_policy = adapter.get("policy")
    if actual_policy != expected_policy:
        raise RuntimeError(
            f"{training_variant} policy错误："
            f"{actual_policy} vs {expected_policy}"
        )
    if adapter.get("full_model_state_loaded") is not True:
        raise RuntimeError("没有加载full_model_state")
    model_report = adapter.get("full_model_state_report", {})
    if model_report.get("strict") is not True:
        raise RuntimeError("full_model_state不是strict加载")
    if (
        model_report.get("missing_keys")
        or model_report.get("unexpected_keys")
        or model_report.get("shape_mismatch")
    ):
        raise RuntimeError("full_model_state存在key或shape异常")

    variant_report = read_json(result / "variant_loader_report.json")
    if variant_report.get("variant") != args.variant:
        raise RuntimeError("variant_loader_report模型variant错误")
    if variant_report.get("training_variant") != training_variant:
        raise RuntimeError("variant_loader_report训练variant错误")
    if variant_report.get("epoch014_only") is not False:
        raise RuntimeError("错误地只使用了epoch014")
    loader = variant_report.get("loader_report", {})
    if (
        loader.get("policy") != expected_policy
        or loader.get("expected_policy") != expected_policy
    ):
        raise RuntimeError("loader report policy不匹配")
    if loader.get("full_model_state_loaded") is not True:
        raise RuntimeError("loader report未确认完整模型加载")
    if bool(loader.get("backbone_frozen_during_training")) != (
        training_variant == "frozen"
    ):
        raise RuntimeError("backbone冻结状态与训练variant不一致")

    plot = read_json(result / "plot_summary.json")
    requested = int(plot["plots_per_base_condition_requested"])
    if plot.get("base_waveform_png_count", 0) < 15 * requested:
        raise RuntimeError("波形图没有覆盖每个base条件")
    if plot.get("pdf_count") != 0 or list(result.rglob("*.pdf")):
        raise RuntimeError("PNG-only结果中出现PDF")

    long_report = None
    if args.require_long_gap:
        long = read_json(result / "long_gap_stress" / "summary.json")
        ref_long = read_json(
            reference / "long_gap_stress" / "summary.json"
        )
        if long.get("condition_count") != 16:
            raise RuntimeError("long-gap条件数不是16")
        if long.get("evaluated_sample_count_by_condition") != ref_long.get(
            "evaluated_sample_count_by_condition"
        ):
            raise RuntimeError("long-gap逐条件样本数不一致")
        long_manifest = (
            result
            / "long_gap_stress"
            / "fixed_vtac_long_gap_mask_manifest.jsonl"
        )
        ref_long_manifest = (
            reference
            / "long_gap_stress"
            / "fixed_vtac_long_gap_mask_manifest.jsonl"
        )
        actual_long_hash = sha256(long_manifest)
        reference_long_hash = sha256(ref_long_manifest)
        if (
            actual_long_hash != reference_long_hash
            or reference_long_hash != EXPECTED_LONG_HASH
        ):
            raise RuntimeError("long-gap fixed mask不一致")
        long_report = {
            "condition_count": 16,
            "manifest_match": True,
            "manifest_sha256": actual_long_hash,
        }

    report = {
        "status": "passed",
        "variant": args.variant,
        "training_variant": training_variant,
        "canonical_unique_record_ids": 2658,
        "loaded_sample_count": 2658,
        "eligible_sample_count": 2654,
        "base_condition_count": 15,
        "phase_mask_counts": list(range(1, 9)),
        "count8_not_duplicated": True,
        "condition_counts_match_job9996": True,
        "base_manifest_sha256": actual_hash,
        "full_model_state_loaded_strict": True,
        "epoch014_only": False,
        "adapter_policy": actual_policy,
        "allowed_adapter_policies": [expected_policy],
        "variant_specific_policy_check": True,
        "png_only": True,
        "plot_summary": plot,
        "optional_long_gap": long_report,
    }
    (result / "protocol_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "protocol_audit=passed "
        + json.dumps(report, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
