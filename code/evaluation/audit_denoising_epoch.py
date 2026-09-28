#!/usr/bin/env python3
"""审计单个epoch的VTaC去噪结果。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


EXPECTED_REFERENCE_HASH = (
    "322590867881a8fc067e9bdb53c51e51a6f2c9092a440c665cde01030f40cedd"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument(
        "--variant", required=True, choices=("frozen", "fullfinetune")
    )
    args = parser.parse_args()
    root = Path(args.result_dir)
    required = [
        root / "evaluation_config.json",
        root / "checkpoint_report.json",
        root / "canonical_manifest_audit.json",
        root / "metrics_by_condition.csv",
        root / "per_sample_metrics.csv",
        root / "fixed_vtac_denoise_manifest.jsonl",
        root / "summary.json",
        root / "vtac_denoise_dashboard.png",
        root / "RUN_COMPLETE.txt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    summary = json.loads((root / "summary.json").read_text())
    canonical = json.loads(
        (root / "canonical_manifest_audit.json").read_text()
    )
    checkpoint = json.loads(
        (root / "checkpoint_report.json").read_text()
    )
    if canonical.get("unique_record_id_count") != 2658:
        raise RuntimeError("canonical unique record_id不是2658")
    if summary.get("loaded_sample_count") != 2658:
        raise RuntimeError("loaded sample不是2658")
    if summary.get("eligible_sample_count") != 2654:
        raise RuntimeError("eligible sample不是2654")
    if summary.get("condition_count") != 16:
        raise RuntimeError("去噪条件不是16")
    if summary.get("training_variant") != args.variant:
        raise RuntimeError("training variant错误")
    if checkpoint.get("variant") != args.variant:
        raise RuntimeError("checkpoint variant错误")
    if checkpoint.get("strict") is not True:
        raise RuntimeError("checkpoint不是strict加载")
    if summary.get("reference_mask_manifest_sha256") != (
        EXPECTED_REFERENCE_HASH
    ):
        raise RuntimeError("权威target manifest hash错误")
    metrics = read_csv(root / "metrics_by_condition.csv")
    if len(metrics) != 48:
        raise RuntimeError(f"metrics行数={len(metrics)}，预期48")
    condition_channels = Counter(
        (row["condition_id"], row["channel"]) for row in metrics
    )
    if len(condition_channels) != 48 or any(
        count != 1 for count in condition_channels.values()
    ):
        raise RuntimeError("condition/channel存在重复")
    per_sample = read_csv(root / "per_sample_metrics.csv")
    if len(per_sample) != 2654 * 16:
        raise RuntimeError(
            f"per-sample行数={len(per_sample)}，预期={2654*16}"
        )
    manifest = root / "fixed_vtac_denoise_manifest.jsonl"
    manifest_rows = sum(1 for _ in manifest.open(encoding="utf-8"))
    if manifest_rows != 2654 * 16:
        raise RuntimeError("fixed denoise manifest行数错误")
    if list(root.rglob("*.pdf")):
        raise RuntimeError("结果中不应出现PDF")
    report = {
        "status": "passed",
        "variant": args.variant,
        "canonical_unique_record_ids": 2658,
        "loaded_sample_count": 2658,
        "eligible_sample_count": 2654,
        "condition_count": 16,
        "noisy_condition_count": 15,
        "metrics_row_count": 48,
        "per_sample_row_count": len(per_sample),
        "fixed_manifest_row_count": manifest_rows,
        "fixed_manifest_sha256": sha256(manifest),
        "reference_target_manifest_sha256": EXPECTED_REFERENCE_HASH,
        "checkpoint_loaded_strict": True,
        "png_only": True,
    }
    (root / "protocol_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("denoising_protocol_audit=passed " + json.dumps(report))


if __name__ == "__main__":
    main()
