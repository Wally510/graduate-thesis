#!/usr/bin/env python3
"""汇总VTaC逐epoch重构结果并绘制收敛曲线。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(
    path: Path, *, delimiter: str = ","
) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--checkpoint-table", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    checkpoint_rows = read_csv(
        Path(args.checkpoint_table), delimiter="\t"
    )
    checkpoint_by_epoch = {
        int(row["epoch"]): row for row in checkpoint_rows
    }

    condition_rows: list[dict] = []
    epoch_summary: list[dict] = []
    count_rows: list[dict] = []
    long_rows: list[dict] = []
    long_summary_rows: list[dict] = []
    manifest_hashes: set[str] = set()
    long_manifest_hashes: set[str] = set()

    for epoch in range(1, args.epochs + 1):
        epoch_dir = root / f"epoch{epoch:02d}"
        for required in (
            epoch_dir / "RUN_COMPLETE.txt",
            epoch_dir / "metrics_by_condition.csv",
            epoch_dir / "per_sample_metrics.csv",
            epoch_dir / "protocol_audit.json",
            epoch_dir / "fixed_vtac_phase_mask_manifest.jsonl",
        ):
            if not required.is_file():
                raise FileNotFoundError(required)
        audit = json.loads(
            (epoch_dir / "protocol_audit.json").read_text(
                encoding="utf-8"
            )
        )
        if audit.get("status") != "passed":
            raise RuntimeError(f"epoch {epoch}协议审计未通过")
        if int(audit.get("base_condition_count", -1)) != 15:
            raise RuntimeError(f"epoch {epoch}不是15条件")
        manifest_hashes.add(str(audit.get("base_manifest_sha256")))

        rows = [
            row
            for row in read_csv(epoch_dir / "metrics_by_condition.csv")
            if row["channel"] == "both"
            and row["scope"] == "masked_samples"
        ]
        if len(rows) != 15:
            raise RuntimeError(
                f"epoch {epoch} masked/both条件数={len(rows)}，预期15"
            )
        for row in rows:
            condition_rows.append(
                {
                    "epoch": epoch,
                    "step": checkpoint_by_epoch[epoch]["step"],
                    **row,
                }
            )
        epoch_summary.append(
            {
                "epoch": epoch,
                "step": int(checkpoint_by_epoch[epoch]["step"]),
                "macro_condition_mae": mean(float(r["mae"]) for r in rows),
                "macro_condition_rmse": mean(float(r["rmse"]) for r in rows),
                "macro_condition_pearson": mean(
                    float(r["pearson"]) for r in rows
                ),
                "evaluated_sample_count": int(
                    rows[0]["evaluated_sample_count"]
                ),
                "condition_count": len(rows),
                "checkpoint": checkpoint_by_epoch[epoch]["checkpoint"],
            }
        )
        grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[int(row["masked_phase_count"])].append(row)
        if sorted(grouped) != list(range(1, 9)):
            raise RuntimeError(f"epoch {epoch}未覆盖mask count 1..8")
        for count in range(1, 9):
            group = grouped[count]
            count_rows.append(
                {
                    "epoch": epoch,
                    "step": checkpoint_by_epoch[epoch]["step"],
                    "masked_phase_count": count,
                    "mask_ratio": count / 8,
                    "strategy_count": len(group),
                    "mae": mean(float(r["mae"]) for r in group),
                    "rmse": mean(float(r["rmse"]) for r in group),
                    "pearson": mean(float(r["pearson"]) for r in group),
                }
            )
        long_dir = epoch_dir / "long_gap_stress"
        long_metrics = [
            row
            for row in read_csv(long_dir / "metrics_by_condition.csv")
            if row["channel"] == "both"
            and row["scope"] == "masked_samples"
        ]
        if len(long_metrics) != 16:
            raise RuntimeError(
                f"epoch {epoch} long-gap条件数={len(long_metrics)}，预期16"
            )
        long_manifest = long_dir / "fixed_vtac_long_gap_mask_manifest.jsonl"
        long_manifest_hashes.add(
            hashlib.sha256(long_manifest.read_bytes()).hexdigest()
        )
        for row in long_metrics:
            long_rows.append(
                {
                    "epoch": epoch,
                    "step": checkpoint_by_epoch[epoch]["step"],
                    **row,
                }
            )
        long_summary_rows.append(
            {
                "epoch": epoch,
                "step": checkpoint_by_epoch[epoch]["step"],
                "macro_condition_mae": mean(
                    float(row["mae"]) for row in long_metrics
                ),
                "macro_condition_rmse": mean(
                    float(row["rmse"]) for row in long_metrics
                ),
                "macro_condition_pearson": mean(
                    float(row["pearson"]) for row in long_metrics
                ),
                "condition_count": 16,
            }
        )

    if len(manifest_hashes) != 1:
        raise RuntimeError(f"各epoch使用了不同mask：{manifest_hashes}")
    if len(long_manifest_hashes) != 1:
        raise RuntimeError(
            f"各epoch使用了不同long-gap mask：{long_manifest_hashes}"
        )
    write_csv(root / "all_epochs_condition_metrics.csv", condition_rows)
    write_csv(root / "all_epochs_summary.csv", epoch_summary)
    write_csv(root / "all_epochs_by_mask_count.csv", count_rows)
    write_csv(root / "long_gap_all_epochs_condition_metrics.csv", long_rows)
    write_csv(root / "long_gap_all_epochs_summary.csv", long_summary_rows)

    first = epoch_summary[0]
    last = epoch_summary[-1]
    report = {
        "status": "complete",
        "dataset": "VTaC canonical test",
        "focus": "base15_equal_width_phase_bins_count1to8",
        "epoch_count": args.epochs,
        "fixed_mask_sha256": next(iter(manifest_hashes)),
        "fixed_long_gap_mask_sha256": next(iter(long_manifest_hashes)),
        "long_gap_condition_count": 16,
        "test_sweep_not_for_checkpoint_selection": True,
        "diagnostic_min_rmse_epoch": min(
            epoch_summary,
            key=lambda row: row["macro_condition_rmse"],
        )["epoch"],
        "epoch1_to_last": {
            "mae_delta": (
                last["macro_condition_mae"]
                - first["macro_condition_mae"]
            ),
            "rmse_delta": (
                last["macro_condition_rmse"]
                - first["macro_condition_rmse"]
            ),
            "pearson_delta": (
                last["macro_condition_pearson"]
                - first["macro_condition_pearson"]
            ),
        },
        "summary_csv": str(root / "all_epochs_summary.csv"),
        "by_mask_count_csv": str(root / "all_epochs_by_mask_count.csv"),
    }
    (root / "convergence_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    epochs = [row["epoch"] for row in epoch_summary]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].plot(
        epochs,
        [row["macro_condition_mae"] for row in epoch_summary],
        marker="o",
    )
    axes[0, 0].set_title("VTaC masked MAE (15-condition macro)")
    axes[0, 1].plot(
        epochs,
        [row["macro_condition_rmse"] for row in epoch_summary],
        marker="o",
    )
    axes[0, 1].set_title("VTaC masked RMSE (15-condition macro)")
    axes[1, 0].plot(
        epochs,
        [row["macro_condition_pearson"] for row in epoch_summary],
        marker="o",
    )
    axes[1, 0].set_title("VTaC masked Pearson (15-condition macro)")
    for count in range(1, 9):
        values = [
            row["rmse"]
            for row in count_rows
            if row["masked_phase_count"] == count
        ]
        axes[1, 1].plot(
            epochs, values, marker="o", label=f"{count}/8"
        )
    axes[1, 1].set_title("RMSE by masked token count")
    axes[1, 1].legend(ncol=2, fontsize=8)
    for axis in axes.flat:
        axis.set_xlabel("Training epoch")
        axis.grid(alpha=0.25)
        axis.set_xticks(epochs)
    fig.suptitle(
        "SplitEnc8 Token-Dense VTaC reconstruction epoch sweep"
    )
    fig.tight_layout()
    fig.savefig(root / "vtac_all_epochs_reconstruction.png", dpi=180)
    plt.close(fig)

    (root / "ALL_EPOCHS_COMPLETE.txt").write_text(
        "status=complete\n"
        f"epoch_count={args.epochs}\n"
        "selection_warning=VTaC_test_not_for_checkpoint_selection\n"
        f"fixed_mask_sha256={next(iter(manifest_hashes))}\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
