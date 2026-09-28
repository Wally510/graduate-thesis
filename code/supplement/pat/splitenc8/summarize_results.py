#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--result-root", required=True)
    args = p.parse_args()
    root = Path(args.result_root)
    rows, folds, groups, strata = [], [], [], []
    payload = {}
    for path in sorted(root.glob("*/*/summary.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        dataset, track = doc["dataset"], doc["track"]
        payload.setdefault(dataset, {})[track] = doc
        ev = doc["evaluation"]
        row = {
            "dataset": dataset, "track": track,
            "leaderboard": doc["leaderboard"],
            "n": doc["sample_count"], "groups": doc["group_count"],
            **{f"pooled_{k}": v for k, v in ev["pooled"].items()},
            **ev["subject_or_case_macro"],
        }
        for metric, value in ev["fold_mean_std"].items():
            row[f"fold_{metric}_mean"] = value["mean"]
            row[f"fold_{metric}_std"] = value["std"]
        rows.append(row)
        fold = pd.read_csv(path.parent / "fold_metrics.csv")
        fold.insert(0, "track", track)
        fold.insert(0, "dataset", dataset)
        folds.append(fold)
        group = pd.read_csv(path.parent / "group_metrics.csv")
        group.insert(0, "track", track)
        group.insert(0, "dataset", dataset)
        groups.append(group)
        stratum = pd.read_csv(path.parent / "stratum_metrics.csv")
        stratum.insert(0, "track", track)
        stratum.insert(0, "dataset", dataset)
        strata.append(stratum)
    if len(rows) != 12:
        raise RuntimeError(f"期望12个 dataset×track 结果，实际{len(rows)}")
    summary = pd.DataFrame(rows)
    fold_df, group_df, stratum_df = map(pd.concat, (folds, groups, strata))
    summary.to_csv(root / "summary_all.csv", index=False)
    fold_df.to_csv(root / "fold_metrics_all.csv", index=False)
    group_df.to_csv(root / "group_metrics_all.csv", index=False)
    stratum_df.to_csv(root / "stratum_metrics_all.csv", index=False)
    (root / "summary_all.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with pd.ExcelWriter(root / "summary_all.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, "summary", index=False)
        fold_df.to_excel(writer, "folds", index=False)
        group_df.to_excel(writer, "groups", index=False)
        stratum_df.to_excel(writer, "strata", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
