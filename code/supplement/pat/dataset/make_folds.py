from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def sha256_rows(frame: pd.DataFrame, columns: list[str]) -> str:
    text = frame[columns].sort_values(columns).to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_dataset_folds(manifest_path: Path, output_root: Path, seed: int) -> None:
    frame = pd.read_csv(manifest_path, dtype={"group_id": str})
    required = {"sample_id", "group_id", "record_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{manifest_path}: missing columns {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"{manifest_path}: duplicate sample_id")

    groups = np.asarray(sorted(frame["group_id"].astype(str).unique()))
    if len(groups) < 5:
        raise ValueError(f"{manifest_path}: only {len(groups)} groups, cannot make 5 folds")
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    group_to_fold: dict[str, int] = {}
    for fold, (_, test_pos) in enumerate(splitter.split(groups)):
        for pos in test_pos:
            group_to_fold[str(groups[pos])] = fold
    frame["test_fold"] = frame["group_id"].astype(str).map(group_to_fold).astype(int)
    frame["val_fold_for_test"] = (frame["test_fold"] - 1) % 5

    group_frame = pd.DataFrame(
        [{"group_id": group, "test_fold": fold} for group, fold in group_to_fold.items()]
    ).sort_values(["test_fold", "group_id"])
    suffix = "_pat_windows.csv"
    dataset = (
        manifest_path.name[: -len(suffix)]
        if manifest_path.name.endswith(suffix)
        else manifest_path.stem
    )
    dataset_root = output_root / dataset
    dataset_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dataset_root / "fold_manifest.csv", index=False)
    group_frame.to_csv(dataset_root / "group_folds.csv", index=False)

    folds = []
    for test_fold in range(5):
        val_fold = (test_fold + 1) % 5
        test_groups = sorted(group_frame.loc[group_frame.test_fold == test_fold, "group_id"])
        val_groups = sorted(group_frame.loc[group_frame.test_fold == val_fold, "group_id"])
        train_groups = sorted(
            group_frame.loc[
                ~group_frame.test_fold.isin([test_fold, val_fold]), "group_id"
            ]
        )
        if set(train_groups) & set(val_groups) or set(train_groups) & set(test_groups):
            raise AssertionError("group leakage")
        if set(val_groups) & set(test_groups):
            raise AssertionError("group leakage")
        folds.append(
            {
                "fold": test_fold,
                "train_groups": train_groups,
                "val_groups": val_groups,
                "test_groups": test_groups,
                "train_windows": int(frame.group_id.astype(str).isin(train_groups).sum()),
                "val_windows": int(frame.group_id.astype(str).isin(val_groups).sum()),
                "test_windows": int(frame.group_id.astype(str).isin(test_groups).sum()),
            }
        )

    payload = {
        "dataset": dataset,
        "seed": seed,
        "split_kind": "deterministic_group_5fold_outer_with_next_fold_validation",
        "group_semantics": "subject" if dataset != "uq" else "case",
        "window_random_split_forbidden": True,
        "manifest_sha256": sha256_rows(frame, ["sample_id", "group_id", "test_fold"]),
        "group_fold_sha256": sha256_rows(group_frame, ["group_id", "test_fold"]),
        "group_count": int(len(groups)),
        "window_count": int(len(frame)),
        "folds": folds,
    }
    (dataset_root / "fold_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--datasets", default="ptt_ppg,mendeley,uq,ergobp")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for dataset in args.datasets.split(","):
        dataset = dataset.strip()
        if dataset:
            make_dataset_folds(
                Path(args.manifest_root) / f"{dataset}_pat_windows.csv",
                Path(args.output_root),
                args.seed,
            )


if __name__ == "__main__":
    main()
