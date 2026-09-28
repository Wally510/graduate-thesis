#!/usr/bin/env python3
"""Build a frozen group-aware common raw10s split."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from common_raw10s import (
    CommonBackend, MAX_BEATS, canonical_group_id, guard_server_path,
    sha256_file, sha256_indices, stable_hash,
)


DEFAULT_PROTOCOL_NAME = "ECG2PPG-COMMON-RAW10S-GROUP-v2"
SPLIT_NAMES = ("train", "val", "test")
RATIOS = np.array([0.8, 0.1, 0.1], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multi-cache-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--protocol-name", default=DEFAULT_PROTOCOL_NAME)
    parser.add_argument("--max-beats", type=int, default=MAX_BEATS)
    parser.add_argument("--require-all-eligible", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def iter_entry_chunks(backend: CommonBackend, entry: dict, max_beats: int):
    eligible = entry["eligible"]
    eligible_start = int(entry["row"]["eligible_global_start"])
    raw_start = 0
    for shard_id, shard in enumerate(entry["raw"].shards):
        raw_end = raw_start + int(shard["n_samples"])
        left = int(np.searchsorted(eligible, raw_start, side="left"))
        right = int(np.searchsorted(eligible, raw_end, side="left"))
        if right > left:
            raw_indices = np.asarray(eligible[left:right], dtype=np.int64)
            local = raw_indices - raw_start
            counts = np.asarray(entry["beat"].shard_arrays(shard_id)["counts"][local], dtype=np.int64)
            keep = (counts >= 2) & (counts <= max_beats)
            if np.any(keep):
                kept_local = local[keep]
                arrays = entry["raw"].shard_arrays(shard_id, waveform=False)
                record_ids = np.asarray(arrays["record_id"][kept_local]).astype(str)
                global_indices = eligible_start + np.arange(left, right, dtype=np.int64)[keep]
            else:
                global_indices = np.empty(0, dtype=np.int64)
                record_ids = np.empty(0, dtype=str)
            yield global_indices, record_ids, counts[keep], counts
        raw_start = raw_end


def assign_groups(counts: Counter[str], dataset: str, seed: int) -> dict[str, int]:
    total = int(sum(counts.values()))
    targets = RATIOS * total
    assigned = np.zeros(3, dtype=np.int64)
    ordered = sorted(counts, key=lambda group: (-counts[group], stable_hash(seed, dataset, group)))
    output: dict[str, int] = {}
    for group in ordered:
        remaining_fraction = (targets - assigned) / np.maximum(targets, 1.0)
        split_id = int(np.argmax(remaining_fraction))
        output[group] = split_id
        assigned[split_id] += int(counts[group])
    if len(ordered) >= 3 and np.any(assigned == 0):
        raise RuntimeError(f"{dataset}: group split produced an empty split")
    return output


def verify(output: Path) -> dict:
    manifest_path = output / "split_manifest.json"
    complete_path = output / "RUN_COMPLETE.txt"
    split_path = output / "split_indices.npz"
    if not manifest_path.is_file() or not complete_path.is_file() or not split_path.is_file():
        raise FileNotFoundError(f"incomplete common protocol: {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not str(manifest.get("protocol", "")).startswith("ECG2PPG-COMMON-RAW10S-GROUP-") or manifest.get("status") != "passed":
        raise ValueError("common protocol manifest did not pass")
    with np.load(split_path) as data:
        arrays = {name: np.asarray(data[name], dtype=np.int64) for name in SPLIT_NAMES}
    for name, values in arrays.items():
        if sha256_indices(values) != manifest["splits"][name]["indices_sha256_int64"]:
            raise ValueError(f"{name} split hash mismatch")
        if len(values) != int(manifest["splits"][name]["count"]):
            raise ValueError(f"{name} split count mismatch")
    # The arrays are sorted and unique within each split.  One merged int64
    # array is much smaller than three Python sets for 5.5 million windows.
    for name, values in arrays.items():
        if len(values) and (np.any(values[1:] <= values[:-1]) or values[0] < 0):
            raise ValueError(f"{name} split is not strictly increasing")
    merged = np.concatenate([arrays[name] for name in SPLIT_NAMES])
    merged.sort()
    if len(merged) and np.any(merged[1:] == merged[:-1]):
        raise ValueError("sample overlap in common split")
    if len(merged) != int(manifest["total_common_samples"]):
        raise ValueError("common split total mismatch")
    if any(int(value) for value in manifest.get("group_overlap", {}).values()):
        raise ValueError("group overlap recorded in common split")
    return manifest


def main() -> None:
    args = parse_args()
    if args.max_beats < 2:
        raise ValueError("max-beats must be at least 2")
    output = guard_server_path(args.output_dir, "protocol output")
    if args.verify_only:
        manifest = verify(output)
        if manifest["protocol"] != args.protocol_name:
            raise ValueError(f"protocol name mismatch: {manifest['protocol']} != {args.protocol_name}")
        manifest_max = int(manifest.get("model_max_beats", MAX_BEATS))
        if manifest_max != args.max_beats:
            raise ValueError(f"max-beats mismatch: {manifest_max} != {args.max_beats}")
        if args.require_all_eligible and int(manifest.get("upper_limit_excluded_samples", -1)) != 0:
            raise ValueError("protocol did not retain every source-eligible sample")
        print(json.dumps({"status": "verified", "counts": manifest["counts"]}, ensure_ascii=False))
        return
    if output.exists():
        raise FileExistsError(f"refusing to overwrite protocol: {output}")
    backend = CommonBackend(args.multi_cache_index)

    group_counts: dict[str, Counter[str]] = {}
    dataset_kept: dict[str, int] = {}
    beat_histogram: Counter[int] = Counter()
    all_eligible_histogram: Counter[int] = Counter()
    for entry in backend.entries:
        dataset = entry["dataset"]
        counts: Counter[str] = Counter()
        kept = 0
        for _global_indices, record_ids, beat_counts, all_counts in iter_entry_chunks(backend, entry, args.max_beats):
            groups = [canonical_group_id(dataset, value) for value in record_ids.tolist()]
            counts.update(groups)
            kept += len(groups)
            beat_histogram.update(int(value) for value in beat_counts.tolist())
            all_eligible_histogram.update(int(value) for value in all_counts.tolist())
        if not counts:
            raise RuntimeError(f"{dataset}: no common eligible samples")
        group_counts[dataset] = counts
        dataset_kept[dataset] = kept
        print(f"[groups] dataset={dataset} samples={kept} groups={len(counts)}", flush=True)

    source_eligible_total = int(sum(all_eligible_histogram.values()))
    retained_total = int(sum(dataset_kept.values()))
    if source_eligible_total != backend.total:
        raise RuntimeError(
            f"source eligible scan mismatch: {source_eligible_total} != index total {backend.total}"
        )
    upper_excluded = source_eligible_total - retained_total
    if args.require_all_eligible and upper_excluded:
        maximum = max(all_eligible_histogram) if all_eligible_histogram else -1
        raise RuntimeError(
            f"max-beats={args.max_beats} excluded {upper_excluded} source-eligible samples; "
            f"detected maximum={maximum}"
        )

    assignments = {
        dataset: assign_groups(counts, dataset, args.seed)
        for dataset, counts in group_counts.items()
    }
    parts: dict[str, list[np.ndarray]] = {name: [] for name in SPLIT_NAMES}
    dataset_split_counts: dict[str, dict[str, int]] = {}
    for entry in backend.entries:
        dataset = entry["dataset"]
        per_split = {name: 0 for name in SPLIT_NAMES}
        assignment = assignments[dataset]
        for global_indices, record_ids, _beat_counts, _all_counts in iter_entry_chunks(backend, entry, args.max_beats):
            if not len(global_indices):
                continue
            split_ids = np.fromiter(
                (assignment[canonical_group_id(dataset, value)] for value in record_ids.tolist()),
                dtype=np.int8,
                count=len(record_ids),
            )
            for split_id, name in enumerate(SPLIT_NAMES):
                selected = global_indices[split_ids == split_id]
                if len(selected):
                    parts[name].append(selected)
                    per_split[name] += int(len(selected))
        dataset_split_counts[dataset] = per_split

    splits = {
        name: np.sort(np.concatenate(parts[name])).astype(np.int64)
        for name in SPLIT_NAMES
    }
    # Do not create the protocol directory until the strict all-eligible audit
    # and both group-assignment passes have succeeded.  In particular, a
    # max-beats violation must not leave an empty directory that a retry could
    # mistake for an existing protocol.
    output.mkdir(parents=True)
    np.savez_compressed(output / "split_indices.npz", **splits)
    with (output / "group_assignments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "group_id", "split", "eligible_windows"])
        for dataset in backend.index["dataset_order"]:
            for group in sorted(group_counts[dataset]):
                writer.writerow([
                    dataset, group, SPLIT_NAMES[assignments[dataset][group]], group_counts[dataset][group],
                ])

    group_sets = {
        name: {
            group for dataset in assignments for group, split_id in assignments[dataset].items()
            if split_id == index
        }
        for index, name in enumerate(SPLIT_NAMES)
    }
    overlap = {
        "train_val": len(group_sets["train"] & group_sets["val"]),
        "train_test": len(group_sets["train"] & group_sets["test"]),
        "val_test": len(group_sets["val"] & group_sets["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"group overlap: {overlap}")
    total = sum(len(values) for values in splits.values())
    manifest = {
        "status": "passed",
        "protocol": args.protocol_name,
        "multi_cache_index": str(backend.index_path),
        "multi_cache_index_sha256": sha256_file(backend.index_path),
        "split_seed": args.seed,
        "split_unit": "dataset_plus_source_record_group",
        "pulsedb_group_rule": "strip_final_colon_segment_index_from_record_id",
        "other_dataset_group_rule": "dataset_plus_record_id",
        "eligibility": f"existing ECG-only beat manifest and 2 <= detected ECG peaks <= {args.max_beats}",
        "model_max_beats": args.max_beats,
        "source_eligible_samples_before_upper_limit": source_eligible_total,
        "upper_limit_excluded_samples": upper_excluded,
        "all_source_eligible_retained": upper_excluded == 0,
        "maximum_detected_beats": max(all_eligible_histogram) if all_eligible_histogram else None,
        "beat_boundaries": "contiguous midpoints between ECG peaks; first starts at 0 and last ends at 2500",
        "target_used_for_segmentation": False,
        "counts": {name: int(len(values)) for name, values in splits.items()},
        "total_common_samples": total,
        "ratios": {name: len(splits[name]) / max(total, 1) for name in SPLIT_NAMES},
        "dataset_common_counts": dataset_kept,
        "dataset_split_counts": dataset_split_counts,
        "dataset_group_counts": {dataset: len(values) for dataset, values in group_counts.items()},
        "group_overlap": overlap,
        "splits": {
            name: {
                "count": int(len(values)),
                "indices_sha256_int64": sha256_indices(values),
            }
            for name, values in splits.items()
        },
        "beat_count_histogram": {str(key): int(beat_histogram[key]) for key in sorted(beat_histogram)},
        "source_eligible_beat_count_histogram": {
            str(key): int(all_eligible_histogram[key]) for key in sorted(all_eligible_histogram)
        },
        "artifacts": {
            "split_indices": "split_indices.npz",
            "group_assignments": "group_assignments.csv",
        },
    }
    (output / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "RUN_COMPLETE.txt").write_text(
        json.dumps({"status": "complete", "protocol": args.protocol_name, "counts": manifest["counts"]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    verify(output)
    print(json.dumps({"status": "complete", "output": str(output), "counts": manifest["counts"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
