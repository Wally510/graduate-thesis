from __future__ import annotations

import argparse
import csv
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from pat_core import PATProtocol, build_pat_window


DATA_ROOT = Path("/bingding/301/BP/downstream_datasets")
PTT_ROOT = DATA_ROOT / "Pulse_Transit_Time_PPG/1.1.0"
MENDELEY_ZIP = (
    DATA_ROOT
    / "bp_dataset_scout_20260716/mendeley_ecg_dual_ppg_2026/"
    "mendeley_original_file_verified.zip"
)
UQ_ZIP = (
    DATA_ROOT
    / "bp_dataset_scout_20260716/uq_vital_signs_full32/"
    "uqvitalsignsdata_case01to32.zip"
)
ERGO_ZIP = DATA_ROOT / "bp_dataset_scout_20260716/ergobp/ErgoBP.zip"


def append_window(
    store: dict,
    dataset: str,
    group_id: str,
    record_id: str,
    stratum: str,
    window_index: int,
    fs: float,
    ecg: np.ndarray,
    ppg: np.ndarray,
    protocol: PATProtocol,
    r_peaks: np.ndarray | None = None,
    r_source: str = "detected",
    sensitivity_ppg: np.ndarray | None = None,
    apply_frozen_range: bool = True,
    ppg_polarity: int | None = None,
) -> None:
    result = build_pat_window(
        ecg,
        ppg,
        fs,
        protocol,
        r_peaks=r_peaks,
        r_source=r_source,
        apply_frozen_range=apply_frozen_range,
        ppg_polarity=ppg_polarity,
    )
    store["audit_total"] += 1
    if not result["valid"]:
        store["rejects"][result["reject_reason"]] += 1
        for key, value in result["reject_reasons"].items():
            store["beat_rejects"][key] += int(value)
        return
    sensitivity_label = float("nan")
    if sensitivity_ppg is not None:
        sensitivity = build_pat_window(
            ecg,
            sensitivity_ppg,
            fs,
            protocol,
            r_peaks=r_peaks,
            r_source=r_source,
            apply_frozen_range=apply_frozen_range,
            ppg_polarity=ppg_polarity,
        )
        if sensitivity["valid"]:
            sensitivity_label = sensitivity["pat_median_ms"]

    sample_id = f"{dataset}|{group_id}|{record_id}|w{window_index:05d}"
    store["ecg"].append(np.asarray(ecg, dtype=np.float32))
    store["ppg"].append(np.asarray(ppg, dtype=np.float32))
    store["rows"].append(
        {
            "sample_id": sample_id,
            "dataset": dataset,
            "group_id": str(group_id),
            "record_id": record_id,
            "stratum": stratum,
            "window_index": int(window_index),
            "fs": float(fs),
            "pat_median_ms": result["pat_median_ms"],
            "pat_mean_ms": result["pat_mean_ms"],
            "pat_iqr_ms": result["pat_iqr_ms"],
            "pat_secondary_median_ms": result["pat_secondary_median_ms"],
            "pat_sensitivity_channel_median_ms": sensitivity_label,
            "valid_beat_count": result["valid_beat_count"],
            "candidate_r_count": result["candidate_r_count"],
            "valid_fraction": result["valid_fraction"],
            "slope_quality_median": result["slope_quality_median"],
            "ppg_polarity": result["ppg_polarity"],
            "ppg_polarity_source": result["ppg_polarity_source"],
            "r_detector": result["r_qc"]["detector"],
            "r_rr_plausible_fraction": result["r_qc"]["rr_plausible_fraction"],
            "r_median_hr_bpm": result["r_qc"]["median_hr_bpm"],
            "label_kind": result["label_kind"],
            "protocol_version": result["protocol_version"],
        }
    )


def new_store() -> dict:
    return {
        "ecg": [],
        "ppg": [],
        "rows": [],
        "audit_total": 0,
        "rejects": Counter(),
        "beat_rejects": Counter(),
    }


def iter_full_windows(x: np.ndarray, fs: float, window_sec: float):
    length = int(round(fs * window_sec))
    for i, start in enumerate(range(0, len(x) - length + 1, length)):
        yield i, start, start + length


def record_allowed(group: str, counts: Counter, max_groups: int, max_per_group: int) -> bool:
    if max_per_group and counts[group] >= max_per_group:
        return False
    if group not in counts and max_groups and len(counts) >= max_groups:
        return False
    counts[group] += 1
    return True


def build_ptt(protocol: PATProtocol, max_records: int, apply_frozen_range: bool, max_groups: int = 0, max_per_group: int = 0) -> dict:
    import wfdb

    store = new_store()
    store["fixed_ppg_polarity"] = -1
    records = sorted(p.stem for p in PTT_ROOT.glob("*.hea"))
    if max_records:
        records = records[:max_records]
    group_counts = Counter()
    for record_id in records:
        subject, activity = record_id.split("_", 1)
        if not record_allowed(subject, group_counts, max_groups, max_per_group):
            continue
        path = str(PTT_ROOT / record_id)
        record = wfdb.rdrecord(path, physical=True)
        names = list(record.sig_name)
        ecg = record.p_signal[:, names.index("ecg")]
        # Official README Data Description maps distal IR to pleth_2.
        ppg = record.p_signal[:, names.index("pleth_2")]
        ppg_red = record.p_signal[:, names.index("pleth_1")]
        fs = float(record.fs)
        ann = wfdb.rdann(path, "atr")
        all_r = np.asarray(ann.sample, dtype=np.int64)
        for wi, start, end in iter_full_windows(ecg, fs, protocol.window_sec):
            local_r = all_r[(all_r >= start) & (all_r < end)] - start
            append_window(
                store,
                "ptt_ppg",
                subject,
                record_id,
                activity,
                wi,
                fs,
                ecg[start:end],
                ppg[start:end],
                protocol,
                r_peaks=local_r,
                r_source="official_manual_verified_atr",
                sensitivity_ppg=ppg_red[start:end],
                apply_frozen_range=apply_frozen_range,
                ppg_polarity=-1,
            )
    return store


def build_mendeley(
    protocol: PATProtocol, max_records: int, apply_frozen_range: bool, max_groups: int = 0, max_per_group: int = 0
) -> dict:
    store = new_store()
    store["fixed_ppg_polarity"] = -1
    prefix = "A dataset of simultaneous collected ECG and PPG signals/Raw_data/"
    with zipfile.ZipFile(MENDELEY_ZIP) as zf:
        members = sorted(
            n for n in zf.namelist() if n.startswith(prefix) and n.endswith(".csv")
        )
        if max_records:
            members = members[:max_records]
        group_counts = Counter()
        for member in members:
            record_id = Path(member).stem
            subject, state = record_id.split("_", 1)
            if not record_allowed(subject, group_counts, max_groups, max_per_group):
                continue
            frame = pd.read_csv(
                zf.open(member),
                usecols=[0, 1, 2],
                dtype=np.float32,
                engine="c",
            )
            ecg = frame.iloc[:, 0].to_numpy()
            red = frame.iloc[:, 1].to_numpy()
            ir = frame.iloc[:, 2].to_numpy()
            fs = 250.0
            for wi, start, end in iter_full_windows(ecg, fs, protocol.window_sec):
                append_window(
                    store,
                    "mendeley",
                    subject,
                    record_id,
                    state,
                    wi,
                    fs,
                    ecg[start:end],
                    ir[start:end],
                    protocol,
                    sensitivity_ppg=red[start:end],
                    apply_frozen_range=apply_frozen_range,
                    ppg_polarity=-1,
                )
    return store


def read_uq_member(zf: zipfile.ZipFile, member: str) -> tuple[np.ndarray, np.ndarray]:
    reader = csv.reader(
        io.TextIOWrapper(zf.open(member), encoding="utf-8-sig", newline="")
    )
    header = next(reader)
    ecg_idx = header.index("ECG")
    ppg_idx = header.index("Pleth")
    ecg, ppg = [], []
    for row in reader:
        if len(row) <= ppg_idx:
            continue
        try:
            ecg_value = float(row[ecg_idx])
            ppg_value = float(row[ppg_idx])
        except (TypeError, ValueError):
            ecg_value = float("nan")
            ppg_value = float("nan")
        ecg.append(ecg_value)
        ppg.append(ppg_value)
    return np.asarray(ecg, dtype=np.float32), np.asarray(ppg, dtype=np.float32)


def build_uq(protocol: PATProtocol, max_records: int, apply_frozen_range: bool, max_groups: int = 0, max_per_group: int = 0) -> dict:
    store = new_store()
    store["fixed_ppg_polarity"] = 1
    pattern = re.compile(r"uqvitalsignsdata/case(\d+)/fulldata/(.+\.csv)$")
    with zipfile.ZipFile(UQ_ZIP) as zf:
        members = sorted(n for n in zf.namelist() if pattern.match(n))
        if max_records:
            members = members[:max_records]
        group_counts = Counter()
        for member in members:
            match = pattern.match(member)
            assert match is not None
            case_id = match.group(1).zfill(2)
            if not record_allowed(case_id, group_counts, max_groups, max_per_group):
                continue
            record_id = Path(member).stem
            ecg, ppg = read_uq_member(zf, member)
            fs = 100.0
            for wi, start, end in iter_full_windows(ecg, fs, protocol.window_sec):
                e = ecg[start:end]
                p = ppg[start:end]
                if np.isfinite(e).mean() < 0.95 or np.isfinite(p).mean() < 0.95:
                    store["audit_total"] += 1
                    store["rejects"]["insufficient_finite_coverage"] += 1
                    continue
                append_window(
                    store,
                    "uq",
                    case_id,
                    record_id,
                    "case",
                    wi,
                    fs,
                    e,
                    p,
                    protocol,
                    apply_frozen_range=apply_frozen_range,
                    ppg_polarity=1,
                )
    return store


def build_ergo(protocol: PATProtocol, max_records: int, apply_frozen_range: bool, max_groups: int = 0, max_per_group: int = 0) -> dict:
    import pyarrow.parquet as pq

    store = new_store()
    store["fixed_ppg_polarity"] = 1
    with zipfile.ZipFile(ERGO_ZIP) as zf:
        metadata = pq.read_table(io.BytesIO(zf.read("metadata_table.parquet"))).to_pandas()
        situation_by_file = (
            metadata.groupby("file_id")["situ_label"].agg(lambda s: s.mode().iloc[0]).to_dict()
        )
        members = sorted(
            n
            for n in zf.namelist()
            if re.match(r"^\d{2}/.+\.parquet$", n) and "metadata" not in n
        )
        accepted_files = 0
        group_counts = Counter()
        for member in members:
            subject = member.split("/", 1)[0]
            if max_per_group and group_counts[subject] >= max_per_group:
                continue
            if subject not in group_counts and max_groups and len(group_counts) >= max_groups:
                continue
            table = pq.read_table(io.BytesIO(zf.read(member)))
            if "ecg" not in table.column_names or "ppg" not in table.column_names:
                continue
            frame = table.select(["ecg", "ppg"]).to_pandas()
            ecg = frame["ecg"].to_numpy(dtype=np.float32)
            ppg = frame["ppg"].to_numpy(dtype=np.float32)
            if np.isfinite(ecg).mean() < 0.10 or np.isfinite(ppg).mean() < 0.10:
                continue
            group_counts[subject] += 1
            accepted_files += 1
            record_id = Path(member).stem
            stratum = str(situation_by_file.get(record_id, "unknown"))
            fs = 500.0
            for wi, start, end in iter_full_windows(ecg, fs, protocol.window_sec):
                e = ecg[start:end]
                p = ppg[start:end]
                if np.isfinite(e).mean() < 0.95 or np.isfinite(p).mean() < 0.95:
                    store["audit_total"] += 1
                    store["rejects"]["insufficient_true_overlap"] += 1
                    continue
                append_window(
                    store,
                    "ergobp",
                    subject,
                    record_id,
                    stratum,
                    wi,
                    fs,
                    e,
                    p,
                    protocol,
                    apply_frozen_range=apply_frozen_range,
                    ppg_polarity=1,
                )
            if max_records and accepted_files >= max_records:
                break
        store["accepted_source_files"] = accepted_files
    return store


def save_store(dataset: str, store: dict, output_root: Path, protocol: PATProtocol) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = store["rows"]
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(
            f"{dataset}: no valid PAT windows; rejects={dict(store['rejects'])}"
        )
    csv_path = output_root / f"{dataset}_pat_windows.csv"
    frame.to_csv(csv_path, index=False)
    np.savez_compressed(
        output_root / f"{dataset}_pat_windows.npz",
        ecg=np.stack(store["ecg"]).astype(np.float32),
        ppg=np.stack(store["ppg"]).astype(np.float32),
        sample_id=frame["sample_id"].to_numpy(dtype=str),
        fs=frame["fs"].to_numpy(dtype=np.float32),
        pat_ms=frame["pat_median_ms"].to_numpy(dtype=np.float32),
    )
    audit = {
        "dataset": dataset,
        "source_data_root": str(DATA_ROOT),
        "window_candidates": int(store["audit_total"]),
        "accepted_windows": int(len(rows)),
        "rejected_windows": int(store["audit_total"] - len(rows)),
        "window_reject_reasons": dict(store["rejects"]),
        "beat_reject_reasons": dict(store["beat_rejects"]),
        "group_count": int(frame["group_id"].nunique()),
        "groups": sorted(frame["group_id"].astype(str).unique().tolist()),
        "stratum_counts": frame["stratum"].value_counts().sort_index().to_dict(),
        "pat_ms_quantiles": {
            str(q): float(frame["pat_median_ms"].quantile(q))
            for q in [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
        },
        "valid_beats_quantiles": {
            str(q): float(frame["valid_beat_count"].quantile(q))
            for q in [0.0, 0.25, 0.5, 0.75, 1.0]
        },
        "protocol": protocol.__dict__,
        "label_boundary": "derived PAT label; not official label; not true PTT",
        "extra": {
            key: value
            for key, value in store.items()
            if key not in {"ecg", "ppg", "rows", "audit_total", "rejects", "beat_rejects"}
        },
    }
    (output_root / f"{dataset}_pat_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False)
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", default="ptt_ppg,mendeley,uq,ergobp"
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--max-records-per-group", type=int, default=0)
    parser.add_argument("--audit-unbounded", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    protocol = PATProtocol()
    builders = {
        "ptt_ppg": build_ptt,
        "mendeley": build_mendeley,
        "uq": build_uq,
        "ergobp": build_ergo,
    }
    output_root = Path(args.output_root)
    for dataset in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        print(f"===== BUILD {dataset} =====", flush=True)
        store = builders[dataset](
            protocol,
            args.max_records,
            not args.audit_unbounded,
            args.max_groups,
            args.max_records_per_group,
        )
        save_store(dataset, store, output_root, protocol)


if __name__ == "__main__":
    main()
