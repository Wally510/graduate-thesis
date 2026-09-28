# -*- coding: utf-8 -*-
"""
Prepare cached .npz files for ECG+PPG foundation pretraining.

This script writes one .npz sample per record/window/segment. The cache is
foundation-only: ECG and PPG are used; SBP/DBP/ABP labels are not saved as
training targets.

Supported sources:
  1) tptcom/sysu style server CSV + signal .txt/.txt.gz files
  2) official PulseDB Segment_Files
  3) UCI cuffless Part_*.mat files, using ECG+PPG only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from prepare_ecg_ppg_pretrain_npz_permissive import (
    extract_pretrain_npz_arrays,
    iter_uci_records,
    limit_saved_beats,
    load_server_signal_file,
    read_csv_robust,
    resolve_signal_path,
)
from train_ecg_ppg_foundation_orthogonal_multisource import (
    flatten_pulsedb_segments,
    load_pulsedb_subject_windows,
    pulsedb_segment_to_ecg_ppg,
)


# =============================================================================
# Editable server config
# =============================================================================
USE_SCRIPT_CONFIG = True

OUT_ROOT = "/data-ai/sl20200894/prepared_pretrain_npz/ecg_ppg_foundation_multisource_fast_uncompressed"

ENABLE_SERVER = True
SERVER_DATASETS = [
    {
        "enabled": True,
        "name": "tptcom",
        "csv": "/data-ai/sl20200894/AnyPPGData/data/tptcom_data_existing_only.csv",
        "root": "/data-ai/sl20200894/AnyPPGData/tptcom",
    },
    {
        "enabled": True,
        "name": "sysu",
        "csv": "/data-ai/sl20200894/AnyPPGData/data/sysu_data_existing_only.csv",
        "root": "/data-ai/sl20200894/AnyPPGData/sysu",
    },
]
SERVER_FILE_COL = "ECG+PPG文件索引"
SERVER_SIGNAL_PATH_COL = ""
SERVER_SIGNAL_COLS = "ecgor,ppgir"
SERVER_INPUT_COLS = "0,1"
SERVER_ECG_COL = 0
SERVER_PPG_TARGET_COL = 1
SERVER_FS = 250

ENABLE_PULSEDB = True
PULSEDB_ROOT = "/data-ai/sl20200894/data/PulseDB"
PULSEDB_USE_MIMIC = True
PULSEDB_USE_VITAL = True
PULSEDB_ECG_FIELD = "ECG_F"
PULSEDB_PPG_FIELD = "PPG_F"
PULSEDB_FS = 125
PULSEDB_LIMIT_SUBJECT_FILES = 0
PULSEDB_LIMIT_SEGMENTS_PER_FILE = 0

ENABLE_UCI = True
UCI_MAT_DIR = "/data-ai/sl20200894/data/cuff+less+blood+pressure+estimation"
UCI_PARTS = "Part_1.mat,Part_2.mat,Part_3.mat,Part_4.mat"
UCI_FS = 125

# Use the same beat length in cache generation and cached training.
BEAT_LEN = 128
MIN_BEATS = 1
MAX_SAVE_BEATS = 64

LIMIT_PER_SOURCE = 0
LOG_EVERY = 100
SKIP_EXISTING = True
COMPRESS_NPZ = False
NUM_SHARDS = 1
SHARD_INDEX = 0


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def in_shard(index: int, num_shards: int, shard_index: int) -> bool:
    if num_shards <= 1:
        return True
    return int(index) % int(num_shards) == int(shard_index)


def save_cached_npz(out_path: Path, arrays: Dict[str, np.ndarray], meta: Dict[str, Any], compress: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arrays)
    payload["meta"] = json.dumps(meta, ensure_ascii=False)
    if compress:
        np.savez_compressed(out_path, **payload)
    else:
        np.savez(out_path, **payload)


class ManifestWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "w", newline="", encoding="utf-8-sig")
        self.fieldnames = [
            "source",
            "subset",
            "subject_file",
            "index",
            "path",
            "out",
            "status",
            "reason",
            "n_beats",
        ]
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()

    def write(self, row: Dict[str, Any]) -> None:
        clean = {key: row.get(key, "") for key in self.fieldnames}
        self.writer.writerow(clean)

    def close(self) -> None:
        self.file.close()


def cache_arrays(
    sig_tc: np.ndarray,
    fs: int,
    input_cols: Sequence[int],
    ecg_col: int,
    ppg_target_col: int,
    beat_len: int,
    min_beats: int,
    max_save_beats: int,
    out_path: Path,
    meta: Dict[str, Any],
    skip_existing: bool,
    compress_npz: bool,
) -> int:
    if skip_existing and out_path.exists():
        with np.load(out_path, allow_pickle=False) as item:
            return int(item["beats"].shape[0])

    arrays = extract_pretrain_npz_arrays(
        sig_tc,
        fs=fs,
        input_cols=input_cols,
        ecg_col=ecg_col,
        ppg_target_col=ppg_target_col,
        beat_len=beat_len,
        min_beats=min_beats,
    )
    if arrays is None:
        raise ValueError("could not extract any beat or pseudo-beat")
    arrays = limit_saved_beats(arrays, max_save_beats)
    save_cached_npz(out_path, arrays, meta=meta, compress=compress_npz)
    return int(arrays["beats"].shape[0])


def convert_server_dataset(args: argparse.Namespace, manifest: ManifestWriter, cfg: Dict[str, Any]) -> None:
    source_name = str(cfg["name"])
    out_dir = Path(args.out_root) / source_name
    signal_cols = [s.strip() for s in args.signal_cols.split(",") if s.strip()]
    input_cols = parse_int_list(args.input_cols)

    df = read_csv_robust(cfg["csv"])
    if args.signal_path_col and args.signal_path_col in df.columns:
        paths = df[args.signal_path_col].tolist()
    else:
        if args.file_col not in df.columns:
            raise KeyError(f"{cfg['csv']} missing {args.file_col}; pass --file-col or --signal-path-col")
        paths = [resolve_signal_path(cfg["root"], p) for p in df[args.file_col]]
    if args.limit_per_source > 0:
        paths = paths[: args.limit_per_source]

    ok = 0
    processed = 0
    for idx, path in enumerate(paths):
        if not in_shard(idx, args.num_shards, args.shard_index):
            continue
        processed += 1
        rec = {
            "source": source_name,
            "subset": "server",
            "subject_file": "",
            "index": idx,
            "path": str(path),
            "status": "skip",
            "reason": "",
        }
        try:
            if path is None or not os.path.exists(str(path)):
                raise FileNotFoundError(str(path))
            sig = load_server_signal_file(path, signal_cols=signal_cols)
            out_path = out_dir / f"{source_name}_{idx:07d}.npz"
            n_beats = cache_arrays(
                sig,
                fs=args.server_fs,
                input_cols=input_cols,
                ecg_col=args.ecg_col,
                ppg_target_col=args.ppg_target_col,
                beat_len=args.beat_len,
                min_beats=args.min_beats,
                max_save_beats=args.max_save_beats,
                out_path=out_path,
                meta=rec,
                skip_existing=args.skip_existing,
                compress_npz=args.compress_npz,
            )
            rec.update({"status": "ok", "out": str(out_path), "n_beats": n_beats})
            ok += 1
        except Exception as exc:
            rec["reason"] = repr(exc)
        manifest.write(rec)
        if processed % args.log_every == 0:
            print(
                f"[server:{source_name}] shard={args.shard_index}/{args.num_shards} "
                f"processed={processed} source_index={idx + 1} ok={ok}",
                flush=True,
            )
    print(
        f"[server:{source_name}] finished shard={args.shard_index}/{args.num_shards} "
        f"processed={processed} total_source={len(paths)} ok={ok} out_dir={out_dir}",
        flush=True,
    )


def find_pulsedb_subject_files(args: argparse.Namespace) -> List[Path]:
    root = Path(args.pulsedb_root)
    folders: List[Path] = []
    if args.pulsedb_use_mimic:
        folders.append(root / "Segment_Files" / "PulseDB_MIMIC")
    if args.pulsedb_use_vital:
        folders.append(root / "Segment_Files" / "PulseDB_Vital")

    files: List[Path] = []
    for folder in folders:
        if not folder.exists():
            print(f"[warn] missing PulseDB folder: {folder}", flush=True)
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            if path.name.lower().startswith(("license", "readme")):
                continue
            if path.suffix.lower() not in {"", ".mat"}:
                continue
            files.append(path)
            if args.pulsedb_limit_subject_files > 0 and len(files) >= args.pulsedb_limit_subject_files:
                return files
    return files


def convert_pulsedb(args: argparse.Namespace, manifest: ManifestWriter) -> None:
    files = find_pulsedb_subject_files(args)
    ok = 0
    seen = 0
    for file_i, path in enumerate(files):
        if not in_shard(file_i, args.num_shards, args.shard_index):
            continue
        subset = path.parent.name.lower()
        out_dir = Path(args.out_root) / subset
        try:
            subj_wins = load_pulsedb_subject_windows(path)
            segments = flatten_pulsedb_segments(subj_wins)
        except Exception as exc:
            rec = {
                "source": "pulsedb",
                "subset": subset,
                "subject_file": path.name,
                "index": "",
                "path": str(path),
                "status": "skip",
                "reason": repr(exc),
            }
            manifest.write(rec)
            continue

        if args.pulsedb_limit_segments_per_file > 0:
            segments = segments[: args.pulsedb_limit_segments_per_file]

        subject_ok = 0
        for seg_i, segment in enumerate(segments):
            seen += 1
            rec = {
                "source": "pulsedb",
                "subset": subset,
                "subject_file": path.name,
                "index": seg_i,
                "path": str(path),
                "status": "skip",
                "reason": "",
            }
            try:
                sig = pulsedb_segment_to_ecg_ppg(
                    segment,
                    path=path,
                    ecg_field=args.pulsedb_ecg_field,
                    ppg_field=args.pulsedb_ppg_field,
                )
                out_name = f"{subset}_{path.stem}_{seg_i:07d}.npz"
                out_path = out_dir / out_name
                n_beats = cache_arrays(
                    sig,
                    fs=args.pulsedb_fs,
                    input_cols=[0, 1],
                    ecg_col=0,
                    ppg_target_col=1,
                    beat_len=args.beat_len,
                    min_beats=args.min_beats,
                    max_save_beats=args.max_save_beats,
                    out_path=out_path,
                    meta=rec,
                    skip_existing=args.skip_existing,
                    compress_npz=args.compress_npz,
                )
                rec.update({"status": "ok", "out": str(out_path), "n_beats": n_beats})
                ok += 1
                subject_ok += 1
            except Exception as exc:
                rec["reason"] = repr(exc)
            manifest.write(rec)
            if seen % args.log_every == 0:
                print(
                    f"[pulsedb] shard={args.shard_index}/{args.num_shards} "
                    f"files={file_i + 1}/{len(files)} seen_segments={seen} ok={ok}",
                    flush=True,
                )
        print(
            f"[pulsedb:{subset}] shard={args.shard_index}/{args.num_shards} "
            f"file={file_i + 1}/{len(files)} {path.name} segments={len(segments)} ok={subject_ok}",
            flush=True,
        )
    print(
        f"[pulsedb] finished shard={args.shard_index}/{args.num_shards} "
        f"subject_files_total={len(files)} seen_segments={seen} ok={ok}",
        flush=True,
    )


def convert_uci(args: argparse.Namespace, manifest: ManifestWriter) -> None:
    parts = [p.strip() for p in args.uci_parts.split(",") if p.strip()]
    out_dir = Path(args.out_root) / "uci"
    ok = 0
    seen = 0
    for record_i, (part, idx, raw) in enumerate(iter_uci_records(args.uci_mat_dir, parts=parts)):
        if args.limit_per_source > 0 and record_i >= args.limit_per_source:
            break
        if not in_shard(record_i, args.num_shards, args.shard_index):
            continue
        seen += 1
        rec = {
            "source": "uci",
            "subset": part,
            "subject_file": part,
            "index": idx,
            "path": str(Path(args.uci_mat_dir) / f"{part}.mat"),
            "status": "skip",
            "reason": "",
        }
        try:
            # UCI is commonly [PPG, ABP, ECG]. ABP is deliberately ignored.
            sig = np.stack([raw[:, 2], raw[:, 0]], axis=1).astype(np.float32)
            out_path = out_dir / f"uci_{part}_{idx:07d}.npz"
            n_beats = cache_arrays(
                sig,
                fs=args.uci_fs,
                input_cols=[0, 1],
                ecg_col=0,
                ppg_target_col=1,
                beat_len=args.beat_len,
                min_beats=args.min_beats,
                max_save_beats=args.max_save_beats,
                out_path=out_path,
                meta=rec,
                skip_existing=args.skip_existing,
                compress_npz=args.compress_npz,
            )
            rec.update({"status": "ok", "out": str(out_path), "n_beats": n_beats})
            ok += 1
        except Exception as exc:
            rec["reason"] = repr(exc)
        manifest.write(rec)
        if seen % args.log_every == 0:
            print(f"[uci] shard={args.shard_index}/{args.num_shards} seen={seen} ok={ok}", flush=True)
    print(f"[uci] finished shard={args.shard_index}/{args.num_shards} seen={seen} ok={ok} out_dir={out_dir}", flush=True)


def apply_script_config(args: argparse.Namespace) -> argparse.Namespace:
    if not USE_SCRIPT_CONFIG:
        return args

    args.out_root = OUT_ROOT
    args.enable_server = ENABLE_SERVER
    args.enable_pulsedb = ENABLE_PULSEDB
    args.enable_uci = ENABLE_UCI
    args.server_csv = [d["csv"] for d in SERVER_DATASETS if d.get("enabled", True)]
    args.server_root = [d["root"] for d in SERVER_DATASETS if d.get("enabled", True)]
    args.server_name = [d.get("name", f"server{i}") for i, d in enumerate(SERVER_DATASETS) if d.get("enabled", True)]
    args.server_fs = SERVER_FS
    args.file_col = SERVER_FILE_COL
    args.signal_path_col = SERVER_SIGNAL_PATH_COL
    args.signal_cols = SERVER_SIGNAL_COLS
    args.input_cols = SERVER_INPUT_COLS
    args.ecg_col = SERVER_ECG_COL
    args.ppg_target_col = SERVER_PPG_TARGET_COL

    args.pulsedb_root = PULSEDB_ROOT
    args.pulsedb_use_mimic = PULSEDB_USE_MIMIC
    args.pulsedb_use_vital = PULSEDB_USE_VITAL
    args.pulsedb_ecg_field = PULSEDB_ECG_FIELD
    args.pulsedb_ppg_field = PULSEDB_PPG_FIELD
    args.pulsedb_fs = PULSEDB_FS
    args.pulsedb_limit_subject_files = PULSEDB_LIMIT_SUBJECT_FILES
    args.pulsedb_limit_segments_per_file = PULSEDB_LIMIT_SEGMENTS_PER_FILE

    args.uci_mat_dir = UCI_MAT_DIR
    args.uci_parts = UCI_PARTS
    args.uci_fs = UCI_FS
    args.beat_len = BEAT_LEN
    args.min_beats = MIN_BEATS
    args.max_save_beats = MAX_SAVE_BEATS
    args.limit_per_source = LIMIT_PER_SOURCE
    args.log_every = LOG_EVERY
    args.skip_existing = SKIP_EXISTING
    args.compress_npz = COMPRESS_NPZ
    args.num_shards = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", NUM_SHARDS))
    args.shard_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", SHARD_INDEX))
    return args


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare multisource ECG+PPG foundation .npz cache")
    p.add_argument("--out-root", default=OUT_ROOT)
    p.add_argument("--enable-server", action="store_true", default=True)
    p.add_argument("--no-enable-server", dest="enable_server", action="store_false")
    p.add_argument("--enable-pulsedb", action="store_true", default=True)
    p.add_argument("--no-enable-pulsedb", dest="enable_pulsedb", action="store_false")
    p.add_argument("--enable-uci", action="store_true", default=True)
    p.add_argument("--no-enable-uci", dest="enable_uci", action="store_false")

    p.add_argument("--server-csv", action="append", default=[])
    p.add_argument("--server-root", action="append", default=[])
    p.add_argument("--server-name", action="append", default=[])
    p.add_argument("--server-fs", type=int, default=SERVER_FS)
    p.add_argument("--file-col", default=SERVER_FILE_COL)
    p.add_argument("--signal-path-col", default="")
    p.add_argument("--signal-cols", default=SERVER_SIGNAL_COLS)
    p.add_argument("--input-cols", default=SERVER_INPUT_COLS)
    p.add_argument("--ecg-col", type=int, default=0)
    p.add_argument("--ppg-target-col", type=int, default=1)

    p.add_argument("--pulsedb-root", default=PULSEDB_ROOT)
    p.add_argument("--pulsedb-use-mimic", action="store_true", default=True)
    p.add_argument("--no-pulsedb-use-mimic", dest="pulsedb_use_mimic", action="store_false")
    p.add_argument("--pulsedb-use-vital", action="store_true", default=True)
    p.add_argument("--no-pulsedb-use-vital", dest="pulsedb_use_vital", action="store_false")
    p.add_argument("--pulsedb-ecg-field", default=PULSEDB_ECG_FIELD)
    p.add_argument("--pulsedb-ppg-field", default=PULSEDB_PPG_FIELD)
    p.add_argument("--pulsedb-fs", type=int, default=PULSEDB_FS)
    p.add_argument("--pulsedb-limit-subject-files", type=int, default=0)
    p.add_argument("--pulsedb-limit-segments-per-file", type=int, default=0)

    p.add_argument("--uci-mat-dir", default=UCI_MAT_DIR)
    p.add_argument("--uci-parts", default=UCI_PARTS)
    p.add_argument("--uci-fs", type=int, default=UCI_FS)

    p.add_argument("--beat-len", type=int, default=BEAT_LEN)
    p.add_argument("--min-beats", type=int, default=MIN_BEATS)
    p.add_argument("--max-save-beats", type=int, default=MAX_SAVE_BEATS)
    p.add_argument("--limit-per-source", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.add_argument("--compress-npz", action="store_true", default=COMPRESS_NPZ)
    p.add_argument("--no-compress-npz", dest="compress_npz", action="store_false")
    p.add_argument("--num-shards", type=int, default=NUM_SHARDS)
    p.add_argument("--shard-index", type=int, default=SHARD_INDEX)
    args = p.parse_args(argv)
    return apply_script_config(args)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.num_shards > 1:
        manifest_path = out_root / f"manifest_shard{args.shard_index:03d}_of_{args.num_shards:03d}.csv"
    else:
        manifest_path = out_root / "manifest.csv"
    manifest = ManifestWriter(manifest_path)
    try:
        print(f"out_root={out_root}", flush=True)
        print("foundation-only cache: ECG/PPG only; BP labels are not training targets", flush=True)
        print(
            f"beat_len={args.beat_len} min_beats={args.min_beats} "
            f"max_save_beats={args.max_save_beats} compress_npz={args.compress_npz}",
            flush=True,
        )
        print(
            f"shard_index={args.shard_index} num_shards={args.num_shards} "
            f"manifest={manifest_path}",
            flush=True,
        )

        if args.enable_server:
            if len(args.server_csv) != len(args.server_root):
                raise ValueError("--server-csv and --server-root must have the same count")
            names = args.server_name or [f"server{i}" for i in range(len(args.server_csv))]
            for name, csv_path, root in zip(names, args.server_csv, args.server_root):
                convert_server_dataset(args, manifest, {"name": name, "csv": csv_path, "root": root})

        if args.enable_pulsedb:
            convert_pulsedb(args, manifest)

        if args.enable_uci:
            convert_uci(args, manifest)
    finally:
        manifest.close()


if __name__ == "__main__":
    main()
