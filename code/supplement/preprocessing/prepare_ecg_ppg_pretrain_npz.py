# -*- coding: utf-8 -*-
"""
Prepare ECG+PPG beat-level .npz files for ecg_ppg_multitask_pretrain.py.

The pretraining model expects one .npz per record/window with:
  beats:    (K, C, L) float32
  time_sec: (K,) float32
  pat:      (K,) float32
  morph:    (K, 6) float32
  hrv:      (3,) float32

Supported sources:
  1) server: CSV + .txt/.txt.gz signal files used in the previous BP pipeline
  2) uci:    E:/ECG_PPG/cuff+less+blood+pressure+estimation/Part_*.mat

Typical workflow:
  python prepare_ecg_ppg_pretrain_npz.py --source uci --mat-dir "E:\\ECG_PPG\\cuff+less+blood+pressure+estimation" --out-dir prepared_pretrain_npz/uci
  python prepare_ecg_ppg_pretrain_npz.py --source server --csv-path tptcom_data_existing_only.csv --signal-root <SERVER_SIGNAL_ROOT> --out-dir prepared_pretrain_npz/server_tptcom
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal


MORPH_DIM = 6
HRV_DIM = 3


def read_csv_robust(csv_path: str | os.PathLike) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin-1"]:
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")


def smart_open_text(path: str | os.PathLike):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def resolve_signal_path(signal_root: str | os.PathLike, rel_path) -> Optional[str]:
    if pd.isna(rel_path):
        return None
    p = str(rel_path).strip()
    if not p:
        return None
    if p.startswith("/") or p.startswith("\\"):
        p = p[1:]
    base = os.path.join(str(signal_root), p)
    candidates = [base]
    if base.endswith(".txt.gz"):
        candidates.append(base[:-3])
    elif base.endswith(".txt"):
        candidates.append(base + ".gz")
    elif base.endswith(".gz"):
        candidates.append(base[:-3])
    for item in candidates:
        if os.path.exists(item):
            return item
    return None


def load_server_signal_file(
    signal_path: str | os.PathLike,
    signal_cols: Sequence[str],
) -> np.ndarray:
    """Load the previous server .txt/.txt.gz format into (T, C)."""
    with smart_open_text(signal_path) as f:
        _schema_line = f.readline()
        _feat_line = f.readline()
        rows = []
        for line in f:
            line = line.strip()
            if line:
                rows.append(line.split(","))
    if not rows:
        raise ValueError(f"empty signal file: {signal_path}")

    df = pd.DataFrame(rows)
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)
    miss = [c for c in signal_cols if c not in df.columns]
    if miss:
        raise KeyError(f"{signal_path} missing columns: {miss}")
    sig = df[list(signal_cols)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    sig = np.where(np.isin(sig, [0, -1, -100]), np.nan, sig)
    return sig


def interp_nan(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    idx = np.arange(len(x))
    ok = np.isfinite(x)
    if ok.sum() >= 2 and (~ok).any():
        x = x.copy()
        x[~ok] = np.interp(idx[~ok], idx[ok], x[ok])
    return np.nan_to_num(x, nan=0.0).astype(np.float32)


def safe_bandpass(x: np.ndarray, fs: int, low: float, high: float, order: int = 3) -> np.ndarray:
    x = interp_nan(x)
    high = min(high, fs * 0.45)
    low = max(low, 0.01)
    if len(x) < fs or high <= low:
        return x.astype(np.float32)
    b, a = signal.butter(order, [low, high], btype="bandpass", fs=fs)
    return signal.filtfilt(b, a, x).astype(np.float32)


def robust_zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if finite.sum() < 10:
        return np.nan_to_num(x, nan=0.0).astype(np.float32)
    med = np.median(x[finite])
    mad = np.median(np.abs(x[finite] - med))
    return ((x - med) / (1.4826 * mad + eps)).astype(np.float32)


def preprocess_channels(sig_tc: np.ndarray, fs: int, ecg_col: int, input_cols: Sequence[int]) -> np.ndarray:
    out = []
    for col in input_cols:
        x = sig_tc[:, col]
        if col == ecg_col:
            y = safe_bandpass(x, fs=fs, low=0.5, high=40.0, order=3)
        else:
            y = safe_bandpass(x, fs=fs, low=0.3, high=8.0, order=3)
        out.append(robust_zscore(y))
    return np.stack(out, axis=1).astype(np.float32)


def detect_r_peaks(ecg_f: np.ndarray, fs: int, hr_min: int = 35, hr_max: int = 220) -> np.ndarray:
    if len(ecg_f) < fs:
        return np.array([], dtype=np.int64)
    der = np.diff(ecg_f, prepend=ecg_f[0])
    energy = der * der
    win = max(1, int(0.12 * fs))
    mwa = np.convolve(energy, np.ones(win, dtype=np.float32) / win, mode="same")

    med = np.median(mwa)
    mad = np.median(np.abs(mwa - med)) + 1e-8
    thr = med + 6.0 * 1.4826 * mad
    min_dist = max(1, int(fs * 60.0 / hr_max))
    peaks, _ = signal.find_peaks(mwa, height=thr, distance=min_dist)
    if len(peaks) == 0:
        return np.array([], dtype=np.int64)

    refine = []
    w = max(1, int(0.05 * fs))
    for p in peaks:
        l = max(0, p - w)
        r = min(len(ecg_f), p + w + 1)
        refine.append(l + int(np.argmax(ecg_f[l:r])))
    r_peaks = np.array(sorted(set(refine)), dtype=np.int64)
    if len(r_peaks) < 3:
        return r_peaks

    rr = np.diff(r_peaks) / float(fs)
    ok = np.ones(len(r_peaks), dtype=bool)
    bad = (rr < 60.0 / hr_max) | (rr > 60.0 / hr_min)
    for i in np.where(bad)[0]:
        ok[i] = False
        ok[i + 1] = False
    return r_peaks[ok]


def resample_segment(seg: np.ndarray, out_len: int) -> np.ndarray:
    seg = np.asarray(seg, dtype=np.float32)
    if len(seg) <= 1:
        return np.zeros(out_len, dtype=np.float32)
    t_old = np.linspace(0.0, 1.0, len(seg), endpoint=False)
    t_new = np.linspace(0.0, 1.0, out_len, endpoint=False)
    return np.interp(t_new, t_old, seg).astype(np.float32)


def estimate_pat_and_foot(
    ppg_f: np.ndarray,
    r_peak: int,
    fs: int,
    min_delay_sec: float = 0.08,
    max_delay_sec: float = 0.55,
) -> Tuple[float, Optional[int], Optional[int]]:
    """Return PAT seconds, foot index, and systolic peak index."""
    lo = r_peak + int(round(min_delay_sec * fs))
    hi = r_peak + int(round(max_delay_sec * fs))
    if lo < 0 or hi >= len(ppg_f) or hi <= lo + 2:
        return np.nan, None, None
    win = ppg_f[lo:hi]
    if not np.isfinite(win).all() or np.std(win) < 1e-6:
        return np.nan, None, None
    peak = lo + int(np.argmax(win))
    if peak <= lo + 1:
        return np.nan, None, peak
    foot = lo + int(np.argmin(ppg_f[lo:peak + 1]))
    pat = (foot - r_peak) / float(fs)
    return float(pat), int(foot), int(peak)


def estimate_morphology(
    ppg_f: np.ndarray,
    start: int,
    end: int,
    foot: Optional[int],
    peak: Optional[int],
    fs: int,
) -> np.ndarray:
    """Six simple PPG morphology targets: amp, PW50, rise, area, peak_pos, max_slope."""
    out = np.full(MORPH_DIM, np.nan, dtype=np.float32)
    if start < 0 or end > len(ppg_f) or end <= start + 4:
        return out
    seg = ppg_f[start:end]
    if not np.isfinite(seg).all() or np.std(seg) < 1e-6:
        return out

    seg_min = float(np.min(seg))
    seg_max = float(np.max(seg))
    amp = seg_max - seg_min
    if amp <= 1e-6:
        return out

    half = seg_min + 0.5 * amp
    above = np.where(seg >= half)[0]
    pw50 = (above[-1] - above[0]) / float(fs) if len(above) >= 2 else np.nan
    area = float(np.trapz(seg - seg_min) / (amp * max(len(seg), 1)))
    max_slope = float(np.max(np.diff(seg, prepend=seg[0])) * fs / amp)

    peak_pos = np.nan
    rise = np.nan
    if peak is not None and start <= peak < end:
        peak_pos = (peak - start) / float(end - start)
    if foot is not None and peak is not None and peak > foot:
        rise = (peak - foot) / float(fs)

    out[:] = [amp, pw50, rise, area, peak_pos, max_slope]
    return out


def compute_hrv(r_peaks: np.ndarray, fs: int) -> np.ndarray:
    if len(r_peaks) < 3:
        return np.full(HRV_DIM, np.nan, dtype=np.float32)
    rr = np.diff(r_peaks) / float(fs)
    if len(rr) < 2:
        return np.full(HRV_DIM, np.nan, dtype=np.float32)
    sdnn = float(np.std(rr))
    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    mean_hr = float(60.0 / np.mean(rr))
    return np.array([sdnn, rmssd, mean_hr], dtype=np.float32)


def extract_pretrain_npz_arrays(
    sig_tc: np.ndarray,
    fs: int,
    input_cols: Sequence[int],
    ecg_col: int,
    ppg_target_col: int,
    beat_len: int,
    min_beats: int = 6,
) -> Optional[Dict[str, np.ndarray]]:
    """Convert one continuous record into variable-length beat arrays."""
    if sig_tc.ndim != 2 or sig_tc.shape[0] < fs * 5:
        return None
    sig_tc = np.asarray(sig_tc, dtype=np.float32)
    processed = preprocess_channels(sig_tc, fs=fs, ecg_col=ecg_col, input_cols=input_cols)
    ecg_f = processed[:, list(input_cols).index(ecg_col)]

    # Use the original ppg_target_col if it is part of input_cols; otherwise filter separately.
    if ppg_target_col in input_cols:
        ppg_f = processed[:, list(input_cols).index(ppg_target_col)]
    else:
        ppg_f = robust_zscore(safe_bandpass(sig_tc[:, ppg_target_col], fs=fs, low=0.3, high=8.0, order=3))

    r_peaks = detect_r_peaks(ecg_f, fs=fs)
    if len(r_peaks) < min_beats + 2:
        return None

    beats = []
    time_sec = []
    pats = []
    morphs = []
    first_time = None

    for i in range(1, len(r_peaks) - 1):
        r = int(r_peaks[i])
        start = int((r_peaks[i - 1] + r_peaks[i]) // 2)
        end = int((r_peaks[i] + r_peaks[i + 1]) // 2)
        dur = (end - start) / float(fs)
        if dur < 0.35 or dur > 2.0 or start < 0 or end > len(processed):
            continue

        pat, foot, peak = estimate_pat_and_foot(ppg_f, r_peak=r, fs=fs)
        beat_ch = [resample_segment(processed[start:end, ch], beat_len) for ch in range(processed.shape[1])]
        beats.append(np.stack(beat_ch, axis=0))
        t = r / float(fs)
        if first_time is None:
            first_time = t
        time_sec.append(t - first_time)
        pats.append(pat)
        morphs.append(estimate_morphology(ppg_f, start=start, end=end, foot=foot, peak=peak, fs=fs))

    if len(beats) < min_beats:
        return None

    return {
        "beats": np.stack(beats, axis=0).astype(np.float32),
        "time_sec": np.asarray(time_sec, dtype=np.float32),
        "pat": np.asarray(pats, dtype=np.float32),
        "morph": np.stack(morphs, axis=0).astype(np.float32),
        "hrv": compute_hrv(r_peaks, fs=fs),
    }


def save_npz(out_path: Path, arrays: Dict[str, np.ndarray], meta: Dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays, meta=json.dumps(meta, ensure_ascii=False))


def limit_saved_beats(arrays: Dict[str, np.ndarray], max_save_beats: int) -> Dict[str, np.ndarray]:
    """Keep a deterministic subset of beats to avoid very large cache files."""
    if max_save_beats <= 0:
        return arrays
    n = int(arrays["beats"].shape[0])
    if n <= max_save_beats:
        return arrays
    idx = np.linspace(0, n - 1, num=max_save_beats).astype(np.int64)
    out = dict(arrays)
    for key in ["beats", "time_sec", "pat", "morph"]:
        out[key] = arrays[key][idx]
    return out


def convert_server(args: argparse.Namespace) -> None:
    df = read_csv_robust(args.csv_path)
    signal_cols = [s.strip() for s in args.signal_cols.split(",") if s.strip()]
    if args.signal_path_col and args.signal_path_col in df.columns:
        paths = df[args.signal_path_col].tolist()
    else:
        if args.file_col not in df.columns:
            raise KeyError(f"CSV missing {args.file_col}; pass --file-col or --signal-path-col")
        paths = [resolve_signal_path(args.signal_root, p) for p in df[args.file_col]]

    input_cols = [int(x) for x in args.input_cols.split(",")]
    out_dir = Path(args.out_dir)
    rows = []
    ok_count = 0
    for idx, path in enumerate(paths):
        if args.limit and idx >= args.limit:
            break
        rec = {"source": args.source_name, "index": idx, "path": str(path), "status": "skip", "reason": ""}
        try:
            if path is None or not os.path.exists(str(path)):
                raise FileNotFoundError(str(path))
            sig = load_server_signal_file(path, signal_cols=signal_cols)
            arrays = extract_pretrain_npz_arrays(
                sig,
                fs=args.fs,
                input_cols=input_cols,
                ecg_col=args.ecg_col,
                ppg_target_col=args.ppg_target_col,
                beat_len=args.beat_len,
                min_beats=args.min_beats,
            )
            if arrays is None:
                raise ValueError("not enough valid beats")
            arrays = limit_saved_beats(arrays, args.max_save_beats)
            out_path = out_dir / f"{args.source_name}_{idx:06d}.npz"
            save_npz(out_path, arrays, meta=rec)
            rec.update({"status": "ok", "out": str(out_path), "n_beats": int(arrays["beats"].shape[0])})
            ok_count += 1
        except Exception as exc:
            rec["reason"] = repr(exc)
        rows.append(rec)
        if (idx + 1) % args.log_every == 0:
            print(f"[server] seen={idx + 1} ok={ok_count}")
    pd.DataFrame(rows).to_csv(out_dir / f"{args.source_name}_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"[server] finished: ok={ok_count} out_dir={out_dir}")


def iter_uci_records(mat_dir: str | os.PathLike, parts: Sequence[str]) -> Iterator[Tuple[str, int, np.ndarray]]:
    import h5py

    for part_name in parts:
        mat_path = Path(mat_dir) / part_name
        with h5py.File(mat_path, "r") as f:
            key = Path(part_name).stem
            refs = f[key]
            for i in range(refs.shape[0]):
                ref = refs[i, 0] if refs.ndim == 2 else refs[i]
                arr = np.asarray(f[ref], dtype=np.float32)
                if arr.ndim != 2:
                    continue
                if arr.shape[0] == 3 and arr.shape[1] > 3:
                    arr = arr.T
                if arr.shape[1] < 3:
                    continue
                yield key, i, arr


def convert_uci(args: argparse.Namespace) -> None:
    # UCI cuff-less dataset columns are typically [PPG, ABP, ECG].
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    out_dir = Path(args.out_dir)
    rows = []
    ok_count = 0
    seen = 0
    for part, idx, raw in iter_uci_records(args.mat_dir, parts=parts):
        if args.limit and seen >= args.limit:
            break
        seen += 1
        rec = {"source": args.source_name, "part": part, "index": idx, "status": "skip", "reason": ""}
        try:
            # Build (T, 2): ECG first, PPG second. ABP is not used for self-supervised pretraining.
            sig = np.stack([raw[:, 2], raw[:, 0]], axis=1).astype(np.float32)
            arrays = extract_pretrain_npz_arrays(
                sig,
                fs=args.fs,
                input_cols=[0, 1],
                ecg_col=0,
                ppg_target_col=1,
                beat_len=args.beat_len,
                min_beats=args.min_beats,
            )
            if arrays is None:
                raise ValueError("not enough valid beats")
            arrays = limit_saved_beats(arrays, args.max_save_beats)
            out_path = out_dir / f"{args.source_name}_{part}_{idx:06d}.npz"
            meta = dict(rec)
            # Optional record-level ABP summary for future supervised BP experiments.
            meta["abp_sbp_p95"] = float(np.nanpercentile(raw[:, 1], 95))
            meta["abp_dbp_p05"] = float(np.nanpercentile(raw[:, 1], 5))
            save_npz(out_path, arrays, meta=meta)
            rec.update({"status": "ok", "out": str(out_path), "n_beats": int(arrays["beats"].shape[0])})
            ok_count += 1
        except Exception as exc:
            rec["reason"] = repr(exc)
        rows.append(rec)
        if seen % args.log_every == 0:
            print(f"[uci] seen={seen} ok={ok_count}")
    pd.DataFrame(rows).to_csv(out_dir / f"{args.source_name}_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"[uci] finished: seen={seen} ok={ok_count} out_dir={out_dir}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare ECG+PPG .npz files for multitask pretraining")
    p.add_argument("--source", choices=["server", "uci"], required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--source-name", default="")
    p.add_argument("--fs", type=int, default=250)
    p.add_argument("--beat-len", type=int, default=256)
    p.add_argument("--max-save-beats", type=int, default=128)
    p.add_argument("--min-beats", type=int, default=6)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)

    # Server CSV + signal-file options.
    p.add_argument("--csv-path", default="")
    p.add_argument("--signal-root", default="")
    p.add_argument("--file-col", default="ECG+PPG文件索引")
    p.add_argument("--signal-path-col", default="")
    p.add_argument("--signal-cols", default="ecgor,ppgir,ppgrd,ppgot")
    p.add_argument("--input-cols", default="0,1,2,3", help="Column indices after --signal-cols loading")
    p.add_argument("--ecg-col", type=int, default=0)
    p.add_argument("--ppg-target-col", type=int, default=1)

    # UCI .mat options.
    p.add_argument("--mat-dir", default="")
    p.add_argument("--parts", default="Part_1.mat,Part_2.mat,Part_3.mat,Part_4.mat")

    args = p.parse_args(argv)
    if not args.source_name:
        args.source_name = args.source
    if args.source == "uci" and args.fs == 250:
        # The public UCI cuff-less BP dataset is commonly sampled at 125 Hz.
        args.fs = 125
    return args


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    if args.source == "server":
        if not args.csv_path:
            raise ValueError("--csv-path is required for --source server")
        if not args.signal_path_col and not args.signal_root:
            raise ValueError("--signal-root is required unless --signal-path-col points to absolute paths")
        convert_server(args)
    elif args.source == "uci":
        if not args.mat_dir:
            raise ValueError("--mat-dir is required for --source uci")
        convert_uci(args)
    else:
        raise ValueError(args.source)


if __name__ == "__main__":
    main()
