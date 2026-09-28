from __future__ import annotations

import argparse
import gzip
import hashlib
import math
import os
import socket
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


DEFAULT_BP_DATA_ROOT = "/data-ai/sl20200894/AnyPPGData"
DEFAULT_BP_NPZ_CACHE_ROOT = "/data-ai/sl20200894/AnyPPGData_npz_cache/orthogonal_foundation_bp_predict_fast20260601_npz"
DEFAULT_FOUNDATION_CKPT_PATH = (
    "/data-ai/sl20200894/Code/foundation_dual_view_alignment_packed_bundle_20260628/"
    "dual_view_virtual_r_alignment_packed_amp.pt"
)

TRAIN_SOURCE_NAMES = ["tptcom", "sysu"]
TEST_SOURCE_NAMES = ["predict"]
ECG_PPG_CHANNELS = [0, 1]
CHANNEL_DISPLAY_NAMES = ["ecg", "ppg_ir", "ppg_rd", "ppg_ot"]
FINETUNE_MODES = ("head", "last1", "last3", "full")
DEFAULT_MEDICATION_COLUMNS = ["服药类别", "服药信息", "服药剂量(mg)", "服药时间"]
SBP_CANDIDATES = [
    "y_sbp",
    "Sbp",
    "SBP",
    "测量收缩压护士2后",
    "测量收缩压护士1后",
    "测量收缩压护士2前",
    "测量收缩压护士1前",
    "电子血压计收缩压1",
    "电子血压计收缩压2",
    "收缩压",
    "sbp",
]
DBP_CANDIDATES = [
    "y_dbp",
    "Dbp",
    "DBP",
    "测量舒张压护士2后",
    "测量舒张压护士1后",
    "测量舒张压护士2前",
    "测量舒张压护士1前",
    "电子血压计舒张压1",
    "电子血压计舒张压2",
    "舒张压",
    "dbp",
]


# ----------------------------- DDP runtime -----------------------------


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    return get_rank() == 0


def main_print(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs)


def visible_gpu_count() -> int:
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def setup_distributed() -> int:
    if not is_distributed():
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA GPUs.")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    visible_count = torch.cuda.device_count()
    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    if visible_count <= 0:
        raise RuntimeError(
            "torch.cuda.is_available() is True, but torch.cuda.device_count() returned 0. "
            f"CUDA_VISIBLE_DEVICES={visible_env}"
        )
    if local_rank >= visible_count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {visible_count} CUDA device(s) are visible. "
            f"CUDA_VISIBLE_DEVICES={visible_env}"
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def get_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def auto_num_workers(requested: int, world_size: int) -> int:
    if requested >= 0:
        return requested
    cpu_count = os.cpu_count() or 4
    per_proc = max(1, cpu_count // max(1, world_size))
    return min(8, per_proc)


def build_loader_kwargs(num_workers: int, pin_memory: bool, prefetch_factor: int) -> dict:
    kwargs = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def worker_init_fn(worker_id: int, base_seed: int, rank: int) -> None:
    worker_seed = int(base_seed + rank * 1000 + worker_id)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "rng"):
        info.dataset.rng = np.random.RandomState(worker_seed)


def make_worker_init_fn(base_seed: int, rank: int = 0) -> Callable[[int], None]:
    return partial(worker_init_fn, base_seed=base_seed, rank=rank)


def all_reduce_sum_(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def should_auto_spawn(auto_ddp: bool) -> bool:
    return auto_ddp and (not is_distributed()) and visible_gpu_count() > 1


def spawn_entry(local_rank: int, world_size: int, worker_fn: Callable, args, shared_state) -> None:
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    setup_distributed()
    try:
        worker_fn(local_rank, args, shared_state)
    finally:
        cleanup_distributed()


def launch(worker_fn: Callable, args, shared_state: Optional[Tuple] = None) -> None:
    if is_distributed():
        local_rank = setup_distributed()
        try:
            worker_fn(local_rank, args, shared_state)
        finally:
            cleanup_distributed()
        return

    if should_auto_spawn(getattr(args, "auto_ddp", True)):
        world_size = visible_gpu_count()
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        if not os.environ.get("MASTER_PORT"):
            port = int(getattr(args, "master_port", 0) or 0)
            os.environ["MASTER_PORT"] = str(port if port > 0 else find_free_port())
        mp.spawn(spawn_entry, nprocs=world_size, args=(world_size, worker_fn, args, shared_state), join=True)
        return

    worker_fn(0, args, shared_state)


# ----------------------------- CSV and signal IO -----------------------------


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv_robust(csv_path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "gb18030", "gbk", "latin-1"]:
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")


def normalize_col_name(x: str) -> str:
    return (
        str(x)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("（", "(")
        .replace("）", ")")
    )


def pick_column_relaxed(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df.empty:
        return None
    exact = {str(c): c for c in df.columns}
    for cand in candidates:
        if cand in exact:
            return exact[cand]
    norm_map = {normalize_col_name(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_col_name(cand)
        if key in norm_map:
            return norm_map[key]
    for cand in candidates:
        key = normalize_col_name(cand)
        for col in df.columns:
            col_key = normalize_col_name(col)
            if key and (key in col_key or col_key in key):
                return col
    return None


def make_data_sources(bp_data_root: str) -> List[dict]:
    csv_root = os.path.join(bp_data_root, "data")
    if not os.path.isdir(csv_root):
        csv_root = bp_data_root
    return [
        {
            "csv_path": os.path.join(csv_root, "tptcom_data_existing_only.csv"),
            "signal_root": os.path.join(bp_data_root, "tptcom"),
            "name": "tptcom",
        },
        {
            "csv_path": os.path.join(csv_root, "sysu_data_existing_only.csv"),
            "signal_root": os.path.join(bp_data_root, "sysu"),
            "name": "sysu",
        },
        {
            "csv_path": os.path.join(csv_root, "Predict_data_existing_only__intersect_source.csv"),
            "signal_root": os.path.join(bp_data_root, "predict"),
            "name": "predict",
        },
    ]


def select_sources_by_names(sources: List[dict], names: List[str]) -> List[dict]:
    selected = [s for s in sources if s.get("name") in set(names)]
    miss = [n for n in names if n not in {s.get("name") for s in selected}]
    if miss:
        raise ValueError(f"Missing source name(s): {miss}")
    return selected


def remap_absolute_signal_path(path: str, data_root: Optional[str], source_name: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = str(path).strip()
    if not p:
        return None
    if os.path.exists(p):
        return p
    if not data_root:
        return None

    normalized = p.replace("\\", "/")
    markers = ["/BP/", "/AnyPPGData/"]
    for marker in markers:
        if marker in normalized:
            suffix = normalized.split(marker, 1)[1].lstrip("/")
            candidate = os.path.join(data_root, suffix)
            if os.path.exists(candidate):
                return candidate

    if source_name:
        parts = normalized.strip("/").split("/")
        if source_name in parts:
            idx = parts.index(source_name)
            candidate = os.path.join(data_root, *parts[idx:])
            if os.path.exists(candidate):
                return candidate
    return None


def resolve_signal_path(
    signal_root: str,
    rel_path,
    *,
    data_root: Optional[str] = None,
    source_name: Optional[str] = None,
) -> Optional[str]:
    if pd.isna(rel_path):
        return None
    p = str(rel_path).strip()
    if not p:
        return None

    mapped = remap_absolute_signal_path(p, data_root=data_root, source_name=source_name)
    if mapped:
        return mapped

    if p.startswith("/"):
        p = p[1:]
    if source_name:
        parts = p.replace("\\", "/").strip("/").split("/")
        if source_name in parts:
            idx = parts.index(source_name)
            candidate = os.path.join(data_root or os.path.dirname(signal_root), *parts[idx:])
            if os.path.exists(candidate):
                return candidate

    base = os.path.join(signal_root, p)
    candidates = [base]
    if base.endswith(".txt.gz"):
        candidates.append(base[:-3])
    elif base.endswith(".txt"):
        candidates.append(base + ".gz")
    if base.endswith(".gz") and not base.endswith(".txt.gz"):
        candidates.append(base[:-3])
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def smart_open_text(path: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def to_float_array(values) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float32)


def preprocess_signals_outliers(
    x: pd.DataFrame,
    signal_cols: List[str],
    sentinels=(0, -1, -100),
    method: str = "mad",
    action: str = "clip",
    mad_z: float = 6.0,
    iqr_k: float = 3.0,
    min_valid: int = 50,
) -> pd.DataFrame:
    out = x.copy()
    for c in signal_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if sentinels is not None:
        out[signal_cols] = out[signal_cols].replace(list(sentinels), np.nan)

    for c in signal_cols:
        s = out[c]
        valid = s.dropna()
        if len(valid) < min_valid:
            continue
        if method.lower() == "iqr":
            q1 = valid.quantile(0.25)
            q3 = valid.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lo = q1 - iqr_k * iqr
            hi = q3 + iqr_k * iqr
        else:
            med = valid.median()
            mad = (valid - med).abs().median()
            if mad == 0:
                continue
            lo = med - (mad_z / 0.6745) * mad
            hi = med + (mad_z / 0.6745) * mad
        if action == "clip":
            out[c] = s.clip(lo, hi)
        elif action == "nan":
            out.loc[(s < lo) | (s > hi), c] = np.nan
        elif action == "drop":
            out = out.loc[~((s < lo) | (s > hi))].copy()
        else:
            raise ValueError("action must be clip, nan, or drop")
    return out


def autofix_24bit_unsigned(x, tol=2e5):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x.astype(np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return x.astype(np.float32)
    med = np.nanmedian(x[finite])
    full = 1 << 24
    if med > 1e6 and abs(full - med) < tol:
        out = x.astype(np.float32, copy=True)
        xi = out[finite].astype(np.int64)
        sign = 1 << 23
        xi[xi >= sign] -= full
        out[finite] = xi.astype(np.float32)
        return out
    return x.astype(np.float32)


def resample_to(x, fs_in: int, fs_out: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if int(fs_in) == int(fs_out):
        return x
    from math import gcd

    g = gcd(int(fs_in), int(fs_out))
    up = int(fs_out // g)
    down = int(fs_in // g)
    return signal.resample_poly(x, up=up, down=down).astype(np.float32)


def interpolate_nans(x, max_gap=0.5, fs=250) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    isn = ~np.isfinite(x)
    if not isn.any():
        return x
    idx = np.arange(len(x))
    good = np.isfinite(x)
    if good.sum() < 2:
        return x
    out = x.copy()
    out[~good] = np.interp(idx[~good], idx[good], x[good])
    max_len = int(max_gap * fs)
    nan_runs = np.diff(np.concatenate([[0], isn.astype(int), [0]]))
    starts = np.where(nan_runs == 1)[0]
    ends = np.where(nan_runs == -1)[0]
    for s, e in zip(starts, ends):
        if (e - s) > max_len:
            out[s:e] = np.nan
    return out


def winsorize_by_quantile(x, q_low=0.005, q_high=0.995) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if finite.sum() < 10:
        return x
    lo = np.quantile(x[finite], q_low)
    hi = np.quantile(x[finite], q_high)
    return np.clip(x, lo, hi)


def notch_filter(x, fs: int, f0=50.0, q=30.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    b, a = signal.iirnotch(w0=f0, Q=q, fs=fs)
    return signal.filtfilt(b, a, x).astype(np.float32)


def butter_bandpass_filter(x, fs: int, low: float, high: float, order=4) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    b, a = signal.butter(order, [low, high], btype="bandpass", fs=fs)
    return signal.filtfilt(b, a, x).astype(np.float32)


def robust_zscore(x, eps=1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if finite.sum() < 10:
        return np.nan_to_num(x, nan=0.0).astype(np.float32)
    med = np.median(x[finite])
    mad = np.median(np.abs(x[finite] - med))
    scale = 1.4826 * mad + eps
    return ((x - med) / scale).astype(np.float32)


def preprocess_ecg_io(ecg, fs_in: int, fs_out=250, notch_hz=50.0) -> np.ndarray:
    x = autofix_24bit_unsigned(ecg)
    x = interpolate_nans(x, max_gap=0.5, fs=int(fs_in))
    x = winsorize_by_quantile(x, 0.002, 0.998)
    x = resample_to(x, fs_in=int(fs_in), fs_out=int(fs_out))
    if notch_hz is not None:
        x = notch_filter(x, fs=int(fs_out), f0=float(notch_hz), q=30.0)
    x = butter_bandpass_filter(x, fs=int(fs_out), low=0.5, high=40.0, order=4)
    return robust_zscore(x)


def preprocess_ppg_io(ppg, fs_in: int, fs_out=250, notch_hz=None) -> np.ndarray:
    x = autofix_24bit_unsigned(ppg)
    x = interpolate_nans(x, max_gap=0.5, fs=int(fs_in))
    x = winsorize_by_quantile(x, 0.002, 0.998)
    x = resample_to(x, fs_in=int(fs_in), fs_out=int(fs_out))
    if notch_hz is not None:
        x = notch_filter(x, fs=int(fs_out), f0=float(notch_hz), q=30.0)
    x = butter_bandpass_filter(x, fs=int(fs_out), low=0.3, high=8.0, order=4)
    return robust_zscore(x)


def detect_signal_cols(columns: List[str]) -> List[str]:
    cols = [str(c).strip() for c in columns]
    std = ["ecgor", "ppgir", "ppgrd", "ppgot"]
    if all(c in cols for c in std):
        return std
    lower = {c.lower(): c for c in cols}
    out = []
    for key_group in [
        ["ecgor", "ecg", "ecg_r"],
        ["ppgir", "ppg_ir", "ppg1"],
        ["ppgrd", "ppg_rd", "ppg2"],
        ["ppgot", "ppg_ot", "ppg3"],
    ]:
        hit = None
        for key in key_group:
            if key in lower:
                hit = lower[key]
                break
            for c in cols:
                if key in c.lower():
                    hit = c
                    break
            if hit:
                break
        if hit and hit not in out:
            out.append(hit)
    for c in cols:
        if len(out) >= 4:
            break
        if c not in out:
            out.append(c)
    return out[:4]


def load_signal_table(path: str) -> Tuple[pd.DataFrame, str]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with smart_open_text(path) as f:
        schema_line = f.readline().strip()
        feat_line = f.readline().strip()
        rows = [line.strip().split(",") for line in f if line.strip()]
    if not rows:
        raise ValueError(f"Empty data rows in {path}")

    feat_cols = [c.strip() for c in feat_line.split(",") if c.strip()]
    first = [c.strip() for c in rows[0]]
    first_is_header = any(any(ch.isalpha() for ch in cell) for cell in first)
    if first_is_header:
        cols = first
        data_rows = rows[1:]
    elif len(feat_cols) >= 4:
        cols = feat_cols
        data_rows = rows
    else:
        cols = [f"ch{i}" for i in range(len(rows[0]))]
        data_rows = rows
    df = pd.DataFrame(data_rows)
    if df.shape[1] < len(cols):
        for _ in range(len(cols) - df.shape[1]):
            df[df.shape[1]] = np.nan
    df = df.iloc[:, : len(cols)]
    df.columns = cols
    return df, schema_line


def load_signals_robust(gz_path: str, fs_in: int = 250, fs_out: int = 250, apply_preprocessing: bool = True) -> np.ndarray:
    df, _ = load_signal_table(gz_path)
    signal_cols = detect_signal_cols(list(df.columns))
    if len(signal_cols) < 4:
        raise KeyError(f"{gz_path} has fewer than 4 signal columns")
    df = df[signal_cols].apply(pd.to_numeric, errors="coerce")
    df = df.replace([0, -1, -100], np.nan)
    df = preprocess_signals_outliers(df, signal_cols=signal_cols, sentinels=(0, -1, -100), action="clip", mad_z=6.0)
    sig = df.to_numpy(dtype=np.float32)
    if sig.shape[1] < 4:
        sig = np.hstack([sig, np.zeros((len(sig), 4 - sig.shape[1]), dtype=np.float32)])
    if apply_preprocessing:
        ch0 = preprocess_ecg_io(sig[:, 0], fs_in=fs_in, fs_out=fs_out, notch_hz=50.0)
        ch1 = preprocess_ppg_io(sig[:, 1], fs_in=fs_in, fs_out=fs_out, notch_hz=None)
        ch2 = preprocess_ppg_io(sig[:, 2], fs_in=fs_in, fs_out=fs_out, notch_hz=None)
        ch3 = preprocess_ppg_io(sig[:, 3], fs_in=fs_in, fs_out=fs_out, notch_hz=None)
        length = min(len(ch0), len(ch1), len(ch2), len(ch3))
        sig = np.column_stack([ch0[:length], ch1[:length], ch2[:length], ch3[:length]]).astype(np.float32)
    return sig


def fill_nan_interp_then_zero(sig_tc: np.ndarray) -> np.ndarray:
    df_sig = pd.DataFrame(sig_tc)
    df_sig = df_sig.interpolate(limit_direction="both", axis=0)
    sig = df_sig.to_numpy(dtype=np.float32)
    return np.nan_to_num(sig, nan=0.0)


# ----------------------------- Foundation-style beat extraction -----------------------------


PSEUDO_BEAT_SEC = 0.5


def load_raw_signal_for_foundation(gz_path: str, min_cols: int = 2) -> np.ndarray:
    """Load raw ECG/PPG columns exactly before foundation preprocessing."""
    df, _ = load_signal_table(gz_path)
    signal_cols = detect_signal_cols(list(df.columns))
    if len(signal_cols) < min_cols:
        raise KeyError(f"{gz_path} has fewer than {min_cols} signal columns")
    sig = df[signal_cols[:min_cols]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    sig = np.where(np.isin(sig, [0, -1, -100]), np.nan, sig)
    return sig.astype(np.float32)


def foundation_interp_nan(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    idx = np.arange(len(x))
    ok = np.isfinite(x)
    if ok.sum() >= 2 and (~ok).any():
        x = x.copy()
        x[~ok] = np.interp(idx[~ok], idx[ok], x[ok])
    return np.nan_to_num(x, nan=0.0).astype(np.float32)


def foundation_safe_bandpass(x: np.ndarray, fs: int, low: float, high: float, order: int = 3) -> np.ndarray:
    x = foundation_interp_nan(x)
    high = min(float(high), float(fs) * 0.45)
    low = max(float(low), 0.01)
    if len(x) < int(fs) or high <= low:
        return x.astype(np.float32)
    b, a = signal.butter(order, [low, high], btype="bandpass", fs=int(fs))
    return signal.filtfilt(b, a, x).astype(np.float32)


def foundation_preprocess_channels(
    sig_tc: np.ndarray,
    fs: int,
    ecg_col: int,
    input_cols: Sequence[int],
) -> np.ndarray:
    out = []
    sig_tc = np.asarray(sig_tc, dtype=np.float32)
    for col in input_cols:
        x = sig_tc[:, int(col)]
        if int(col) == int(ecg_col):
            y = foundation_safe_bandpass(x, fs=fs, low=0.5, high=40.0, order=3)
        else:
            y = foundation_safe_bandpass(x, fs=fs, low=0.3, high=8.0, order=3)
        out.append(robust_zscore(y))
    return np.stack(out, axis=1).astype(np.float32)


def foundation_resample_segment(seg: np.ndarray, out_len: int) -> np.ndarray:
    seg = np.asarray(seg, dtype=np.float32)
    if len(seg) <= 1:
        return np.zeros(out_len, dtype=np.float32)
    t_old = np.linspace(0.0, 1.0, len(seg), endpoint=False)
    t_new = np.linspace(0.0, 1.0, out_len, endpoint=False)
    return np.interp(t_new, t_old, seg).astype(np.float32)


def foundation_zscore_per_beat(beats: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    beats = np.asarray(beats, dtype=np.float32)
    mean = beats.mean(axis=-1, keepdims=True)
    std = beats.std(axis=-1, keepdims=True)
    return ((beats - mean) / (std + eps)).astype(np.float32)


def foundation_ensure_2d_signal(sig_tc: np.ndarray, min_cols: int) -> np.ndarray:
    sig = np.asarray(sig_tc, dtype=np.float32)
    if sig.ndim == 1:
        sig = sig[:, None]
    if sig.ndim != 2:
        sig = sig.reshape(-1, 1)
    if sig.shape[0] < 2:
        sig = np.pad(sig, ((0, 2 - sig.shape[0]), (0, 0)), mode="constant")
    if sig.shape[1] < min_cols:
        sig = np.pad(sig, ((0, 0), (0, min_cols - sig.shape[1])), mode="constant")
    return sig.astype(np.float32)


def foundation_detect_r_peaks_lenient(ecg_f: np.ndarray, fs: int) -> np.ndarray:
    ecg = np.asarray(ecg_f, dtype=np.float32)
    if len(ecg) < 3:
        return np.array([], dtype=np.int64)

    der = np.diff(ecg, prepend=ecg[0])
    energy = der * der
    win = max(1, int(0.10 * max(int(fs), 1)))
    mwa = np.convolve(energy, np.ones(win, dtype=np.float32) / win, mode="same")

    finite = np.isfinite(mwa)
    if finite.sum() < 3:
        return np.array([], dtype=np.int64)
    med = float(np.median(mwa[finite]))
    mad = float(np.median(np.abs(mwa[finite] - med))) + 1e-8
    thr = med + 3.0 * 1.4826 * mad

    peaks, _ = signal.find_peaks(mwa, height=thr, distance=max(1, int(0.18 * max(int(fs), 1))))
    if len(peaks) == 0:
        peaks, _ = signal.find_peaks(ecg, distance=max(1, int(0.18 * max(int(fs), 1))))
    if len(peaks) == 0:
        return np.array([], dtype=np.int64)

    refine = []
    w = max(1, int(0.05 * max(int(fs), 1)))
    for p in peaks:
        left = max(0, int(p) - w)
        right = min(len(ecg), int(p) + w + 1)
        if right > left:
            refine.append(left + int(np.argmax(ecg[left:right])))
    return np.array(sorted(set(refine)), dtype=np.int64)


def foundation_arrays_from_segments(
    processed: np.ndarray,
    fs: int,
    beat_len: int,
    starts: Sequence[int],
    ends: Sequence[int],
    centers: Sequence[int],
) -> Optional[Dict[str, np.ndarray]]:
    beats = []
    time_sec = []
    first_time = None
    n = int(processed.shape[0])

    for start, end, center in zip(starts, ends, centers):
        start = int(np.clip(start, 0, max(n - 1, 0)))
        end = int(np.clip(end, start + 1, n))
        center = int(np.clip(center, start, max(end - 1, start)))
        if end <= start:
            continue
        beat_ch = [foundation_resample_segment(processed[start:end, ch], beat_len) for ch in range(processed.shape[1])]
        beats.append(np.stack(beat_ch, axis=0))
        t = center / float(max(int(fs), 1))
        if first_time is None:
            first_time = t
        time_sec.append(t - first_time)

    if not beats:
        return None
    return {
        "beats": np.stack(beats, axis=0).astype(np.float32),
        "time_sec": np.asarray(time_sec, dtype=np.float32),
    }


def foundation_arrays_from_r_peaks(
    processed: np.ndarray,
    r_peaks: np.ndarray,
    fs: int,
    beat_len: int,
) -> Optional[Dict[str, np.ndarray]]:
    r_peaks = np.asarray(r_peaks, dtype=np.int64)
    if len(r_peaks) == 0:
        return None

    n = int(processed.shape[0])
    if len(r_peaks) >= 2:
        rr_default = int(np.median(np.diff(r_peaks)))
    else:
        rr_default = int(round(PSEUDO_BEAT_SEC * max(int(fs), 1)))
    rr_default = max(rr_default, 2)

    starts = []
    ends = []
    centers = []
    for i, r in enumerate(r_peaks):
        prev_r = int(r_peaks[i - 1]) if i > 0 else int(r) - rr_default
        next_r = int(r_peaks[i + 1]) if i + 1 < len(r_peaks) else int(r) + rr_default
        start = int(round((prev_r + int(r)) / 2.0))
        end = int(round((int(r) + next_r) / 2.0))
        start = max(0, start)
        end = min(n, end)
        if end <= start:
            half = max(1, rr_default // 2)
            start = max(0, int(r) - half)
            end = min(n, int(r) + half)
        if end > start:
            starts.append(start)
            ends.append(end)
            centers.append(int(r))

    return foundation_arrays_from_segments(processed, fs, beat_len, starts, ends, centers)


def foundation_arrays_from_even_windows(processed: np.ndarray, fs: int, beat_len: int) -> Dict[str, np.ndarray]:
    n = int(processed.shape[0])
    duration = n / float(max(int(fs), 1))
    count = max(1, int(np.ceil(duration / PSEUDO_BEAT_SEC)))
    edges = np.linspace(0, n, num=count + 1)
    starts = np.floor(edges[:-1]).astype(np.int64)
    ends = np.ceil(edges[1:]).astype(np.int64)
    centers = ((starts + ends) // 2).astype(np.int64)
    arrays = foundation_arrays_from_segments(processed, fs, beat_len, starts, ends, centers)
    if arrays is not None:
        return arrays
    return {
        "beats": np.zeros((1, processed.shape[1], beat_len), dtype=np.float32),
        "time_sec": np.zeros((1,), dtype=np.float32),
    }


def extract_foundation_beat_arrays(
    sig_tc: np.ndarray,
    fs: int,
    input_cols: Sequence[int],
    ecg_col: int,
    beat_len: int,
) -> Dict[str, np.ndarray]:
    """Match the permissive foundation preprocessing used before cached pretraining."""
    fs = int(fs) if int(fs) > 0 else 250
    input_cols = list(input_cols)
    min_cols = max(input_cols + [int(ecg_col)]) + 1
    sig = foundation_ensure_2d_signal(sig_tc, min_cols=min_cols)
    processed = foundation_preprocess_channels(sig, fs=fs, ecg_col=ecg_col, input_cols=input_cols)
    ecg_idx = input_cols.index(ecg_col) if ecg_col in input_cols else 0
    r_peaks = foundation_detect_r_peaks_lenient(processed[:, ecg_idx], fs=fs)
    arrays = foundation_arrays_from_r_peaks(processed, r_peaks, fs=fs, beat_len=beat_len)
    if arrays is not None:
        return arrays
    return foundation_arrays_from_even_windows(processed, fs=fs, beat_len=beat_len)


def sample_or_pad_foundation_beats(
    beats: np.ndarray,
    time_sec: np.ndarray,
    max_beats: int,
    rng: np.random.RandomState,
    training: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    beats = np.asarray(beats, dtype=np.float32)
    time_sec = np.asarray(time_sec, dtype=np.float32)
    if beats.ndim != 3 or beats.shape[0] < 1:
        raise ValueError(f"invalid foundation beats shape: {beats.shape}")
    beats = foundation_zscore_per_beat(beats)

    n = int(beats.shape[0])
    if n >= max_beats:
        if training:
            idx = np.sort(rng.choice(n, size=max_beats, replace=False))
        else:
            idx = np.linspace(0, n - 1, num=max_beats).astype(np.int64)
        return beats[idx], time_sec[idx], np.ones(max_beats, dtype=bool)

    out = np.zeros((max_beats, beats.shape[1], beats.shape[2]), dtype=np.float32)
    out[:n, : beats.shape[1], : beats.shape[2]] = beats
    time_pad = np.zeros((max_beats,), dtype=np.float32)
    time_pad[: min(n, len(time_sec))] = time_sec[: min(n, len(time_sec))]
    beat_mask = np.zeros(max_beats, dtype=bool)
    beat_mask[:n] = True
    return out, time_pad, beat_mask


def load_foundation_beats_from_path(
    signal_path: str,
    fs: int,
    beat_len: int,
    max_beats: int,
    rng: np.random.RandomState,
    training: bool,
    input_cols: Sequence[int] = (0, 1),
    ecg_col: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_cols = max(list(input_cols) + [int(ecg_col)]) + 1
    sig = load_raw_signal_for_foundation(signal_path, min_cols=min_cols)
    arrays = extract_foundation_beat_arrays(sig, fs=fs, input_cols=input_cols, ecg_col=ecg_col, beat_len=beat_len)
    return sample_or_pad_foundation_beats(
        beats=arrays["beats"],
        time_sec=arrays["time_sec"],
        max_beats=max_beats,
        rng=rng,
        training=training,
    )


def safe_cache_token(value, max_len: int = 80) -> str:
    text = str(value) if value is not None and not pd.isna(value) else "unknown"
    text = text.strip()
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-", "."}:
            out.append(ch)
        else:
            out.append("_")
    token = "".join(out).strip("._-")
    return (token[:max_len] if token else "unknown")


def make_bp_npz_cache_path(cache_root: str, row: pd.Series) -> str:
    signal_path = str(row.get("signal_path", "")).replace("\\", "/")
    source = safe_cache_token(row.get("source_name", "unknown"))
    subject = safe_cache_token(row.get("subject_id", infer_subject_from_path(signal_path)))
    name = Path(signal_path).name
    if name.endswith(".gz"):
        name = name[:-3]
    stem = Path(name).stem or "signal"
    digest = hashlib.sha1(signal_path.encode("utf-8", errors="replace")).hexdigest()[:12]
    return str(Path(cache_root) / source / subject / f"{safe_cache_token(stem, max_len=96)}_{digest}.npz")


def attach_npz_cache_paths(df: pd.DataFrame, cache_root: str) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["npz_path"] = [make_bp_npz_cache_path(cache_root, row) for _, row in out.iterrows()]
    return out


def save_foundation_npz_cache(npz_path: str, arrays: Dict[str, np.ndarray], row: pd.Series, compressed: bool = False) -> None:
    Path(npz_path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = f"{npz_path}.tmp.{os.getpid()}.npz"
    payload = {
        "beats": np.asarray(arrays["beats"], dtype=np.float32),
        "time_sec": np.asarray(arrays["time_sec"], dtype=np.float32),
        "signal_path": np.asarray(str(row.get("signal_path", ""))),
        "subject_id": np.asarray(str(row.get("subject_id", ""))),
        "source_name": np.asarray(str(row.get("source_name", ""))),
    }
    if compressed:
        np.savez_compressed(tmp_path, **payload)
    else:
        np.savez(tmp_path, **payload)
    os.replace(tmp_path, npz_path)


def build_npz_cache_for_frame(
    df: pd.DataFrame,
    *,
    fs: int,
    beat_len: int,
    cache_root: str,
    tag: str,
    force: bool = False,
    compressed: bool = False,
    log_every: int = 200,
) -> pd.DataFrame:
    if df.empty:
        return df
    if "npz_path" not in df.columns:
        df = attach_npz_cache_paths(df, cache_root)
    total = len(df)
    built = 0
    skipped = 0
    failed = 0
    bad_examples = []
    for i, row in enumerate(df.itertuples(index=False), start=1):
        npz_path = str(getattr(row, "npz_path"))
        signal_path = str(getattr(row, "signal_path"))
        if (not force) and os.path.exists(npz_path):
            skipped += 1
        else:
            try:
                sig = load_raw_signal_for_foundation(signal_path, min_cols=2)
                arrays = extract_foundation_beat_arrays(sig, fs=fs, input_cols=ECG_PPG_CHANNELS, ecg_col=0, beat_len=beat_len)
                save_foundation_npz_cache(
                    npz_path,
                    arrays,
                    pd.Series(row._asdict()),
                    compressed=compressed,
                )
                built += 1
            except Exception as exc:
                failed += 1
                if len(bad_examples) < 20:
                    bad_examples.append((signal_path, f"{type(exc).__name__}: {exc}"))
        if is_main_process() and (i == 1 or i % max(1, log_every) == 0 or i == total):
            print(
                f"[npz:{tag}] checked {i}/{total} built={built} cached={skipped} failed={failed}",
                flush=True,
            )
    main_print(f"[npz:{tag}] done total={total} built={built} cached={skipped} failed={failed} root={cache_root}")
    if bad_examples:
        main_print(f"[npz:{tag}] failed examples (up to 20):")
        for path, reason in bad_examples:
            main_print(f"  - {path} | {reason}")
    return df


def filter_npz_cache_rows(df: pd.DataFrame, tag: str = "train") -> pd.DataFrame:
    if df.empty or "npz_path" not in df.columns:
        return df
    keep = df["npz_path"].astype(str).map(os.path.exists).to_numpy(dtype=bool)
    out = df.loc[keep].reset_index(drop=True)
    dropped = int((~keep).sum())
    main_print(f"[npz:{tag}] keep={len(out)} drop_missing_npz={dropped} total={len(df)}")
    if dropped > 0:
        missing = df.loc[~keep, ["signal_path", "npz_path"]].head(20)
        main_print(f"[npz:{tag}] missing examples (up to 20):")
        for _, row in missing.iterrows():
            main_print(f"  - {row['signal_path']} -> {row['npz_path']}")
    return out


def prepare_npz_cache_frames(args: argparse.Namespace, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = attach_npz_cache_paths(train_df, args.bp_npz_cache_root)
    test_df = attach_npz_cache_paths(test_df, args.bp_npz_cache_root)
    if getattr(args, "require_existing_npz_cache", False):
        main_print(f"\n[npz] cache_root={args.bp_npz_cache_root}")
        main_print("[npz] require_existing_npz_cache=True; skip cache building and only filter existing NPZ files")
    elif is_main_process():
        main_print(f"\n[npz] cache_root={args.bp_npz_cache_root}")
        train_df = build_npz_cache_for_frame(
            train_df,
            fs=args.fs,
            beat_len=args.beat_len,
            cache_root=args.bp_npz_cache_root,
            tag="train",
            force=args.rebuild_npz_cache,
            compressed=args.compress_npz_cache,
            log_every=args.npz_log_every,
        )
        test_df = build_npz_cache_for_frame(
            test_df,
            fs=args.fs,
            beat_len=args.beat_len,
            cache_root=args.bp_npz_cache_root,
            tag="test",
            force=args.rebuild_npz_cache,
            compressed=args.compress_npz_cache,
            log_every=args.npz_log_every,
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if args.skip_bad_samples:
        train_df = filter_npz_cache_rows(train_df, tag="train")
        test_df = filter_npz_cache_rows(test_df, tag="test")
        if train_df.empty:
            raise ValueError("Training set is empty after NPZ cache filtering.")
        if test_df.empty:
            raise ValueError("Test set is empty after NPZ cache filtering.")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_foundation_beats_from_npz(
    npz_path: str,
    max_beats: int,
    rng: np.random.RandomState,
    training: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as item:
        beats = item["beats"].astype(np.float32)
        time_sec = item["time_sec"].astype(np.float32)
    return sample_or_pad_foundation_beats(
        beats=beats,
        time_sec=time_sec,
        max_beats=max_beats,
        rng=rng,
        training=training,
    )


# ----------------------------- Beat extraction -----------------------------


def fix_unsigned24_if_needed(x: np.ndarray, tol=2e5) -> np.ndarray:
    return autofix_24bit_unsigned(x, tol=tol)


def interp_bad(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    idx = np.arange(len(x))
    ok = np.isfinite(x)
    if ok.sum() >= 2 and (~ok).any():
        x[~ok] = np.interp(idx[~ok], idx[ok], x[ok])
    return x


def butter_bandpass(x: np.ndarray, fs: int, low: float, high: float, order=3) -> np.ndarray:
    b, a = signal.butter(order, [low, high], btype="bandpass", fs=fs)
    return signal.filtfilt(b, a, x).astype(np.float32)


def preprocess_ecg_cycle(ecg_raw: np.ndarray, fs: int) -> np.ndarray:
    x = fix_unsigned24_if_needed(ecg_raw)
    x = interp_bad(x)
    return butter_bandpass(x, fs, 5.0, 40.0, order=3)


def preprocess_ppg_cycle(ppg_raw: np.ndarray, fs: int) -> np.ndarray:
    x = fix_unsigned24_if_needed(ppg_raw)
    x = interp_bad(x)
    return butter_bandpass(x, fs, 0.3, 8.0, order=3)


def detect_r_peaks(ecg_f: np.ndarray, fs: int, hr_min=35, hr_max=220) -> np.ndarray:
    ecg_f = np.asarray(ecg_f, dtype=np.float32)
    if len(ecg_f) < fs:
        return np.array([], dtype=int)
    der = np.diff(ecg_f, prepend=ecg_f[0])
    sq = der * der
    win = max(1, int(0.12 * fs))
    mwa = np.convolve(sq, np.ones(win, dtype=np.float32) / win, mode="same")
    med = np.median(mwa)
    mad = np.median(np.abs(mwa - med)) + 1e-8
    thr = med + 6.0 * 1.4826 * mad
    min_dist = max(1, int(fs * 60.0 / hr_max))
    peaks, _ = signal.find_peaks(mwa, height=thr, distance=min_dist)
    if len(peaks) == 0:
        return np.array([], dtype=int)
    refine = []
    w = int(0.05 * fs)
    for p in peaks:
        left = max(0, p - w)
        right = min(len(ecg_f), p + w + 1)
        refine.append(left + int(np.argmax(ecg_f[left:right])))
    refine = np.array(sorted(set(refine)), dtype=int)
    if len(refine) < 3:
        return refine
    rr = np.diff(refine) / fs
    ok = np.ones_like(refine, dtype=bool)
    bad = (rr < 60.0 / hr_max) | (rr > 60.0 / hr_min)
    for i in np.where(bad)[0]:
        ok[i] = False
        ok[i + 1] = False
    return refine[ok]


def make_rr_boundaries(r_peaks: np.ndarray) -> np.ndarray:
    r_peaks = np.asarray(r_peaks, dtype=int)
    if len(r_peaks) < 3:
        return np.array([], dtype=int)
    return ((r_peaks[:-1] + r_peaks[1:]) // 2).astype(int)


def segment_by_boundaries(x: np.ndarray, boundaries: np.ndarray):
    boundaries = np.asarray(boundaries, dtype=int)
    if len(boundaries) < 2:
        return []
    return [x[boundaries[i - 1] : boundaries[i]] for i in range(1, len(boundaries))]


def is_saturated(seg: np.ndarray, q=0.001) -> bool:
    if len(seg) < 10:
        return True
    lo, hi = np.quantile(seg, [q, 1 - q])
    frac_lo = np.mean(seg <= lo)
    frac_hi = np.mean(seg >= hi)
    return (frac_lo > 0.05) or (frac_hi > 0.05)


def filter_cycles(segs, fs: int, rr_min=0.35, rr_max=2.0, drop_saturation=True):
    keep = []
    idx_keep = []
    for i, seg in enumerate(segs):
        if not np.isfinite(seg).all():
            continue
        dur = len(seg) / fs
        if dur < rr_min or dur > rr_max:
            continue
        if drop_saturation and is_saturated(seg):
            continue
        keep.append(seg)
        idx_keep.append(i)
    return keep, np.array(idx_keep, dtype=int)


def resample_cycles_to_fixed(segs, cycle_len=256) -> np.ndarray:
    out = np.zeros((len(segs), cycle_len), dtype=np.float32)
    for i, seg in enumerate(segs):
        seg = np.asarray(seg, dtype=np.float32)
        t_old = np.linspace(0, 1, len(seg), endpoint=False)
        t_new = np.linspace(0, 1, cycle_len, endpoint=False)
        out[i] = np.interp(t_new, t_old, seg)
    return out


def template_correlation_filter(x_ecg: np.ndarray, corr_thr=0.8) -> np.ndarray:
    if x_ecg.shape[0] < 5:
        return np.ones(x_ecg.shape[0], dtype=bool)
    x0 = x_ecg - x_ecg.mean(axis=1, keepdims=True)
    x0 = x0 / (np.linalg.norm(x0, axis=1, keepdims=True) + 1e-8)
    template = np.median(x0, axis=0, keepdims=True)
    template = template / (np.linalg.norm(template) + 1e-8)
    corr = (x0 @ template.T).squeeze()
    return corr >= corr_thr


def extract_cycles_4ch(
    sig_tc: np.ndarray,
    fs: int,
    cycle_len=256,
    trim_sec=2.0,
    corr_thr=0.8,
    hr_min=35,
    hr_max=220,
    rr_min=0.35,
    rr_max=2.0,
) -> np.ndarray:
    ecg = preprocess_ecg_cycle(sig_tc[:, 0], fs)
    ppg_ir = preprocess_ppg_cycle(sig_tc[:, 1], fs)
    ppg_rd = preprocess_ppg_cycle(sig_tc[:, 2], fs)
    ppg_ot = preprocess_ppg_cycle(sig_tc[:, 3], fs)

    n0 = int(trim_sec * fs)
    n1 = len(ecg) - n0
    if n1 <= n0 + fs:
        return np.zeros((0, 4, cycle_len), dtype=np.float32)

    ecg_c = ecg[n0:n1]
    ppg_ir_c = ppg_ir[n0:n1]
    ppg_rd_c = ppg_rd[n0:n1]
    ppg_ot_c = ppg_ot[n0:n1]
    r = detect_r_peaks(ecg_c, fs, hr_min=hr_min, hr_max=hr_max)
    if len(r) < 5:
        return np.zeros((0, 4, cycle_len), dtype=np.float32)

    boundaries = make_rr_boundaries(r)
    ecg_segs = segment_by_boundaries(ecg_c, boundaries)
    ppg_ir_all = segment_by_boundaries(ppg_ir_c, boundaries)
    ppg_rd_all = segment_by_boundaries(ppg_rd_c, boundaries)
    ppg_ot_all = segment_by_boundaries(ppg_ot_c, boundaries)

    ecg_segs, idx_keep = filter_cycles(ecg_segs, fs, rr_min=rr_min, rr_max=rr_max)
    if len(ecg_segs) == 0:
        return np.zeros((0, 4, cycle_len), dtype=np.float32)
    ppg_ir_segs = [ppg_ir_all[i] for i in idx_keep]
    ppg_rd_segs = [ppg_rd_all[i] for i in idx_keep]
    ppg_ot_segs = [ppg_ot_all[i] for i in idx_keep]

    x_ecg = resample_cycles_to_fixed(ecg_segs, cycle_len=cycle_len)
    x_ir = resample_cycles_to_fixed(ppg_ir_segs, cycle_len=cycle_len)
    x_rd = resample_cycles_to_fixed(ppg_rd_segs, cycle_len=cycle_len)
    x_ot = resample_cycles_to_fixed(ppg_ot_segs, cycle_len=cycle_len)
    mask = template_correlation_filter(x_ecg, corr_thr=corr_thr)
    x_ecg = x_ecg[mask]
    x_ir = x_ir[mask]
    x_rd = x_rd[mask]
    x_ot = x_ot[mask]
    if len(x_ecg) == 0:
        return np.zeros((0, 4, cycle_len), dtype=np.float32)
    return np.stack([x_ecg, x_ir, x_rd, x_ot], axis=1).astype(np.float32)


def fallback_uniform_cycles(sig_tc: np.ndarray, cycle_len: int, max_cycles: int) -> np.ndarray:
    t = sig_tc.shape[0]
    if t <= cycle_len:
        pad = cycle_len - t
        sig = np.pad(sig_tc, ((0, pad), (0, 0)), mode="edge")
        sig = np.transpose(sig, (1, 0))
        return np.stack([sig] * max_cycles, axis=0).astype(np.float32)
    starts = np.linspace(0, t - cycle_len, num=max_cycles).astype(int)
    cycles = [sig_tc[s : s + cycle_len, :].T for s in starts]
    return np.stack(cycles, axis=0).astype(np.float32)


def sample_or_pad_cycles(cycles_ncl: np.ndarray, max_cycles: int, rng: np.random.RandomState, training: bool):
    n = cycles_ncl.shape[0]
    if n == 0:
        raise ValueError("cycles_ncl must not be empty")
    if n >= max_cycles:
        if training:
            idx = np.sort(rng.choice(n, size=max_cycles, replace=False))
        else:
            idx = np.linspace(0, n - 1, num=max_cycles).astype(int)
        return cycles_ncl[idx], np.ones(max_cycles, dtype=bool)
    out = np.zeros((max_cycles, cycles_ncl.shape[1], cycles_ncl.shape[2]), dtype=np.float32)
    out[:n] = cycles_ncl
    mask = np.zeros(max_cycles, dtype=bool)
    mask[:n] = True
    return out, mask


def select_channels(cycles: np.ndarray, keep_indices: List[int]) -> np.ndarray:
    if len(keep_indices) == cycles.shape[1]:
        return np.ascontiguousarray(cycles)
    return np.ascontiguousarray(cycles[:, keep_indices, :])


# ----------------------------- BP metadata and dataset -----------------------------


def map_sex_value(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"男", "male", "m", "1", "1.0"}:
        return 1.0
    if s in {"女", "female", "f", "0", "0.0"}:
        return 0.0
    try:
        return float(x)
    except Exception:
        return np.nan


def infer_subject_from_path(path: str) -> str:
    if pd.isna(path) or not path:
        return ""
    parts = str(path).replace("\\", "/").rstrip("/").split("/")
    if len(parts) >= 2:
        return str(parts[-2])
    return str(parts[-1]).split(".")[0] if parts else ""


def load_bp_csv_with_subject(
    csv_path: str,
    signal_root: str,
    data_root: Optional[str] = None,
    source_name: Optional[str] = None,
    col_file: str = "ECG+PPG文件索引",
    col_age: str = "年龄",
    col_sex: str = "性别",
    col_h: str = "身高（cm）",
    col_w: str = "体重（kg）",
    col_sbp: str = "y_sbp",
    col_dbp: str = "y_dbp",
    col_subject: str = "账号",
) -> pd.DataFrame:
    df = read_csv_robust(csv_path).copy()
    file_col = pick_column_relaxed(df, [col_file, "ECG+PPG文件索引", "signal_path", "file_path", "path"])
    age_col = pick_column_relaxed(df, [col_age, "年龄", "age"])
    sex_col = pick_column_relaxed(df, [col_sex, "性别", "sex", "gender"])
    h_col = pick_column_relaxed(df, [col_h, "身高（cm）", "身高(cm)", "height_cm", "height"])
    w_col = pick_column_relaxed(df, [col_w, "体重（kg）", "体重(kg)", "weight_kg", "weight"])
    sbp_col = pick_column_relaxed(df, [col_sbp] + SBP_CANDIDATES)
    dbp_col = pick_column_relaxed(df, [col_dbp] + DBP_CANDIDATES)
    subject_col = pick_column_relaxed(df, [col_subject, "账号", "subject_id", "subject", "user_id", "id"])

    missing = [
        name
        for name, col in [
            ("file", file_col),
            ("age", age_col),
            ("sex", sex_col),
            ("height", h_col),
            ("weight", w_col),
            ("sbp", sbp_col),
            ("dbp", dbp_col),
        ]
        if col is None
    ]
    if missing:
        raise KeyError(f"{csv_path} missing required columns: {missing}. Available={list(df.columns)}")

    def resolve_row_path(row: pd.Series) -> Optional[str]:
        candidates = []
        if "signal_path" in row.index and pd.notna(row["signal_path"]):
            candidates.append(row["signal_path"])
        if file_col in row.index and pd.notna(row[file_col]):
            candidates.append(row[file_col])
        for raw in candidates:
            hit = resolve_signal_path(
                signal_root,
                raw,
                data_root=data_root,
                source_name=source_name,
            )
            if hit:
                return hit
        return None

    df["signal_path"] = df.apply(resolve_row_path, axis=1)
    if subject_col is not None:
        df["subject_id"] = df[subject_col].astype(str)
    else:
        df["subject_id"] = df["signal_path"].apply(infer_subject_from_path)

    df["age"] = pd.to_numeric(df[age_col], errors="coerce")
    df["sex"] = df[sex_col].apply(map_sex_value).astype(np.float32)
    df["height_cm"] = pd.to_numeric(df[h_col], errors="coerce")
    df["weight_kg"] = pd.to_numeric(df[w_col], errors="coerce")
    df["y_sbp"] = pd.to_numeric(df[sbp_col], errors="coerce")
    df["y_dbp"] = pd.to_numeric(df[dbp_col], errors="coerce")
    keep = ["signal_path", "subject_id", "age", "sex", "height_cm", "weight_kg", "y_sbp", "y_dbp"]
    out = df.dropna(subset=keep).copy().reset_index(drop=True)
    out = out[out["y_sbp"].between(50, 260) & out["y_dbp"].between(30, 180)].reset_index(drop=True)
    main_print(f"[load:{Path(csv_path).name}] rows={len(out)} labels=({sbp_col}, {dbp_col})")
    return out


def normalize_text_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().replace({"nan": "", "None": "", "none": ""})


def build_medication_flag_frame(
    raw_df: pd.DataFrame,
    signal_root: str,
    data_root: Optional[str] = None,
    source_name: Optional[str] = None,
    col_file: str = "ECG+PPG文件索引",
    med_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    med_cols = med_cols or DEFAULT_MEDICATION_COLUMNS
    out = raw_df.copy()
    file_col = pick_column_relaxed(out, [col_file, "ECG+PPG文件索引", "signal_path", "file_path", "path"])
    if file_col is None:
        raise KeyError(f"Unable to resolve signal path column. Available columns: {list(out.columns)}")
    def resolve_row_path(row: pd.Series) -> Optional[str]:
        candidates = []
        if "signal_path" in row.index and pd.notna(row["signal_path"]):
            candidates.append(row["signal_path"])
        if file_col in row.index and pd.notna(row[file_col]):
            candidates.append(row[file_col])
        for raw in candidates:
            hit = resolve_signal_path(
                signal_root,
                raw,
                data_root=data_root,
                source_name=source_name,
            )
            if hit:
                return hit
        return None

    out["signal_path"] = out.apply(resolve_row_path, axis=1)

    resolved_med_cols = []
    for col in med_cols:
        hit = pick_column_relaxed(out, [col])
        if hit and hit not in resolved_med_cols:
            resolved_med_cols.append(hit)

    has_record = pd.Series(False, index=out.index)
    med_count = pd.Series(0, index=out.index, dtype=np.int64)
    for col in resolved_med_cols:
        vals = normalize_text_series(out[col])
        nonempty = vals.ne("")
        has_record = has_record | nonempty
        med_count = med_count + nonempty.astype(np.int64)

    return pd.DataFrame(
        {
            "signal_path": out["signal_path"].astype(str),
            "has_med_record": has_record.astype(np.float32),
            "med_record_count": med_count.astype(np.int64),
        }
    ).drop_duplicates(subset=["signal_path"], keep="last")


def load_single_source_with_med(src: dict, args: argparse.Namespace) -> pd.DataFrame:
    csv_path = src["csv_path"]
    signal_root = src["signal_root"]
    if not os.path.exists(csv_path):
        main_print(f"[load] CSV not found: {csv_path}")
        return pd.DataFrame()
    if not os.path.exists(signal_root):
        main_print(f"[load] signal root not found: {signal_root}")
        return pd.DataFrame()

    raw_df = read_csv_robust(csv_path).copy()
    med_df = build_medication_flag_frame(
        raw_df,
        signal_root=signal_root,
        data_root=args.bp_data_root,
        source_name=src.get("name"),
        col_file=args.col_file,
        med_cols=args.med_cols,
    )
    loaded = load_bp_csv_with_subject(
        csv_path=csv_path,
        signal_root=signal_root,
        data_root=args.bp_data_root,
        source_name=src.get("name"),
        col_file=args.col_file,
        col_subject=args.col_subject,
        col_sbp=args.col_sbp,
        col_dbp=args.col_dbp,
    )
    if loaded.empty:
        return loaded
    loaded = loaded.merge(med_df, on="signal_path", how="left")
    loaded["has_med_record"] = pd.to_numeric(loaded["has_med_record"], errors="coerce").fillna(0.0).astype(np.float32)
    loaded["med_record_count"] = pd.to_numeric(loaded["med_record_count"], errors="coerce").fillna(0).astype(np.int64)
    loaded["source_name"] = src.get("name", "unknown")
    return loaded.reset_index(drop=True)


def filter_bad_signal_rows(df: pd.DataFrame, fs: int = 250, beat_len: int = 128, tag: str = "train") -> pd.DataFrame:
    if df.empty:
        return df
    keep_mask = []
    bad_examples = []
    total = len(df)
    for i, row in enumerate(df.itertuples(index=False), start=1):
        path = str(getattr(row, "signal_path", "")).strip()
        ok = True
        reason = ""
        if not path or not os.path.exists(path):
            ok = False
            reason = "FileNotFoundError: signal_path missing"
        else:
            try:
                sig = load_raw_signal_for_foundation(path, min_cols=2)
                arrays = extract_foundation_beat_arrays(sig, fs=fs, input_cols=ECG_PPG_CHANNELS, ecg_col=0, beat_len=beat_len)
                if arrays["beats"].shape[0] < 1:
                    raise ValueError("no foundation beats extracted")
            except Exception as exc:
                ok = False
                reason = f"{type(exc).__name__}: {exc}"
        keep_mask.append(ok)
        if not ok and len(bad_examples) < 20:
            bad_examples.append((path, reason))
        if is_main_process() and (i % 200 == 0 or i == total):
            print(f"[filter:{tag}] checked {i}/{total}", flush=True)
    keep_mask = np.asarray(keep_mask, dtype=bool)
    out = df.loc[keep_mask].reset_index(drop=True)
    dropped = int((~keep_mask).sum())
    main_print(f"[filter:{tag}] keep={len(out)} drop={dropped} total={total}")
    if bad_examples:
        main_print(f"[filter:{tag}] dropped examples (up to 20):")
        for path, reason in bad_examples:
            main_print(f"  - {path} | {reason}")
    return out


def load_frames(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_sources = make_data_sources(args.bp_data_root)
    train_sources = select_sources_by_names(data_sources, TRAIN_SOURCE_NAMES)
    test_sources = select_sources_by_names(data_sources, TEST_SOURCE_NAMES)

    train_parts = [load_single_source_with_med(src, args) for src in train_sources]
    test_parts = [load_single_source_with_med(src, args) for src in test_sources]
    train_df = pd.concat([p for p in train_parts if not p.empty], ignore_index=True) if any(not p.empty for p in train_parts) else pd.DataFrame()
    test_df = pd.concat([p for p in test_parts if not p.empty], ignore_index=True) if any(not p.empty for p in test_parts) else pd.DataFrame()

    if train_df.empty:
        raise ValueError("Training set is empty. Check BP_DATA_ROOT and train CSV/signal paths.")
    if test_df.empty:
        raise ValueError("Test set is empty. Check BP_DATA_ROOT and predict CSV/signal paths.")

    if args.skip_bad_samples and not args.use_npz_cache:
        main_print("\n[filter] Pre-checking signal files...")
        train_df = filter_bad_signal_rows(train_df, fs=args.fs, beat_len=args.beat_len, tag="train")
        test_df = filter_bad_signal_rows(test_df, fs=args.fs, beat_len=args.beat_len, tag="test")
        if train_df.empty:
            raise ValueError("Training set is empty after bad-sample filtering.")
        if test_df.empty:
            raise ValueError("Test set is empty after bad-sample filtering.")

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def compute_cov_stats(df: pd.DataFrame, cov_cols_z: List[str]) -> Dict[str, Tuple[float, float]]:
    stats = {}
    for c in cov_cols_z:
        x = df[c].to_numpy(dtype=np.float32)
        x = x[np.isfinite(x)]
        if len(x) < 2:
            stats[c] = (0.0, 1.0)
        else:
            stats[c] = (float(np.mean(x)), float(np.std(x)) + 1e-6)
    return stats


def normalize_covariates(
    df: pd.DataFrame,
    cov_cols: List[str],
    cov_stats: Dict[str, Tuple[float, float]],
    cov_cols_z: List[str],
) -> np.ndarray:
    rows = []
    for _, row in df.iterrows():
        vec = []
        for c in cov_cols:
            v = row[c]
            if pd.isna(v):
                v = 0.0
            if c in cov_cols_z and c in cov_stats:
                mu, sigma = cov_stats[c]
                v = (float(v) - mu) / sigma
            vec.append(float(v))
        rows.append(vec)
    return np.array(rows, dtype=np.float32)


def log_bp(y_sbp: np.ndarray, y_dbp: np.ndarray) -> np.ndarray:
    y = np.stack([y_sbp, y_dbp], axis=1).astype(np.float32)
    return np.log(np.clip(y, 1.0, None))


class DirectBPDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cov_cols: List[str],
        cov_stats: Dict[str, Tuple[float, float]],
        cycle_len: int = 128,
        max_cycles: int = 25,
        fs: int = 250,
        apply_preprocessing: bool = True,
        seed: int = 42,
        training: bool = False,
        keep_channel_indices: Optional[List[int]] = None,
        use_npz_cache: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.cov_cols = list(cov_cols)
        self.cov_stats = cov_stats
        self.cov_cols_z = ["age", "height_cm", "weight_kg"]
        self.cycle_len = cycle_len
        self.max_cycles = max_cycles
        self.fs = fs
        self.apply_preprocessing = apply_preprocessing
        self.training = training
        self.rng = np.random.RandomState(seed)
        self.keep_channel_indices = keep_channel_indices or ECG_PPG_CHANNELS
        self.use_npz_cache = bool(use_npz_cache)
        self.cov_arr = normalize_covariates(self.df, cov_cols=self.cov_cols, cov_stats=self.cov_stats, cov_cols_z=self.cov_cols_z)
        self.y_log = log_bp(self.df["y_sbp"].to_numpy(dtype=np.float32), self.df["y_dbp"].to_numpy(dtype=np.float32))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        if self.use_npz_cache:
            cycles, time_sec, cycle_mask = load_foundation_beats_from_npz(
                npz_path=row["npz_path"],
                max_beats=self.max_cycles,
                rng=self.rng,
                training=self.training,
            )
        else:
            cycles, time_sec, cycle_mask = load_foundation_beats_from_path(
                signal_path=row["signal_path"],
                fs=self.fs,
                beat_len=self.cycle_len,
                max_beats=self.max_cycles,
                rng=self.rng,
                training=self.training,
                input_cols=self.keep_channel_indices,
                ecg_col=0,
            )
        return {
            "ppg": torch.from_numpy(cycles).float(),
            "time_sec": torch.from_numpy(time_sec).float(),
            "mask": torch.from_numpy(cycle_mask).bool(),
            "cov": torch.from_numpy(self.cov_arr[idx]).float(),
            "y": torch.from_numpy(self.y_log[idx]).float(),
        }


# ----------------------------- Foundation encoder -----------------------------


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PhaseAwareBeatEncoder(nn.Module):
    def __init__(self, in_channels: int, d_model: int, phase_tokens: int = 8):
        super().__init__()
        self.phase_tokens = int(phase_tokens)
        if self.phase_tokens < 1:
            raise ValueError("phase_tokens must be >= 1")
        self.net = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=9, stride=2),
            ConvBlock(32, 64, kernel_size=7, stride=2),
            ConvBlock(64, 128, kernel_size=5, stride=2),
        )
        self.phase_pool = nn.AdaptiveAvgPool1d(self.phase_tokens)
        self.proj = nn.Linear(128, d_model)
        self.phase_embed = nn.Parameter(torch.zeros(1, self.phase_tokens, d_model))
        nn.init.normal_(self.phase_embed, std=0.02)

    def forward_phase(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        phase_feat = self.phase_pool(feat).transpose(1, 2)
        return self.proj(phase_feat) + self.phase_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_phase(x).mean(dim=1)


class OrthogonalBeatPhaseBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.phase_norm = nn.LayerNorm(d_model)
        self.phase_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.beat_norm = nn.LayerNorm(d_model)
        self.beat_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, beat_mask: torch.Tensor) -> torch.Tensor:
        b, k, p, d = x.shape
        phase_x = self.phase_norm(x).reshape(b * k, p, d)
        phase_y, _ = self.phase_attn(phase_x, phase_x, phase_x, need_weights=False)
        x = x + phase_y.reshape(b, k, p, d)

        beat_x = self.beat_norm(x).permute(0, 2, 1, 3).reshape(b * p, k, d)
        beat_pad = (~beat_mask).unsqueeze(1).expand(b, p, k).reshape(b * p, k)
        beat_y, _ = self.beat_attn(beat_x, beat_x, beat_x, key_padding_mask=beat_pad, need_weights=False)
        beat_y = beat_y.reshape(b, p, k, d).permute(0, 2, 1, 3)
        x = x + beat_y
        return x + self.ffn(self.ffn_norm(x))


class ReliabilityGatedPooling(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        hidden = max(d_model // 2, 32)
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, beat_z: torch.Tensor, beat_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(beat_z).squeeze(-1)
        logits = logits.masked_fill(~beat_mask, -1e4)
        weights = torch.softmax(logits, dim=1) * beat_mask.to(dtype=beat_z.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = (beat_z * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights


class CumulativeTimeEncoding(nn.Module):
    def __init__(self, d_model: int, num_bands: int = 32, max_period: float = 10000.0):
        super().__init__()
        self.num_bands = num_bands
        freq = torch.exp(-math.log(max_period) * torch.arange(num_bands).float() / max(num_bands - 1, 1))
        self.register_buffer("freq", freq)
        self.proj = nn.Linear(num_bands * 2 + 1, d_model)

    def forward(self, time_sec: torch.Tensor) -> torch.Tensor:
        angles = time_sec.unsqueeze(-1) * self.freq.view(1, 1, -1)
        feats = torch.cat([time_sec.unsqueeze(-1), torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.proj(feats)


class ECGPPGMultiTaskTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        beat_len: int = 128,
        morph_dim: int = 6,
        hrv_dim: int = 3,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 9,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        phase_tokens: int = 8,
        num_input_modes: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.beat_len = beat_len
        self.morph_dim = morph_dim
        self.hrv_dim = hrv_dim
        self.d_model = d_model
        self.phase_tokens = int(phase_tokens)
        if in_channels < 2:
            raise ValueError("modality-flexible model expects at least 2 channels: ECG + PPG")
        self.ecg_encoder = PhaseAwareBeatEncoder(in_channels=1, d_model=d_model, phase_tokens=self.phase_tokens)
        self.ppg_encoder = PhaseAwareBeatEncoder(in_channels=1, d_model=d_model, phase_tokens=self.phase_tokens)
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.time_encoding = CumulativeTimeEncoding(d_model=d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        self.input_mode_embed = nn.Embedding(num_input_modes, d_model)
        nn.init.zeros_(self.input_mode_embed.weight)
        self.blocks = nn.ModuleList(
            [
                OrthogonalBeatPhaseBlock(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.reliability_pool = ReliabilityGatedPooling(d_model=d_model, dropout=dropout)
        recon_dim = in_channels * beat_len
        self.recon_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, recon_dim))
        self.pat_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        self.morph_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, morph_dim))
        self.hrv_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, hrv_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def encode_context(
        self,
        beats: torch.Tensor,
        time_sec: torch.Tensor,
        beat_mask: torch.Tensor,
        pretrain_mask: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
        input_mode: Optional[torch.Tensor] = None,
        modality_status: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b, k, c, l = beats.shape
        if c != self.in_channels or l != self.beat_len:
            raise ValueError(f"Expected beats (B,K,{self.in_channels},{self.beat_len}), got {tuple(beats.shape)}")
        ecg_x = beats[:, :, 0:1, :].reshape(b * k, 1, l)
        ppg_x = beats[:, :, 1:2, :].reshape(b * k, 1, l)
        ecg_z = self.ecg_encoder.forward_phase(ecg_x).reshape(b, k, self.phase_tokens, self.d_model)
        ppg_z = self.ppg_encoder.forward_phase(ppg_x).reshape(b, k, self.phase_tokens, self.d_model)
        z = torch.stack([ecg_z, ppg_z], dim=2)
        if modality_mask is None:
            modality_mask = torch.ones(b, 2, dtype=torch.bool, device=beats.device)
        else:
            modality_mask = modality_mask.to(device=beats.device, dtype=torch.bool)
        gate_logits = self.fusion_gate(z.mean(dim=3)).squeeze(-1)
        gate_logits = gate_logits.masked_fill(~modality_mask[:, None, :].expand(b, k, 2), -1e4)
        alpha = torch.softmax(gate_logits, dim=2)
        alpha = alpha.masked_fill(~modality_mask[:, None, :].expand(b, k, 2), 0.0)
        alpha = alpha / alpha.sum(dim=2, keepdim=True).clamp_min(1e-6)
        phase_z = (alpha[:, :, :, None, None] * z).sum(dim=2)
        if input_mode is None:
            # Backward-compatible alias: modality_status now means input_mode
            # (0=both, 1=ppg_only), applied after mask-aware fusion.
            input_mode = modality_status
        if input_mode is not None:
            input_mode = input_mode.to(device=beats.device, dtype=torch.long).clamp(
                0, self.input_mode_embed.num_embeddings - 1
            )
            phase_z = phase_z + self.input_mode_embed(input_mode).view(b, 1, 1, self.d_model)
        phase_z = phase_z + self.time_encoding(time_sec).unsqueeze(2)
        if pretrain_mask is not None:
            mask_token = self.mask_token.expand(b, k, self.phase_tokens, -1)
            phase_z = torch.where(pretrain_mask.unsqueeze(-1).unsqueeze(-1), mask_token, phase_z)
        for block in self.blocks:
            phase_z = block(phase_z, beat_mask)
        phase_z = self.norm(phase_z)
        beat_z = phase_z.mean(dim=2)
        cls_z, reliability = self.reliability_pool(beat_z, beat_mask)
        return {"cls": cls_z, "beat_z": beat_z, "phase_z": phase_z, "reliability": reliability}

    def forward(
        self,
        beats: torch.Tensor,
        time_sec: torch.Tensor,
        beat_mask: torch.Tensor,
        pretrain_mask: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
        input_mode: Optional[torch.Tensor] = None,
        modality_status: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b, k, _, _ = beats.shape
        ctx = self.encode_context(
            beats,
            time_sec,
            beat_mask,
            pretrain_mask,
            modality_mask=modality_mask,
            input_mode=input_mode,
            modality_status=modality_status,
        )
        beat_z = ctx["beat_z"]
        cls_z = ctx["cls"]
        recon = self.recon_head(beat_z).reshape(b, k, self.in_channels, self.beat_len)
        return {
            "cls": cls_z,
            "beat_z": beat_z,
            "phase_z": ctx["phase_z"],
            "reliability": ctx["reliability"],
            "recon": recon,
            "pat": self.pat_head(beat_z),
            "morph": self.morph_head(beat_z),
            "hrv": self.hrv_head(cls_z),
        }


def strip_checkpoint_prefix(key: str) -> str:
    for prefix in ("module.", "model.", "foundation."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    return key


def load_matching_foundation_state(model: nn.Module, ckpt_path: str) -> Tuple[int, int, int]:
    path = Path(ckpt_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"foundation checkpoint not found: {path}")
    raw = torch.load(path, map_location="cpu")
    state = raw.get("model_state", raw.get("state_dict", raw)) if isinstance(raw, dict) else raw
    if not isinstance(state, dict):
        raise TypeError(f"unsupported checkpoint format at {path}")
    current = model.state_dict()
    matched = {}
    skipped_shape = 0
    for key, value in state.items():
        if not hasattr(value, "shape"):
            continue
        key2 = strip_checkpoint_prefix(str(key))
        if key2 not in current:
            continue
        if tuple(value.shape) != tuple(current[key2].shape):
            skipped_shape += 1
            continue
        matched[key2] = value
    if not matched:
        raise RuntimeError(
            "No checkpoint tensors matched the foundation model. "
            "Check --beat-len/--d-model/--nhead/--num-layers against the pretraining script."
        )
    model.load_state_dict(matched, strict=False)
    missing = len([k for k in current if k not in matched])
    return len(matched), missing, skipped_shape


class OrthogonalFoundationBPModel(nn.Module):
    def __init__(
        self,
        *,
        foundation_ckpt_path: str,
        cov_dim: int,
        in_channels: int = 2,
        beat_len: int = 128,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 9,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        phase_tokens: int = 8,
        default_rr_sec: float = 0.8,
        finetune_mode: str = "head",
        head_hidden_dim: int = 256,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        if finetune_mode not in FINETUNE_MODES:
            raise ValueError(f"Unsupported finetune_mode={finetune_mode!r}; choose from {FINETUNE_MODES}")
        self.foundation = ECGPPGMultiTaskTransformer(
            in_channels=in_channels,
            beat_len=beat_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            phase_tokens=phase_tokens,
        )
        self.loaded_count, self.missing_count, self.skipped_shape_count = load_matching_foundation_state(
            self.foundation, foundation_ckpt_path
        )
        self.default_rr_sec = float(default_rr_sec)
        self.reg_head = nn.Sequential(
            nn.LayerNorm(d_model + cov_dim),
            nn.Linear(d_model + cov_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, head_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim // 2, 2),
        )
        self.configure_finetuning(finetune_mode)

    def configure_finetuning(self, mode: str) -> None:
        for param in self.foundation.parameters():
            param.requires_grad = False
        if mode == "full":
            for param in self.foundation.parameters():
                param.requires_grad = True
        elif mode in {"last1", "last3"}:
            n = int(mode.replace("last", ""))
            for block in self.foundation.blocks[-n:]:
                for param in block.parameters():
                    param.requires_grad = True
            for module in (self.foundation.norm, self.foundation.reliability_pool):
                for param in module.parameters():
                    param.requires_grad = True

    def selected_trainable_foundation_layers(self) -> List[int]:
        out = []
        for i, block in enumerate(self.foundation.blocks):
            if any(param.requires_grad for param in block.parameters()):
                out.append(i)
        return out

    def forward(
        self,
        beats: torch.Tensor,
        time_sec: torch.Tensor,
        beat_mask: torch.Tensor,
        cov: torch.Tensor,
    ) -> torch.Tensor:
        b, k, _, _ = beats.shape
        if time_sec is None:
            idx = torch.arange(k, device=beats.device, dtype=beats.dtype).unsqueeze(0).expand(b, -1)
            time_sec = idx * self.default_rr_sec
        else:
            time_sec = time_sec.to(device=beats.device, dtype=beats.dtype)
        time_sec = time_sec.masked_fill(~beat_mask, 0.0)
        ctx = self.foundation.encode_context(beats, time_sec, beat_mask, pretrain_mask=None)
        z = torch.cat([ctx["cls"], cov], dim=-1)
        return self.reg_head(z)


# ----------------------------- Train / evaluate -----------------------------


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    stats = torch.zeros(8, device=device, dtype=torch.float64)
    for batch in loader:
        ppg = batch["ppg"].to(device, non_blocking=True)
        time_sec = batch["time_sec"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        cov = batch["cov"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        pred = model(ppg, time_sec, mask, cov)
        loss = F.mse_loss(pred, y, reduction="sum")
        bs = ppg.shape[0]
        pred_mmhg = torch.exp(pred).to(torch.float64)
        y_mmhg = torch.exp(y).to(torch.float64)
        err = pred_mmhg - y_mmhg
        stats[0] += loss.detach().to(torch.float64)
        stats[1] += float(bs)
        stats[2:4] += err.abs().sum(dim=0)
        stats[4:6] += err.sum(dim=0)
        stats[6:8] += (err ** 2).sum(dim=0)
    all_reduce_sum_(stats)
    sample_count = float(stats[1].item())
    if sample_count == 0:
        return {
            "loss": float("nan"),
            "mae": np.array([np.nan, np.nan]),
            "me": np.array([np.nan, np.nan]),
            "sd": np.array([np.nan, np.nan]),
            "rmse": np.array([np.nan, np.nan]),
        }
    mae = (stats[2:4] / sample_count).cpu().numpy()
    me = (stats[4:6] / sample_count).cpu().numpy()
    rmse = torch.sqrt(stats[6:8] / sample_count).cpu().numpy()
    var = torch.clamp(stats[6:8] / sample_count - (stats[4:6] / sample_count) ** 2, min=0.0)
    sd = torch.sqrt(var).cpu().numpy()
    return {"loss": float((stats[0] / sample_count).item()), "mae": mae, "me": me, "sd": sd, "rmse": rmse}


def format_metrics(prefix: str, metrics: Dict[str, np.ndarray]) -> str:
    mae = metrics["mae"]
    me = metrics["me"]
    sd = metrics["sd"]
    rmse = metrics["rmse"]
    return (
        f"{prefix} loss={metrics['loss']:.4f} | "
        f"SBP MAE={mae[0]:.2f} ME={me[0]:.2f} SD={sd[0]:.2f} RMSE={rmse[0]:.2f} | "
        f"DBP MAE={mae[1]:.2f} ME={me[1]:.2f} SD={sd[1]:.2f} RMSE={rmse[1]:.2f}"
    )


def build_optimizer_groups(model: OrthogonalFoundationBPModel, args: argparse.Namespace) -> List[dict]:
    encoder_params = [p for p in model.foundation.parameters() if p.requires_grad]
    head_params = [p for p in model.reg_head.parameters() if p.requires_grad]
    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": args.encoder_lr})
    if head_params:
        groups.append({"params": head_params, "lr": args.head_lr})
    return groups


def train_worker(local_rank: int, args: argparse.Namespace, shared_frames=None) -> None:
    set_seed(args.seed + get_rank())
    device = get_device(local_rank)
    if shared_frames is None:
        train_full_df, test_df = load_frames(args)
    else:
        train_full_df, test_df = shared_frames
    if args.use_npz_cache and ("npz_path" not in train_full_df.columns or "npz_path" not in test_df.columns):
        train_full_df, test_df = prepare_npz_cache_frames(args, train_full_df, test_df)

    train_df = train_full_df.reset_index(drop=True)
    eval_df = test_df.reset_index(drop=True)
    cov_cols = ["age", "height_cm", "weight_kg", "has_med_record"]
    cov_stats = compute_cov_stats(train_df, ["age", "height_cm", "weight_kg"])

    train_ds = DirectBPDataset(
        train_df,
        cov_cols=cov_cols,
        cov_stats=cov_stats,
        cycle_len=args.beat_len,
        max_cycles=args.max_beats,
        fs=args.fs,
        apply_preprocessing=True,
        seed=args.seed,
        training=True,
        keep_channel_indices=ECG_PPG_CHANNELS,
        use_npz_cache=args.use_npz_cache,
    )
    eval_ds = DirectBPDataset(
        eval_df,
        cov_cols=cov_cols,
        cov_stats=cov_stats,
        cycle_len=args.beat_len,
        max_cycles=args.max_beats,
        fs=args.fs,
        apply_preprocessing=True,
        seed=args.seed + 1,
        training=False,
        keep_channel_indices=ECG_PPG_CHANNELS,
        use_npz_cache=args.use_npz_cache,
    )

    world_size = get_world_size()
    num_workers = auto_num_workers(args.num_workers, world_size)
    loader_kwargs = build_loader_kwargs(num_workers=num_workers, pin_memory=torch.cuda.is_available(), prefetch_factor=args.prefetch_factor)
    init_fn = make_worker_init_fn(args.seed, get_rank())

    if is_distributed():
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=get_rank(), shuffle=True)
        eval_sampler = DistributedSampler(eval_ds, num_replicas=world_size, rank=get_rank(), shuffle=False)
    else:
        train_sampler = None
        eval_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        worker_init_fn=init_fn,
        **loader_kwargs,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=eval_sampler,
        worker_init_fn=init_fn,
        **loader_kwargs,
    )

    model = OrthogonalFoundationBPModel(
        foundation_ckpt_path=args.foundation_ckpt_path,
        cov_dim=len(cov_cols),
        in_channels=2,
        beat_len=args.beat_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        phase_tokens=args.phase_tokens,
        default_rr_sec=args.default_rr_sec,
        finetune_mode=args.finetune_mode,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
    ).to(device)

    if is_distributed():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    unwrapped = unwrap_model(model)
    optimizer = torch.optim.AdamW(build_optimizer_groups(unwrapped, args), lr=args.head_lr, weight_decay=args.weight_decay)

    main_print(f"device: {device}")
    main_print("script: direct_bp_orthogonal_foundation_singlefile.py")
    main_print(f"BP_DATA_ROOT: {args.bp_data_root}")
    main_print(f"foundation checkpoint: {Path(args.foundation_ckpt_path).expanduser()}")
    main_print(
        f"loaded foundation tensors={unwrapped.loaded_count} missing={unwrapped.missing_count} "
        f"shape_skipped={unwrapped.skipped_shape_count}"
    )
    main_print(
        f"input: beats [B,{args.max_beats},2,{args.beat_len}] channels=ecg,ppg_ir "
        f"time_sec=foundation_extracted source={'npz' if args.use_npz_cache else 'raw_txt_gz'}"
    )
    if args.use_npz_cache:
        main_print(f"npz cache root: {args.bp_npz_cache_root}")
    main_print(f"finetune_mode={args.finetune_mode} selected_blocks={unwrapped.selected_trainable_foundation_layers()}")
    main_print(f"train rows={len(train_df)} subjects={train_df['subject_id'].nunique()} sources={TRAIN_SOURCE_NAMES}")
    main_print(f"test rows={len(eval_df)} subjects={eval_df['subject_id'].nunique()} sources={TEST_SOURCE_NAMES}")

    best_val = float("inf")
    history: List[str] = []
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        sample_count = 0
        for batch in train_loader:
            beats = batch["ppg"].to(device, non_blocking=True)
            time_sec = batch["time_sec"].to(device, non_blocking=True)
            beat_mask = batch["mask"].to(device, non_blocking=True)
            cov = batch["cov"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(beats, time_sec, beat_mask, cov)
            loss = F.mse_loss(pred, y)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), args.grad_clip)
            optimizer.step()

            bs = beats.shape[0]
            running_loss += float(loss.item()) * bs
            sample_count += int(bs)

        train_stats = torch.tensor([running_loss, sample_count], device=device, dtype=torch.float64)
        all_reduce_sum_(train_stats)
        train_loss = float(train_stats[0].item() / max(float(train_stats[1].item()), 1.0))
        test_metrics = evaluate(model, eval_loader, device)
        msg = f"Epoch {epoch:03d} | train loss={train_loss:.4f} | {format_metrics('test', test_metrics)}"
        if is_main_process():
            print(msg, flush=True)
            history.append(msg)

        if test_metrics["loss"] < best_val:
            best_val = float(test_metrics["loss"])
            if is_main_process():
                torch.save(unwrap_model(model).state_dict(), args.ckpt_path)

    if is_distributed():
        torch.distributed.barrier()
    if os.path.exists(args.ckpt_path):
        state = torch.load(args.ckpt_path, map_location=device)
        unwrap_model(model).load_state_dict(state, strict=True)

    final_metrics = evaluate(model, eval_loader, device)
    final_msg = format_metrics("test", final_metrics)
    main_print("\nFinal:", final_msg)
    if is_main_process():
        with open(args.metrics_path, "w", encoding="utf-8") as f:
            for line in history:
                f.write(line + "\n")
            f.write("\nFinal: " + final_msg + "\n")
        print(f"[saved] checkpoint -> {args.ckpt_path}")
        print(f"[saved] metrics    -> {args.metrics_path}")


# ----------------------------- CLI -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-file BP validation/fine-tuning for the orthogonal ECG+PPG foundation encoder."
    )
    default_use_npz_cache = os.environ.get("USE_NPZ_CACHE", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
    default_rebuild_npz_cache = os.environ.get("REBUILD_NPZ_CACHE", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
    default_compress_npz_cache = os.environ.get("COMPRESS_NPZ_CACHE", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
    default_require_existing_npz_cache = os.environ.get("REQUIRE_EXISTING_NPZ_CACHE", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
    parser.add_argument("--bp-data-root", default=os.environ.get("BP_DATA_ROOT", DEFAULT_BP_DATA_ROOT))
    parser.add_argument("--bp-npz-cache-root", default=os.environ.get("BP_NPZ_CACHE_ROOT", DEFAULT_BP_NPZ_CACHE_ROOT))
    parser.add_argument("--foundation-ckpt-path", default=os.environ.get("FOUNDATION_CKPT_PATH", DEFAULT_FOUNDATION_CKPT_PATH))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--beat-len", type=int, default=128)
    parser.add_argument("--max-beats", type=int, default=25)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--nhead", type=int, default=12)
    parser.add_argument("--num-layers", type=int, default=9)
    parser.add_argument("--dim-feedforward", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--phase-tokens", type=int, default=8)
    parser.add_argument("--default-rr-sec", type=float, default=0.8)
    parser.add_argument("--finetune-mode", choices=FINETUNE_MODES, default="head")
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--master-port", type=int, default=0)
    parser.add_argument("--auto-ddp", action="store_true", default=True)
    parser.add_argument("--no-auto-ddp", dest="auto_ddp", action="store_false")
    parser.add_argument("--ckpt-path", default="direct_bp_orthogonal_foundation_singlefile_head_best.pt")
    parser.add_argument("--metrics-path", default="direct_bp_orthogonal_foundation_singlefile_head_metrics.txt")
    parser.add_argument("--col-file", default="ECG+PPG文件索引")
    parser.add_argument("--col-subject", default="账号")
    parser.add_argument("--col-sbp", default="y_sbp")
    parser.add_argument("--col-dbp", default="y_dbp")
    parser.add_argument("--med-cols", nargs="+", default=list(DEFAULT_MEDICATION_COLUMNS))
    parser.add_argument("--skip-bad-samples", action="store_true", default=True)
    parser.add_argument("--no-skip-bad-samples", dest="skip_bad_samples", action="store_false")
    parser.add_argument("--use-npz-cache", action="store_true", default=default_use_npz_cache)
    parser.add_argument("--no-use-npz-cache", dest="use_npz_cache", action="store_false")
    parser.add_argument("--rebuild-npz-cache", action="store_true", default=default_rebuild_npz_cache)
    parser.add_argument("--compress-npz-cache", action="store_true", default=default_compress_npz_cache)
    parser.add_argument("--require-existing-npz-cache", action="store_true", default=default_require_existing_npz_cache)
    parser.add_argument("--npz-log-every", type=int, default=int(os.environ.get("NPZ_LOG_EVERY", "200")))
    args = parser.parse_args()
    if not args.foundation_ckpt_path:
        raise ValueError("Please pass --foundation-ckpt-path or set FOUNDATION_CKPT_PATH to the trained .pt file.")
    if args.d_model % args.nhead != 0:
        raise ValueError(f"--d-model ({args.d_model}) must be divisible by --nhead ({args.nhead})")
    return args


def main() -> None:
    args = parse_args()
    preload_frames = should_auto_spawn(args.auto_ddp) or (args.use_npz_cache and not is_distributed())
    shared_frames = load_frames(args) if preload_frames else None
    if args.use_npz_cache and shared_frames is not None:
        shared_frames = prepare_npz_cache_frames(args, *shared_frames)
    launch(train_worker, args, shared_frames)


if __name__ == "__main__":
    main()
