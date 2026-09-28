from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from anyppg_bp_model_staged_ft import AnyPPGEncoderStagedFT
except ImportError:
    AnyPPGEncoderStagedFT = None
try:
    from csfm_all_windows_dataset import _load_preprocess_signal, _resolve_csfm_project_root
    from csfm_bp_model_staged_ft import CSFMEncoderStagedFT
except ImportError:
    _load_preprocess_signal = None
    _resolve_csfm_project_root = None
    CSFMEncoderStagedFT = None
try:
    from pulseppg_bp_model import PulsePPGEncoder
except ImportError:
    PulsePPGEncoder = None

DEFAULT_ORTHOGONAL_MODULE_PATH = str(Path(__file__).resolve().with_name("direct_bp_orthogonal_foundation_singlefile.py"))
DEFAULT_ORTHOGONAL_CKPT_PATH = (
    "/data-ai/sl20200894/Code/foundation_dual_view_alignment_packed_bundle_20260628/"
    "dual_view_virtual_r_alignment_packed_amp.pt"
)


@dataclass
class ButRecord:
    record_id: str
    subject_id: str
    quality: int
    hr: float
    sbp: float
    dbp: float
    ppg: np.ndarray
    ecg: np.ndarray | None = None


def set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def reset_probe_seed(seed: int, fold: int, task_offset: int) -> None:
    set_seed(seed + task_offset * 1000 + fold)


METRIC_LOG_ORDER = (
    "mae",
    "me",
    "sd",
    "rmse",
    "pearson",
    "sbp_mae",
    "sbp_me",
    "sbp_sd",
    "sbp_rmse",
    "sbp_pearson",
    "dbp_mae",
    "dbp_me",
    "dbp_sd",
    "dbp_rmse",
    "dbp_pearson",
    "accuracy",
    "balanced_accuracy",
    "auroc",
    "auprc",
    "f1",
    "macro_f1",
    "precision",
    "sensitivity",
    "specificity",
)


def format_metric_log(report: dict[str, float]) -> str:
    keys = [key for key in METRIC_LOG_ORDER if key in report]
    keys.extend(key for key in sorted(report) if key not in keys)
    return " ".join(f"{key}={float(report[key]):.4f}" for key in keys)


def log_epoch_metrics(
    task: str,
    fold: int,
    epoch: int,
    total_epochs: int,
    train_report: dict[str, float],
    test_report: dict[str, float],
) -> None:
    print(
        f"    [{task}] fold={fold} epoch={epoch:03d}/{total_epochs} "
        f"train({format_metric_log(train_report)}) test({format_metric_log(test_report)})",
        flush=True,
    )


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    sd = max(float(x.std()), 1e-6)
    return ((x - float(x.mean())) / sd).astype(np.float32, copy=False)


def resample_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == target_len:
        return x
    if x.size <= 1:
        return np.zeros(target_len, dtype=np.float32)
    old = np.linspace(0.0, 1.0, x.size, dtype=np.float32)
    new = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    return np.interp(new, old, x).astype(np.float32, copy=False)


def parse_int_list(value: str | Sequence[int]) -> list[int]:
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return [int(p) for p in parts]
    return [int(p) for p in value]


def requested_models(args) -> set[str]:
    return {m.strip().lower() for m in str(args.models).split(",") if m.strip()}


def needs_ecg_ppg(args) -> bool:
    return bool({"csfm", "orthogonal"} & requested_models(args))


def add_orthogonal_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--orthogonal-module-path", default=DEFAULT_ORTHOGONAL_MODULE_PATH)
    parser.add_argument("--orthogonal-ckpt-path", default=DEFAULT_ORTHOGONAL_CKPT_PATH)
    parser.add_argument("--orthogonal-fs", type=int, default=250)
    parser.add_argument("--orthogonal-beat-len", type=int, default=128)
    parser.add_argument("--orthogonal-max-beats", type=int, default=25)
    parser.add_argument("--orthogonal-d-model", type=int, default=768)
    parser.add_argument("--orthogonal-nhead", type=int, default=12)
    parser.add_argument("--orthogonal-num-layers", type=int, default=9)
    parser.add_argument("--orthogonal-dim-feedforward", type=int, default=2048)
    parser.add_argument("--orthogonal-dropout", type=float, default=0.1)
    parser.add_argument("--orthogonal-phase-tokens", type=int, default=8)
    return parser


def normalized_channel_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def choose_signal_channel(
    sig: np.ndarray,
    names: Sequence[object],
    aliases: Sequence[str],
) -> np.ndarray | None:
    if sig.ndim != 2 or sig.shape[1] < 1:
        return None
    norm_names = [normalized_channel_name(n) for n in names]
    norm_aliases = [normalized_channel_name(a) for a in aliases if str(a).strip()]
    for alias in norm_aliases:
        if alias in norm_names:
            return sig[:, norm_names.index(alias)]
    for alias in norm_aliases:
        for idx, name in enumerate(norm_names):
            if alias and (alias in name or name in alias):
                return sig[:, idx]
    return None


def read_csv_utf8_sig(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_bp(value: object) -> tuple[float, float] | None:
    nums = re.findall(r"\d+(?:\.\d+)?", str(value))
    if len(nums) < 2:
        return None
    sbp, dbp = float(nums[0]), float(nums[1])
    if not (60 <= sbp <= 260 and 30 <= dbp <= 160):
        return None
    return sbp, dbp


def read_but_ppg(record_dir: Path, record_id: str) -> np.ndarray:
    import wfdb

    rec_path = record_dir / f"{record_id}_PPG"
    rec = wfdb.rdrecord(str(rec_path), physical=True)
    sig = np.asarray(rec.p_signal, dtype=np.float32)
    names = list(getattr(rec, "sig_name", []) or [])
    if sig.ndim != 2:
        sig = sig.reshape(-1, 1)
    if "PPG_G" in names and sig.shape[1] > names.index("PPG_G"):
        x = sig[:, names.index("PPG_G")]
    elif sig.shape[0] == 1 and sig.shape[1] >= 30:
        # Some early BUT PPG headers are represented as 300 one-sample signals.
        x = sig[0, :]
    elif sig.shape[1] >= 3:
        x = sig[:, 1]
    elif sig.shape[1] >= 1:
        x = sig[:, 0]
    else:
        x = sig.reshape(-1)
    return zscore(x)


def read_but_ecg(record_dir: Path, record_id: str, aliases: Sequence[str]) -> np.ndarray | None:
    import wfdb

    for suffix in ("ECG", "ecg"):
        rec_path = record_dir / f"{record_id}_{suffix}"
        if not (rec_path.with_suffix(".hea")).exists():
            continue
        rec = wfdb.rdrecord(str(rec_path), physical=True)
        sig = np.asarray(rec.p_signal, dtype=np.float32)
        if sig.ndim != 2:
            sig = sig.reshape(-1, 1)
        names = list(getattr(rec, "sig_name", []) or [])
        x = choose_signal_channel(sig, names, aliases)
        if x is None and sig.shape[1] >= 1:
            x = sig[:, 0]
        if x is not None:
            return zscore(x)
    return None


def load_but_records(args: argparse.Namespace) -> list[ButRecord]:
    root = Path(args.data_root)
    q_rows = read_csv_utf8_sig(root / "quality-hr-ann.csv")
    subject_rows = read_csv_utf8_sig(root / "subject-info.csv")
    bp_by_id: dict[str, tuple[float, float]] = {}
    for row in subject_rows:
        rid = str(row.get("ID", "")).strip()
        bp = parse_bp(row.get("Blood pressure [mmHg]", ""))
        if rid and bp is not None:
            bp_by_id[rid] = bp
    records: list[ButRecord] = []
    for row in q_rows:
        rid = str(row.get("ID", "")).strip()
        if not rid:
            continue
        try:
            quality = int(float(row["Quality"]))
            hr = float(row["HR"])
        except Exception:
            continue
        rec_dir = root / rid
        if not (rec_dir / f"{rid}_PPG.hea").exists():
            continue
        try:
            ppg = read_but_ppg(rec_dir, rid)
            ecg = read_but_ecg(rec_dir, rid, args.ecg_channel.split(","))
        except Exception as exc:
            print(f"[warn] skip {rid}: {exc!r}", flush=True)
            continue
        if ecg is None and needs_ecg_ppg(args) and not args.allow_missing_ecg:
            print(f"[warn] skip {rid}: no ECG channel/file for ECG+PPG input", flush=True)
            continue
        if ppg.size < 30 or not np.isfinite(ppg).any():
            continue
        bp = bp_by_id.get(rid)
        sbp, dbp = bp if bp is not None else (float("nan"), float("nan"))
        records.append(ButRecord(record_id=rid, subject_id=rid[:3], quality=quality, hr=hr, sbp=sbp, dbp=dbp, ppg=ppg, ecg=ecg))
        if args.max_records > 0 and len(records) >= args.max_records:
            break
    print(
        f"[data] loaded records={len(records)} subjects={len(set(r.subject_id for r in records))} "
        f"quality_pos={sum(r.quality for r in records)} bp_valid={sum(np.isfinite(r.sbp) and np.isfinite(r.dbp) for r in records)}",
        flush=True,
    )
    return records


def make_group_folds(records: Sequence[ButRecord], n_splits: int, seed: int):
    groups = np.asarray([r.subject_id for r in records], dtype=object)
    y = np.asarray([r.quality for r in records], dtype=np.int64)
    n_splits = min(n_splits, len(set(groups)))
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [(train.tolist(), test.tolist()) for train, test in splitter.split(np.zeros(len(records)), y, groups)]
    except Exception as exc:
        print(f"[warn] StratifiedGroupKFold unavailable ({exc!r}); using GroupKFold.", flush=True)
        from sklearn.model_selection import GroupKFold

        splitter = GroupKFold(n_splits=n_splits)
        return [(train.tolist(), test.tolist()) for train, test in splitter.split(np.zeros(len(records)), y, groups)]


def metric_classification(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(np.int64)
    prob = np.nan_to_num(prob.astype(np.float64), nan=0.5, posinf=1.0, neginf=0.0)
    pred = (prob >= 0.5).astype(np.int64)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    out = {
        "accuracy": float((pred == y_true).mean()),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
    }
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        out["auroc"] = float(roc_auc_score(y_true, prob)) if len(set(y_true.tolist())) == 2 else float("nan")
        out["auprc"] = float(average_precision_score(y_true, prob))
    except Exception:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out


def metric_regression(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(np.float64)
    pred = np.nan_to_num(pred.astype(np.float64), nan=float(np.nanmean(y_true)))
    err = pred - y_true
    out = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "me": float(np.mean(err)),
        "sd": float(np.std(err)),
    }
    if y_true.size > 1 and np.std(y_true) > 1e-9 and np.std(pred) > 1e-9:
        out["pearson"] = float(np.corrcoef(y_true, pred)[0, 1])
    else:
        out["pearson"] = float("nan")
    return out


def metric_bp_report(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    report = {}
    for col, name in enumerate(("sbp", "dbp")):
        metrics = metric_regression(y_true[:, col], pred[:, col])
        report.update({f"{name}_{key}": value for key, value in metrics.items()})
    return report


class MLPHead(nn.Module):
    def __init__(self, dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def standardize_features(x: np.ndarray, train_idx: Sequence[int]) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    mu = x[train_idx].mean(axis=0, keepdims=True)
    sd = np.maximum(x[train_idx].std(axis=0, keepdims=True), 1e-6)
    return np.nan_to_num(((x - mu) / sd).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def train_quality_head(x, y, folds, args, device):
    reports = []
    for fold, (train_idx, test_idx) in enumerate(folds, 1):
        reset_probe_seed(args.seed, fold, 11)
        xs = standardize_features(x, train_idx)
        x_train = torch.from_numpy(xs[train_idx]).float()
        y_train = torch.from_numpy(y[train_idx].astype(np.float32))
        x_test = torch.from_numpy(xs[test_idx]).float().to(device)
        head = MLPHead(x.shape[1], args.head_hidden, 1, args.dropout).to(device)
        pos = float(y_train.sum().item())
        neg = float(y_train.numel() - pos)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], device=device))
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
        for epoch in range(1, args.epochs + 1):
            head.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(head(xb).squeeze(-1), yb)
                loss.backward()
                opt.step()
            head.eval()
            with torch.no_grad():
                train_prob = torch.sigmoid(head(x_train.to(device)).squeeze(-1)).cpu().numpy()
                test_prob = torch.sigmoid(head(x_test).squeeze(-1)).cpu().numpy()
            log_epoch_metrics(
                "quality",
                fold,
                epoch,
                args.epochs,
                metric_classification(y[train_idx], train_prob),
                metric_classification(y[test_idx], test_prob),
            )
        head.eval()
        with torch.no_grad():
            prob = torch.sigmoid(head(x_test).squeeze(-1)).cpu().numpy()
        rep = metric_classification(y[test_idx], prob)
        print(f"    [quality] fold={fold} {rep}", flush=True)
        reports.append(rep)
    return summarize_reports(reports)


def train_hr_head(x, y, folds, args, device):
    reports = []
    for fold, (train_idx, test_idx) in enumerate(folds, 1):
        reset_probe_seed(args.seed, fold, 12)
        xs = standardize_features(x, train_idx)
        y_mu = float(y[train_idx].mean())
        y_sd = max(float(y[train_idx].std()), 1e-6)
        x_train = torch.from_numpy(xs[train_idx]).float()
        y_train = torch.from_numpy(((y[train_idx] - y_mu) / y_sd).astype(np.float32))
        x_test = torch.from_numpy(xs[test_idx]).float().to(device)
        head = MLPHead(x.shape[1], args.head_hidden, 1, args.dropout).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
        for epoch in range(1, args.epochs + 1):
            head.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = F.smooth_l1_loss(head(xb).squeeze(-1), yb)
                loss.backward()
                opt.step()
            head.eval()
            with torch.no_grad():
                train_pred = head(x_train.to(device)).squeeze(-1).cpu().numpy() * y_sd + y_mu
                test_pred = head(x_test).squeeze(-1).cpu().numpy() * y_sd + y_mu
            log_epoch_metrics(
                "hr",
                fold,
                epoch,
                args.epochs,
                metric_regression(y[train_idx], train_pred),
                metric_regression(y[test_idx], test_pred),
            )
        head.eval()
        with torch.no_grad():
            pred = head(x_test).squeeze(-1).cpu().numpy() * y_sd + y_mu
        rep = metric_regression(y[test_idx], pred)
        print(f"    [hr] fold={fold} {rep}", flush=True)
        reports.append(rep)
    return summarize_reports(reports)


def train_bp_head(x, y, folds, args, device):
    reports = []
    valid = np.isfinite(y).all(axis=1)
    for fold, (train_idx, test_idx) in enumerate(folds, 1):
        reset_probe_seed(args.seed, fold, 13)
        train_idx = [i for i in train_idx if valid[i]]
        test_idx = [i for i in test_idx if valid[i]]
        if len(train_idx) < 2 or len(test_idx) < 1:
            print(f"    [bp] fold={fold} skipped train_n={len(train_idx)} test_n={len(test_idx)}", flush=True)
            continue
        xs = standardize_features(x, train_idx)
        y_mu = y[train_idx].mean(axis=0, keepdims=True)
        y_sd = np.maximum(y[train_idx].std(axis=0, keepdims=True), 1e-6)
        x_train = torch.from_numpy(xs[train_idx]).float()
        y_train = torch.from_numpy(((y[train_idx] - y_mu) / y_sd).astype(np.float32))
        x_test = torch.from_numpy(xs[test_idx]).float().to(device)
        head = MLPHead(x.shape[1], args.head_hidden, 2, args.dropout).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
        for epoch in range(1, args.epochs + 1):
            head.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = F.smooth_l1_loss(head(xb), yb)
                loss.backward()
                opt.step()
            head.eval()
            with torch.no_grad():
                train_pred = head(x_train.to(device)).cpu().numpy() * y_sd + y_mu
                test_pred = head(x_test).cpu().numpy() * y_sd + y_mu
            log_epoch_metrics(
                "bp",
                fold,
                epoch,
                args.epochs,
                metric_bp_report(y[train_idx], train_pred),
                metric_bp_report(y[test_idx], test_pred),
            )
        head.eval()
        with torch.no_grad():
            pred = head(x_test).cpu().numpy() * y_sd + y_mu
        rep = metric_bp_report(y[test_idx], pred)
        print(f"    [bp] fold={fold} {rep}", flush=True)
        reports.append(rep)
    return summarize_reports(reports) if reports else {}


def summarize_reports(reports: list[dict[str, float]]) -> dict[str, float]:
    out = {}
    for key in sorted(reports[0]):
        vals = np.asarray([r[key] for r in reports], dtype=np.float64)
        out[f"{key}_mean"] = float(np.nanmean(vals))
        out[f"{key}_std"] = float(np.nanstd(vals))
    return out


def prepare_windows(records: Sequence[ButRecord], model_name: str, args) -> np.ndarray:
    target_fs = {"anyppg": 125, "pulseppg": 50, "csfm": 250, "orthogonal": args.orthogonal_fs}[model_name]
    target_len = int(round(args.window_sec * target_fs))
    xs = []
    preprocess_signal = None
    project_root = None
    if model_name == "csfm":
        if _resolve_csfm_project_root is None or _load_preprocess_signal is None:
            raise ImportError("CSFM preprocessing modules are not available; run with --models orthogonal or install CSFM code.")
        project_root = _resolve_csfm_project_root()
        preprocess_signal = _load_preprocess_signal(project_root)
        csfm_channel_ids = parse_int_list(args.csfm_channel_ids)
        if len(csfm_channel_ids) != 2:
            raise ValueError(f"CSFM ECG+PPG benchmark expects exactly two channel ids, got {csfm_channel_ids}")
    for rec in records:
        if model_name == "csfm":
            assert preprocess_signal is not None and project_root is not None
            if rec.ecg is None:
                raise ValueError(
                    f"Record {rec.record_id} has no ECG signal; CSFM benchmark is configured as ECG+PPG."
                )
            ecg = resample_1d(rec.ecg, target_len)
            ppg = resample_1d(rec.ppg, target_len)
            x = preprocess_signal(
                signal=np.stack([ecg, ppg], axis=0),
                channels=csfm_channel_ids,
                fs=target_fs,
                project_root=project_root,
            )
            xs.append(np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))
        elif model_name == "orthogonal":
            if rec.ecg is None:
                raise ValueError(
                    f"Record {rec.record_id} has no ECG signal; orthogonal foundation benchmark is ECG+PPG."
                )
            ecg = zscore(resample_1d(rec.ecg, target_len))
            ppg = zscore(resample_1d(rec.ppg, target_len))
            xs.append(np.stack([ecg, ppg], axis=0).astype(np.float32))
        else:
            x = resample_1d(rec.ppg, target_len)
            xs.append(zscore(x)[None, :])
    return np.stack(xs, axis=0).astype(np.float32)


class OrthogonalFoundationEncoder(nn.Module):
    def __init__(
        self,
        module_path: str | None,
        ckpt_path: str | None,
        device: torch.device,
        *,
        fs: int = 250,
        beat_len: int = 128,
        max_beats: int = 25,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 9,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        phase_tokens: int = 8,
    ):
        super().__init__()
        self.fs = int(fs)
        self.beat_len = int(beat_len)
        self.max_beats = int(max_beats)
        self.device_for_pretrain = device
        module_path = module_path or os.environ.get("ORTHOGONAL_MODULE_PATH") or DEFAULT_ORTHOGONAL_MODULE_PATH
        ckpt_path = ckpt_path or os.environ.get("ORTHOGONAL_CKPT_PATH") or DEFAULT_ORTHOGONAL_CKPT_PATH
        mod = self._load_module(module_path)
        self.extract_foundation_beat_arrays = mod.extract_foundation_beat_arrays
        self.sample_or_pad_foundation_beats = mod.sample_or_pad_foundation_beats
        self.foundation = mod.ECGPPGMultiTaskTransformer(
            in_channels=2,
            beat_len=beat_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            phase_tokens=phase_tokens,
        )
        matched, missing, skipped = mod.load_matching_foundation_state(self.foundation, ckpt_path)
        print(
            f"[orthogonal] loaded checkpoint={ckpt_path} matched={matched} missing={missing} skipped_shape={skipped}",
            flush=True,
        )
        for param in self.foundation.parameters():
            param.requires_grad = False

    @staticmethod
    def _load_module(module_path: str):
        path = Path(module_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"orthogonal module not found: {path}")
        spec = importlib.util.spec_from_file_location("orthogonal_foundation_singlefile", str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import orthogonal module from {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _batch_to_foundation_inputs(self, xb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        arr = xb.detach().cpu().numpy().astype(np.float32)
        beat_list, time_list, mask_list = [], [], []
        rng = np.random.RandomState(0)
        for item in arr:
            sig_tc = np.transpose(item, (1, 0))
            arrays = self.extract_foundation_beat_arrays(
                sig_tc,
                fs=self.fs,
                input_cols=(0, 1),
                ecg_col=0,
                beat_len=self.beat_len,
            )
            beats, time_sec, beat_mask = self.sample_or_pad_foundation_beats(
                arrays["beats"],
                arrays["time_sec"],
                max_beats=self.max_beats,
                rng=rng,
                training=False,
            )
            beat_list.append(beats)
            time_list.append(time_sec)
            mask_list.append(beat_mask)
        beats_t = torch.from_numpy(np.stack(beat_list)).float().to(self.device_for_pretrain)
        time_t = torch.from_numpy(np.stack(time_list)).float().to(self.device_for_pretrain)
        mask_t = torch.from_numpy(np.stack(mask_list)).bool().to(self.device_for_pretrain)
        return beats_t, time_t, mask_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beats, time_sec, beat_mask = self._batch_to_foundation_inputs(x)
        ctx = self.foundation.encode_context(beats, time_sec, beat_mask, pretrain_mask=None)
        return ctx["cls"]


def build_encoder(model_name: str, args, device):
    if model_name == "anyppg":
        if AnyPPGEncoderStagedFT is None:
            raise ImportError("AnyPPG module is not available; pass --models without anyppg or install AnyPPG code.")
        return AnyPPGEncoderStagedFT(module_path=args.anyppg_module_path or None, ckpt_path=args.anyppg_ckpt_path or None, freeze=True).to(device)
    if model_name == "pulseppg":
        if PulsePPGEncoder is None:
            raise ImportError("PulsePPG module is not available; pass --models without pulseppg or install PulsePPG code.")
        return PulsePPGEncoder(module_path=args.pulseppg_module_path or None, ckpt_path=args.pulseppg_ckpt_path or None, freeze=True).to(device)
    if model_name == "csfm":
        if CSFMEncoderStagedFT is None:
            raise ImportError("CSFM module is not available; pass --models without csfm or install CSFM code.")
        return CSFMEncoderStagedFT(
            loader_path=args.csfm_loader_path or None,
            project_root=args.csfm_project_root or None,
            ckpt_path=args.csfm_ckpt_path or None,
            variant=args.csfm_variant,
            device=str(device),
            finetune_mode="head",
            channel_ids=parse_int_list(args.csfm_channel_ids),
        ).to(device)
    if model_name == "orthogonal":
        return OrthogonalFoundationEncoder(
            module_path=args.orthogonal_module_path or None,
            ckpt_path=args.orthogonal_ckpt_path or None,
            device=device,
            fs=args.orthogonal_fs,
            beat_len=args.orthogonal_beat_len,
            max_beats=args.orthogonal_max_beats,
            d_model=args.orthogonal_d_model,
            nhead=args.orthogonal_nhead,
            num_layers=args.orthogonal_num_layers,
            dim_feedforward=args.orthogonal_dim_feedforward,
            dropout=args.orthogonal_dropout,
            phase_tokens=args.orthogonal_phase_tokens,
        ).to(device)
    raise ValueError(model_name)


def extract_embeddings(windows: np.ndarray, model_name: str, args, device) -> np.ndarray:
    encoder = build_encoder(model_name, args, device)
    encoder.eval()
    outs = []
    loader = DataLoader(torch.from_numpy(windows).float(), batch_size=args.embed_batch_size, shuffle=False)
    print(f"[{model_name}] embedding windows={len(windows)} batches={len(loader)}", flush=True)
    with torch.no_grad():
        for i, xb in enumerate(loader, 1):
            out = encoder(xb.to(device))
            if out.dim() > 2:
                out = out.mean(dim=tuple(range(2, out.dim())))
            out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            outs.append(out.cpu().float().numpy())
            if i % max(args.log_every_batches, 1) == 0:
                print(f"[{model_name}] embedded {i}/{len(loader)}", flush=True)
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(outs, axis=0).astype(np.float32)


def run_model(model_name: str, records: Sequence[ButRecord], folds, args, device, out_dir: Path):
    print(f"\n===== MODEL {model_name} =====", flush=True)
    windows = prepare_windows(records, model_name, args)
    print(f"[{model_name}] input shape={windows.shape}", flush=True)
    emb_path = out_dir / f"{model_name}_embeddings.npy"
    emb = extract_embeddings(windows, model_name, args, device)
    np.save(emb_path, emb)
    y_quality = np.asarray([r.quality for r in records], dtype=np.int64)
    y_hr = np.asarray([r.hr for r in records], dtype=np.float32)
    y_bp = np.asarray([[r.sbp, r.dbp] for r in records], dtype=np.float32)
    summary = {
        "model": model_name,
        "quality": train_quality_head(emb, y_quality, folds, args, device),
        "hr": train_hr_head(emb, y_hr, folds, args, device),
        "bp": train_bp_head(emb, y_bp, folds, args, device),
    }
    with (out_dir / f"{model_name}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[{model_name}] SUMMARY {summary}", flush=True)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="BUT PPG downstream benchmark: BP/HR regression and PPG quality classification.")
    parser.add_argument("--data-root", default="/data-ai/sl20200894/downstream_datasets/BUT_PPG/2.0.0")
    parser.add_argument("--output-dir", default="/data-ai/sl20200894/Code/downstream_benchmarks_orthogonal/butppg_orthogonal")
    parser.add_argument("--models", default="orthogonal")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--ecg-channel", default="ECG,ECG_II,II,I,chest_ecg")
    parser.add_argument("--allow-missing-ecg", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embed-batch-size", type=int, default=128)
    parser.add_argument("--head-hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-every-batches", type=int, default=20)
    parser.add_argument("--anyppg-module-path", default="")
    parser.add_argument("--anyppg-ckpt-path", default="")
    parser.add_argument("--pulseppg-module-path", default="")
    parser.add_argument("--pulseppg-ckpt-path", default="")
    parser.add_argument("--csfm-loader-path", default="")
    parser.add_argument("--csfm-project-root", default="")
    parser.add_argument("--csfm-ckpt-path", default="")
    parser.add_argument("--csfm-variant", default="Base")
    parser.add_argument("--csfm-channel-ids", default="1,12")
    add_orthogonal_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[start] {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"[config] {vars(args)}", flush=True)
    print(f"[device] {device}", flush=True)
    records = load_but_records(args)
    folds = make_group_folds(records, args.folds, args.seed)
    manifest = [
        {
            "fold": i + 1,
            "train_subjects": sorted({records[j].subject_id for j in train_idx}),
            "test_subjects": sorted({records[j].subject_id for j in test_idx}),
            "train_n": len(train_idx),
            "test_n": len(test_idx),
        }
        for i, (train_idx, test_idx) in enumerate(folds)
    ]
    with (out_dir / "folds.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    summaries = []
    for model_name in [m.strip().lower() for m in args.models.split(",") if m.strip()]:
        summaries.append(run_model(model_name, records, folds, args, device, out_dir))
    compact = {
        "task": "BUT_PPG BP/HR regression and signal-quality classification",
        "anyppg_input": "PPG",
        "pulseppg_input": "PPG",
        "csfm_input": "ECG+PPG",
        "csfm_channel_ids": args.csfm_channel_ids,
        "orthogonal_input": "ECG+PPG beats [B,25,2,128]",
        "orthogonal_ckpt_path": args.orthogonal_ckpt_path,
        "models": {s["model"]: s for s in summaries},
    }
    with (out_dir / "all_models_summary.json").open("w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False, indent=2)
    print("\n===== ALL MODELS SUMMARY =====", flush=True)
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    print(f"[done] {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
