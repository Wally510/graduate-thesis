#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

FORBIDDEN = "/bingding" + "/301/BP"
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
TRACKS = ("ppg_only", "ecg_ppg_direct", "ecg_only")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def import_model(path: Path):
    spec = importlib.util.spec_from_file_location("splitenc8_pat_model", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_corr(y: np.ndarray, pred: np.ndarray, kind: str) -> float:
    if len(y) < 2 or np.std(y) < 1e-12 or np.std(pred) < 1e-12:
        return float("nan")
    result = stats.pearsonr(y, pred) if kind == "pearson" else stats.spearmanr(y, pred)
    return float(result.statistic)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - y
    return {
        "n": int(len(y)),
        "mae_ms": float(mean_absolute_error(y, pred)),
        "rmse_ms": float(math.sqrt(mean_squared_error(y, pred))),
        "bias_ms": float(np.mean(err)),
        "sd_error_ms": float(np.std(err, ddof=1)) if len(err) > 1 else float("nan"),
        "pearson_r": safe_corr(y, pred, "pearson"),
        "spearman_rho": safe_corr(y, pred, "spearman"),
        "r2": float(r2_score(y, pred)) if len(y) > 1 else float("nan"),
    }


def macro_metrics(frame: pd.DataFrame) -> dict:
    rows = [metrics(g.y_true.to_numpy(), g.y_pred.to_numpy())
            for _, g in frame.groupby("group_id") if len(g) >= 2]
    keys = ("mae_ms", "rmse_ms", "bias_ms", "sd_error_ms",
            "pearson_r", "spearman_rho", "r2")
    return {
        **{f"{key}_macro": float(np.nanmean([r[key] for r in rows]))
           for key in keys},
        "macro_group_count": len(rows),
    }


def finite_interp(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).copy()
    good = np.isfinite(x)
    if good.sum() >= 2 and not good.all():
        idx = np.arange(len(x))
        x[~good] = np.interp(idx[~good], idx[good], x[good])
    return np.nan_to_num(x, nan=0.0).astype(np.float32)


def bandpass(x: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    x = finite_interp(x)
    high = min(high, 0.45 * fs)
    if len(x) < int(fs) or high <= low:
        return x
    b, a = signal.butter(3, [low, high], btype="bandpass", fs=fs)
    return signal.filtfilt(b, a, x).astype(np.float32)


def robust_zscore(x: np.ndarray) -> np.ndarray:
    x = finite_interp(x)
    med = float(np.median(x))
    scale = float(np.median(np.abs(x - med))) * 1.4826
    if scale < 1e-6:
        scale = float(np.std(x))
    return ((x - med) / max(scale, 1e-6)).astype(np.float32)


def resample_beat(x: np.ndarray, n: int = 128) -> np.ndarray:
    if len(x) < 2:
        return np.zeros(n, dtype=np.float32)
    old = np.linspace(0.0, 1.0, len(x), endpoint=False)
    new = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.interp(new, old, x).astype(np.float32)


def center_segments(processed: np.ndarray, centers: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(sorted(set(map(int, centers))), dtype=np.int64)
    n = len(processed)
    if len(centers) == 0:
        centers = np.arange(int(0.5 * fs), n, max(2, int(0.8 * fs)), dtype=np.int64)
    if len(centers) == 0:
        centers = np.asarray([n // 2], dtype=np.int64)
    default = int(np.median(np.diff(centers))) if len(centers) > 1 else max(2, int(0.8 * fs))
    beats, times = [], []
    first = None
    for i, center in enumerate(centers):
        prev = centers[i - 1] if i else center - default
        nxt = centers[i + 1] if i + 1 < len(centers) else center + default
        start = max(0, int(round((prev + center) / 2)))
        end = min(n, int(round((center + nxt) / 2)))
        if end <= start:
            continue
        channels = [resample_beat(processed[start:end, c]) for c in range(processed.shape[1])]
        beat = np.stack(channels)
        beat = (beat - beat.mean(-1, keepdims=True)) / (beat.std(-1, keepdims=True) + 1e-6)
        beats.append(beat.astype(np.float32))
        t = center / fs
        first = t if first is None else first
        times.append(t - first)
    return np.stack(beats), np.asarray(times, dtype=np.float32)


def ppg_centers(ppg: np.ndarray, fs: float) -> np.ndarray:
    distance = max(1, int(0.30 * fs))
    prominence = max(0.15, 0.20 * float(np.std(ppg)))
    peaks, _ = signal.find_peaks(ppg, distance=distance, prominence=prominence)
    if len(peaks) < 3:
        peaks, _ = signal.find_peaks(ppg, distance=distance)
    return peaks.astype(np.int64)


def ecg_centers(ecg: np.ndarray, fs: float) -> np.ndarray:
    der = np.diff(ecg, prepend=ecg[0])
    energy = der * der
    win = max(1, int(0.10 * fs))
    mwa = np.convolve(energy, np.ones(win) / win, mode="same")
    med = float(np.median(mwa))
    mad = float(np.median(np.abs(mwa - med))) + 1e-8
    peaks, _ = signal.find_peaks(
        mwa, height=med + 3.0 * 1.4826 * mad,
        distance=max(1, int(0.18 * fs)),
    )
    refined = []
    w = max(1, int(0.05 * fs))
    for p in peaks:
        lo, hi = max(0, p - w), min(len(ecg), p + w + 1)
        refined.append(lo + int(np.argmax(ecg[lo:hi])))
    return np.asarray(refined, dtype=np.int64)


def make_beats(ecg: np.ndarray | None, ppg: np.ndarray | None, fs: float, track: str,
               max_beats: int = 25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if track == "ppg_only":
        if ppg is None:
            raise ValueError("ppg_only requires PPG")
        p = robust_zscore(bandpass(ppg, fs, 0.3, 8.0))
        beats, times = center_segments(p[:, None], ppg_centers(p, fs), fs)
    elif track == "ecg_only":
        if ecg is None:
            raise ValueError("ecg_only requires ECG")
        e = robust_zscore(bandpass(ecg, fs, 0.5, 40.0))
        beats, times = center_segments(e[:, None], ecg_centers(e, fs), fs)
    else:
        if ecg is None or ppg is None:
            raise ValueError("ecg_ppg_direct requires real ECG and PPG")
        e = robust_zscore(bandpass(ecg, fs, 0.5, 40.0))
        p = robust_zscore(bandpass(ppg, fs, 0.3, 8.0))
        processed = np.stack([e, p], axis=1)
        beats, times = center_segments(processed, ecg_centers(e, fs), fs)
    if len(beats) > max_beats:
        idx = np.linspace(0, len(beats) - 1, max_beats).astype(int)
        beats, times = beats[idx], times[idx]
    out = np.zeros((max_beats, beats.shape[1], 128), dtype=np.float32)
    time_out = np.zeros(max_beats, dtype=np.float32)
    mask = np.zeros(max_beats, dtype=bool)
    out[:len(beats)] = beats
    time_out[:len(times)] = times
    mask[:len(beats)] = True
    return out, time_out, mask


def load_model(args, device):
    module = import_model(Path(args.model_module))
    model = module.ECGPPGMultiTaskTransformer(
        in_channels=2, beat_len=128, d_model=768, nhead=12,
        num_layers=9, dim_feedforward=2048, dropout=0.1, phase_tokens=8,
    )
    load_result = module.load_matching_foundation_state(model, args.checkpoint)
    matched, missing, skipped, missing_keys = load_result[:4]
    critical = ("ecg_encoder.", "ppg_encoder.", "fusion_gate.", "time_encoding.",
                "input_mode_embed.", "blocks.", "norm.", "reliability_pool.")
    if skipped or any(k.startswith(critical) for k in missing_keys):
        raise RuntimeError("SplitEnc8 backbone未严格完整加载")
    for param in model.parameters():
        param.requires_grad = False
    model.eval().to(device)
    return model, {"matched": matched, "missing_heads": missing, "skipped": skipped}


def embed(args, frame: pd.DataFrame, positions: np.ndarray, archive, output: Path) -> tuple[np.ndarray, dict]:
    import torch

    device = torch.device(args.device)
    model, load_meta = load_model(args, device)
    fs_all = archive["fs"]
    ecg_all = None if args.track == "ppg_only" else archive["ecg"]
    ppg_all = None if args.track == "ecg_only" else archive["ppg"]
    embeddings, beat_counts = [], []
    started = time.time()
    for batch_start in range(0, len(frame), args.batch_size):
        batch_pos = positions[batch_start:batch_start + args.batch_size]
        made = [
            make_beats(
                None if ecg_all is None else ecg_all[i],
                None if ppg_all is None else ppg_all[i],
                float(fs_all[i]),
                args.track,
            )
            for i in batch_pos
        ]
        beats = torch.from_numpy(np.stack([m[0] for m in made])).to(device)
        times = torch.from_numpy(np.stack([m[1] for m in made])).to(device)
        masks = torch.from_numpy(np.stack([m[2] for m in made])).to(device)
        beat_counts.extend(masks.sum(1).cpu().numpy().astype(int).tolist())
        with torch.inference_mode():
            if args.track == "ecg_ppg_direct":
                mode = torch.zeros(len(made), dtype=torch.long, device=device)
                z = model.encode_context(
                    beats, times, masks, pretrain_mask=None,
                    modality_mask=torch.ones((len(made), 2), dtype=torch.bool, device=device),
                    input_mode=mode,
                )["cls"]
            else:
                b, k, _, length = beats.shape
                encoder = model.ppg_encoder if args.track == "ppg_only" else model.ecg_encoder
                phase = encoder.forward_phase(beats.reshape(b * k, 1, length)).reshape(b, k, 8, 768)
                weight = masks[:, :, None, None].to(phase.dtype)
                z = (phase * weight).sum((1, 2)) / (weight.sum((1, 2)) * 8.0).clamp_min(1.0)
        embeddings.append(z.cpu().float().numpy())
        if (batch_start // args.batch_size + 1) % 20 == 0:
            print(f"[embed] {min(batch_start + args.batch_size, len(frame))}/{len(frame)}", flush=True)
    x = np.concatenate(embeddings).astype(np.float32)
    np.save(output / "embeddings.npy", x)
    meta = {
        **load_meta,
        "track": args.track,
        "embedding_dim": int(x.shape[1]),
        "model_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "frozen_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "trainable_encoder_parameter_count": 0,
        "ridge_trainable_parameter_count": int(x.shape[1] + 1),
        "embedding_seconds": time.time() - started,
        "beat_count_min_median_max": [
            int(np.min(beat_counts)), float(np.median(beat_counts)), int(np.max(beat_counts))
        ],
        "preprocessing": {
            "window": "same frozen non-overlapping 10-second window",
            "beat_len": 128,
            "max_beats": 25,
            "zscore": "per beat, per channel",
            "padding": "zero padding with boolean valid-beat mask",
            "ppg_only_segmentation": "PPG peaks only; ECG array is not used by make_beats path",
            "ecg_ppg_direct_segmentation": "real ECG R-like peaks; real ECG+PPG fusion; input_mode=0",
            "ecg_only_segmentation": "ECG R-like peaks only",
            "aggregation": "masked phase+beat mean for unimodal tracks; reliability-gated cls for direct fusion",
        },
    }
    return x, meta


def fit_predict(x, y, train, val, test):
    scaler = StandardScaler().fit(x[train])
    train_x = scaler.transform(x[train])
    val_x = scaler.transform(x[val])
    test_x = scaler.transform(x[test])
    best_alpha, best_mae = None, float("inf")
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha).fit(train_x, y[train])
        score = mean_absolute_error(y[val], model.predict(val_x))
        if score < best_mae:
            best_alpha, best_mae = alpha, score
    final = Ridge(alpha=best_alpha).fit(train_x, y[train])
    return final.predict(test_x), float(best_alpha), float(best_mae)


def evaluate(x: np.ndarray, frame: pd.DataFrame, output: Path) -> dict:
    y = frame.pat_median_ms.to_numpy(float)
    predictions, fold_rows = [], []
    for test_fold in range(5):
        val_fold = (test_fold + 1) % 5
        fold = frame.test_fold.to_numpy(int)
        train = np.flatnonzero((fold != test_fold) & (fold != val_fold))
        val = np.flatnonzero(fold == val_fold)
        test = np.flatnonzero(fold == test_fold)
        group_sets = [set(frame.iloc[idx].group_id.astype(str)) for idx in (train, val, test)]
        if group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2]:
            raise RuntimeError("subject/case leakage")
        pred, alpha, val_mae = fit_predict(x, y, train, val, test)
        cols = [c for c in ("sample_id", "group_id", "record_id", "stratum") if c in frame]
        part = frame.iloc[test][cols].copy()
        part["fold"] = test_fold
        part["val_fold"] = val_fold
        part["y_true"] = y[test]
        part["y_pred"] = pred
        part["error_ms"] = pred - y[test]
        predictions.append(part)
        row = {"fold": test_fold, "val_fold": val_fold, "alpha": alpha, "val_mae_ms": val_mae}
        row.update(metrics(y[test], pred))
        row.update(macro_metrics(part))
        fold_rows.append(row)
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(output / "window_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)
    group_rows = []
    for group, part in pred.groupby("group_id"):
        row = {
            "group_id": group,
            "valid_window_count": int(len(part)),
            "pat_q05_ms": float(part.y_true.quantile(0.05)),
            "pat_median_ms": float(part.y_true.median()),
            "pat_q95_ms": float(part.y_true.quantile(0.95)),
        }
        row.update(metrics(part.y_true.to_numpy(), part.y_pred.to_numpy()))
        group_rows.append(row)
    pd.DataFrame(group_rows).to_csv(output / "group_metrics.csv", index=False)
    stratum_rows = []
    for stratum, part in pred.groupby("stratum", dropna=False):
        row = {"stratum": str(stratum)}
        row.update(metrics(part.y_true.to_numpy(), part.y_pred.to_numpy()))
        stratum_rows.append(row)
    pd.DataFrame(stratum_rows).to_csv(output / "stratum_metrics.csv", index=False)
    keys = ("mae_ms", "rmse_ms", "bias_ms", "sd_error_ms",
            "pearson_r", "spearman_rho", "r2")
    summary = {
        "fold_metrics": fold_rows,
        "fold_mean_std": {
            k: {"mean": float(np.nanmean([r[k] for r in fold_rows])),
                "std": float(np.nanstd([r[k] for r in fold_rows], ddof=1))}
            for k in keys
        },
        "pooled": metrics(pred.y_true.to_numpy(), pred.y_pred.to_numpy()),
        "subject_or_case_macro": macro_metrics(pred),
        "strata": stratum_rows,
    }
    return summary


def plots(output: Path) -> None:
    pred = pd.read_csv(output / "window_predictions.csv")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(pred.y_true, pred.y_pred, s=6, alpha=0.25)
    lo, hi = pred[["y_true", "y_pred"]].min().min(), pred[["y_true", "y_pred"]].max().max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set(xlabel="Derived PAT true (ms)", ylabel="Predicted (ms)")
    fig.tight_layout()
    fig.savefig(output / "scatter.png", dpi=180)
    plt.close(fig)

    mean = (pred.y_true + pred.y_pred) / 2
    err = pred.y_pred - pred.y_true
    bias, sd = err.mean(), err.std(ddof=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(mean, err, s=6, alpha=0.25)
    for y, style in ((bias, "-"), (bias + 1.96 * sd, "--"), (bias - 1.96 * sd, "--")):
        ax.axhline(y, color="k", ls=style, lw=1)
    ax.set(xlabel="Mean of true and predicted (ms)", ylabel="Prediction error (ms)")
    fig.tight_layout()
    fig.savefig(output / "bland_altman.png", dpi=180)
    plt.close(fig)

    strata = pd.read_csv(output / "stratum_metrics.csv")
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(strata)), 4))
    ax.bar(strata.stratum.astype(str), strata.mae_ms)
    ax.tick_params(axis="x", rotation=45)
    ax.set(ylabel="MAE (ms)", xlabel="Frozen stratum")
    fig.tight_layout()
    fig.savefig(output / "stratum_mae.png", dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=("ptt_ppg", "mendeley", "uq", "ergobp"))
    p.add_argument("--track", required=True, choices=TRACKS)
    p.add_argument("--frozen-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-module", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-windows-per-group", type=int, default=0)
    args = p.parse_args()
    for value in vars(args).values():
        if FORBIDDEN in str(value):
            raise RuntimeError("禁止访问旧服务器路径")
    seed_all(args.seed)
    root = Path(args.frozen_root)
    frame = pd.read_csv(
        root / "folds_v2_fixed_polarity" / args.dataset / "fold_manifest.csv",
        dtype={"group_id": str},
    )
    archive = np.load(root / "data_v2_fixed_polarity" / f"{args.dataset}_pat_windows.npz")
    index = {v: i for i, v in enumerate(archive["sample_id"].astype(str))}
    positions = np.asarray([index[v] for v in frame.sample_id.astype(str)])
    if args.max_windows_per_group:
        keep = (frame.groupby("group_id").cumcount() < args.max_windows_per_group).to_numpy()
        frame, positions = frame.loc[keep].reset_index(drop=True), positions[keep]
    output = Path(args.output_root) / args.dataset / args.track
    output.mkdir(parents=True, exist_ok=True)
    run_started = time.time()
    x, model_meta = embed(args, frame, positions, archive, output)
    summary = evaluate(x, frame, output)
    plots(output)
    payload = {
        "dataset": args.dataset,
        "track": args.track,
        "leaderboard": {
            "ppg_only": "main_fair_PPG_only_to_PAT",
            "ecg_ppg_direct": "separate_ECG_PPG_direct_PAT",
            "ecg_only": "diagnostic_ECG_only_to_PAT",
        }[args.track],
        "label_boundary": "derived PAT = PPG foot - ECG R; not official label; not true PTT",
        "seed": args.seed,
        "alpha_candidates": ALPHAS,
        "sample_count": len(frame),
        "group_count": int(frame.group_id.nunique()),
        "model_metadata": model_meta,
        "total_runtime_seconds": time.time() - run_started,
        "evaluation": summary,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
