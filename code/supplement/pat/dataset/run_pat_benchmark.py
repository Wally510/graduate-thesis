from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


OLD_CODE = Path("/bingding/301/BP/Code")
BASE_PATH = OLD_CODE / "benchmark_butppg_foundation_models.py"
PAP_PATH = (
    OLD_CODE
    / "downstream_benchmarks/papagei_sigma_frozen_bestval_20260615/"
    "benchmark_papagei_sigma_frozen_bestval.py"
)
OFFICIAL_DIR = OLD_CODE / "downstream_benchmarks/official_preprocess_20260709"
for path in [OLD_CODE, PAP_PATH.parent, OFFICIAL_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=float(np.nanmedian(x)))
    sd = float(np.std(x))
    return ((x - float(np.mean(x))) / max(sd, 1e-6)).astype(np.float32)


def resample_len(x: np.ndarray, length: int) -> np.ndarray:
    if len(x) == length:
        return np.asarray(x, dtype=np.float32)
    return signal.resample(np.asarray(x, dtype=np.float64), length).astype(np.float32)


def unified_windows(records: list[SimpleNamespace], model: str) -> tuple[np.ndarray, str]:
    if model == "csfm":
        from official_preprocess_adapter import prepare_official_windows

        args = SimpleNamespace(window_sec=10.0, csfm_channel_ids="1,12")
        return (
            prepare_official_windows(records, model, args, dataset_key="pat"),
            "CSFM own invariant preprocessing; a no-official CSFM tensor is not valid",
        )
    target = 1250 if model in {"anyppg", "papagei_s", "papagei_p"} else 500
    windows = np.stack([zscore(resample_len(record.ppg, target))[None] for record in records])
    return windows.astype(np.float32), "finite interpolation + length alignment + per-window z-score; no model filter/statistics"


def official_windows(records: list[SimpleNamespace], model: str) -> tuple[np.ndarray, str]:
    from official_preprocess_adapter import prepare_official_windows

    args = SimpleNamespace(window_sec=10.0, csfm_channel_ids="1,12")
    return (
        prepare_official_windows(records, model, args, dataset_key="pat"),
        "official_preprocess_adapter.py model-specific preprocessing",
    )


def encoder_args(embed_batch_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        embed_batch_size=embed_batch_size,
        log_every_batches=20,
        anyppg_module_path="",
        anyppg_ckpt_path="",
        pulseppg_module_path="",
        pulseppg_ckpt_path="",
        csfm_loader_path="",
        csfm_project_root="",
        csfm_ckpt_path="",
        csfm_variant="Base",
        csfm_channel_ids="1,12",
    )


def extract_embeddings(
    windows: np.ndarray, model: str, device_name: str, embed_batch_size: int
) -> tuple[np.ndarray, dict]:
    import torch
    from torch.utils.data import DataLoader

    base = import_path("pat_base_models", BASE_PATH)
    device = torch.device(device_name)
    args = encoder_args(embed_batch_size)
    if model in {"anyppg", "pulseppg", "csfm"}:
        encoder = base.build_encoder(model, args, device)
    else:
        pap = import_path("pat_pap_sigma_models", PAP_PATH)
        encoder = pap.build_encoder(model, input_len=int(windows.shape[-1]), device=device)
    encoder.eval()
    total = int(sum(p.numel() for p in encoder.parameters()))
    trainable = int(sum(p.numel() for p in encoder.parameters() if p.requires_grad))
    outputs = []
    started = time.time()
    loader = DataLoader(
        torch.from_numpy(windows).float(),
        batch_size=embed_batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, 1):
            out = encoder(batch.to(device))
            if out.dim() > 2:
                out = out.mean(dim=tuple(range(2, out.dim())))
            outputs.append(torch.nan_to_num(out).cpu().float().numpy())
            if batch_index % 20 == 0:
                print(f"[{model}] embedding {batch_index}/{len(loader)}", flush=True)
    embedding = np.concatenate(outputs).astype(np.float32)
    elapsed = time.time() - started
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embedding, {
        "parameter_count": total,
        "trainable_parameter_count_during_embedding": trainable,
        "embedding_dim": int(embedding.shape[1]),
        "embedding_seconds": elapsed,
    }


def safe_corr(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    if len(x) < 2 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return float("nan")
    return float(stats.pearsonr(x, y).statistic if kind == "pearson" else stats.spearmanr(x, y).statistic)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    error = pred - y
    return {
        "n": int(len(y)),
        "mae_ms": float(mean_absolute_error(y, pred)),
        "rmse_ms": float(mean_squared_error(y, pred) ** 0.5),
        "bias_ms": float(np.mean(error)),
        "sd_error_ms": float(np.std(error, ddof=1)) if len(error) > 1 else float("nan"),
        "pearson_r": safe_corr(y, pred, "pearson"),
        "spearman_rho": safe_corr(y, pred, "spearman"),
        "r2": float(r2_score(y, pred)) if len(y) > 1 else float("nan"),
    }


def macro_metrics(frame: pd.DataFrame) -> dict:
    values = []
    for _, part in frame.groupby("group_id"):
        if len(part) >= 2:
            values.append(metrics(part.y_true.to_numpy(), part.y_pred.to_numpy()))
    keys = ["mae_ms", "rmse_ms", "bias_ms", "sd_error_ms", "pearson_r", "spearman_rho", "r2"]
    result = {
        f"{key}_macro": float(np.nanmean([row[key] for row in values]))
        for key in keys
    }
    result["macro_group_count"] = len(values)
    return result


def hand_features(windows: np.ndarray, fs_values: np.ndarray) -> np.ndarray:
    rows = []
    for x, fs in zip(windows, fs_values):
        x = zscore(x)
        dx = np.diff(x)
        peaks, _ = signal.find_peaks(x, distance=max(1, int(0.3 * fs)), prominence=0.25)
        freqs, power = signal.welch(x, fs=float(fs), nperseg=min(len(x), 256))
        band = (freqs >= 0.4) & (freqs <= 4.0)
        peak_hz = float(freqs[band][np.argmax(power[band])]) if band.any() else 0.0
        rows.append(
            [
                np.std(x),
                stats.skew(x),
                stats.kurtosis(x),
                np.std(dx),
                np.percentile(x, 95) - np.percentile(x, 5),
                len(peaks) / 10.0,
                peak_hz,
            ]
        )
    return np.nan_to_num(np.asarray(rows, dtype=np.float32))


def ridge_predict(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, float]:
    best_alpha, best_mae = None, float("inf")
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        scaler = StandardScaler().fit(x[train])
        model = Ridge(alpha=alpha).fit(scaler.transform(x[train]), y[train])
        mae = mean_absolute_error(y[val], model.predict(scaler.transform(x[val])))
        if mae < best_mae:
            best_alpha, best_mae = alpha, mae
    fit = np.concatenate([train, val])
    scaler = StandardScaler().fit(x[fit])
    model = Ridge(alpha=float(best_alpha)).fit(scaler.transform(x[fit]), y[fit])
    return model.predict(scaler.transform(x[test])), float(best_alpha)


def run_method(
    method: str,
    x: np.ndarray | None,
    frame: pd.DataFrame,
    output_dir: Path,
    direct_secondary: bool = False,
) -> dict:
    y = frame.pat_median_ms.to_numpy(dtype=np.float64)
    predictions = []
    fold_metrics = []
    for test_fold in range(5):
        val_fold = (test_fold + 1) % 5
        test = np.flatnonzero(frame.test_fold.to_numpy() == test_fold)
        val = np.flatnonzero(frame.test_fold.to_numpy() == val_fold)
        train = np.flatnonzero(~frame.test_fold.isin([test_fold, val_fold]).to_numpy())
        train_groups = set(frame.iloc[train].group_id)
        val_groups = set(frame.iloc[val].group_id)
        test_groups = set(frame.iloc[test].group_id)
        assert not (train_groups & val_groups or train_groups & test_groups or val_groups & test_groups)
        alpha = None
        if method == "train_mean":
            pred = np.full(len(test), np.mean(y[np.concatenate([train, val])]))
        elif method == "train_median":
            pred = np.full(len(test), np.median(y[np.concatenate([train, val])]))
        elif direct_secondary:
            pred = frame.iloc[test].pat_secondary_median_ms.to_numpy(dtype=float)
        else:
            assert x is not None
            pred, alpha = ridge_predict(x, y, train, val, test)
        part = frame.iloc[test][["sample_id", "group_id", "record_id", "stratum"]].copy()
        part["fold"] = test_fold
        part["y_true"] = y[test]
        part["y_pred"] = pred
        part["error_ms"] = pred - y[test]
        predictions.append(part)
        fold_result = {"fold": test_fold, "alpha": alpha}
        fold_result.update(metrics(y[test], pred))
        fold_result.update(macro_metrics(part))
        fold_metrics.append(fold_result)
    pred_frame = pd.concat(predictions).sort_index()
    pred_frame.to_csv(output_dir / f"{method}_predictions.csv", index=False)
    summary = {
        "method": method,
        "fold_metrics": fold_metrics,
        "pooled": metrics(pred_frame.y_true.to_numpy(), pred_frame.y_pred.to_numpy()),
        "subject_or_case_macro": macro_metrics(pred_frame),
        "fold_mean_std": {
            key: {
                "mean": float(np.nanmean([row[key] for row in fold_metrics])),
                "std": float(np.nanstd([row[key] for row in fold_metrics], ddof=1)),
            }
            for key in ["mae_ms", "rmse_ms", "bias_ms", "sd_error_ms", "pearson_r", "spearman_rho", "r2"]
        },
    }
    (output_dir / f"{method}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def load_data(args) -> tuple[pd.DataFrame, list[SimpleNamespace]]:
    manifest = pd.read_csv(
        Path(args.fold_root) / args.dataset / "fold_manifest.csv",
        dtype={"group_id": str},
    )
    archive = np.load(Path(args.data_root) / f"{args.dataset}_pat_windows.npz")
    # NPZ is compressed: materialize each member once. Indexing archive["ppg"]
    # inside the record loop would decompress the full member for every sample.
    sample_ids = archive["sample_id"].astype(str)
    ppg_all = archive["ppg"]
    ecg_all = archive["ecg"]
    fs_all = archive["fs"]
    index = {sample_id: i for i, sample_id in enumerate(sample_ids)}
    positions = np.asarray([index[s] for s in manifest.sample_id.astype(str)])
    if args.max_windows_per_group:
        keep = (
            manifest.groupby("group_id", sort=False).cumcount()
            < args.max_windows_per_group
        ).to_numpy()
        manifest = manifest.loc[keep].reset_index(drop=True)
        positions = positions[keep]
    records = [
        SimpleNamespace(
            record_id=row.sample_id,
            ppg=ppg_all[pos],
            ecg=ecg_all[pos],
            ppg_fs=float(fs_all[pos]),
            ecg_fs=float(fs_all[pos]),
        )
        for pos, row in zip(positions, manifest.itertuples())
    ]
    return manifest, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True, choices=["anyppg", "pulseppg", "csfm", "papagei_s", "papagei_p", "sigmappg"])
    parser.add_argument("--track", required=True, choices=["official", "unified_minimal"])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--fold-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embed-batch-size", type=int, default=128)
    parser.add_argument("--max-windows-per-group", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_root) / args.dataset / args.track / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, records = load_data(args)
    if args.track == "official":
        windows, preprocessing = official_windows(records, args.model)
    else:
        windows, preprocessing = unified_windows(records, args.model)
    embedding, model_meta = extract_embeddings(
        windows, args.model, args.device, args.embed_batch_size
    )
    np.save(output_dir / "embeddings.npy", embedding)
    summaries = {
        "frozen_encoder_ridge": run_method("frozen_encoder_ridge", embedding, frame, output_dir),
        "train_mean": run_method("train_mean", None, frame, output_dir),
        "train_median": run_method("train_median", None, frame, output_dir),
    }
    ppg = np.stack([record.ppg for record in records])
    fs = np.asarray([record.ppg_fs for record in records])
    summaries["handcrafted_ppg_ridge"] = run_method(
        "handcrafted_ppg_ridge", hand_features(ppg, fs), frame, output_dir
    )
    if args.model == "csfm":
        summaries["direct_secondary_foot"] = run_method(
            "direct_secondary_foot", None, frame, output_dir, direct_secondary=True
        )
    payload = {
        "dataset": args.dataset,
        "model": args.model,
        "track": args.track,
        "track_note": preprocessing,
        "leaderboard": "ECG+PPG_direct_PAT" if args.model == "csfm" else "PPG_only_to_PAT",
        "label_boundary": "derived PAT label; not official label; not true PTT",
        "sample_count": len(frame),
        "group_count": int(frame.group_id.nunique()),
        "model_metadata": model_meta,
        "methods": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
