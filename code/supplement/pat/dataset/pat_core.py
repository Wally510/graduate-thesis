from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import signal, stats


@dataclass(frozen=True)
class PATProtocol:
    version: str = "pat_tangent_fixed_polarity_v2_20260728"
    ecg_band_hz: tuple[float, float] = (5.0, 35.0)
    ppg_band_hz: tuple[float, float] = (0.5, 8.0)
    search_ms: tuple[float, float] = (40.0, 700.0)
    frozen_accept_ms: tuple[float, float] = (60.0, 600.0)
    min_valid_beats: int = 5
    window_sec: float = 10.0
    r_refractory_sec: float = 0.30


def finite_interp(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    ok = np.isfinite(x)
    if ok.all():
        return x
    if not ok.any():
        return np.zeros_like(x)
    idx = np.arange(len(x))
    out = x.copy()
    out[~ok] = np.interp(idx[~ok], idx[ok], x[ok])
    return out


def zero_phase_bandpass(
    x: np.ndarray, fs: float, low: float, high: float, order: int = 3
) -> np.ndarray:
    x = finite_interp(x)
    high = min(float(high), float(fs) * 0.45)
    if len(x) < max(30, order * 12) or not 0 < low < high:
        return x - np.median(x)
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def robust_scale(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    mad = np.median(np.abs(x - np.median(x)))
    return max(1.4826 * mad, float(np.std(x)) * 0.25, 1e-8)


def detect_r_peaks(ecg: np.ndarray, fs: float, protocol: PATProtocol) -> tuple[np.ndarray, dict]:
    filtered = zero_phase_bandpass(
        ecg, fs, protocol.ecg_band_hz[0], protocol.ecg_band_hz[1]
    )
    envelope = np.abs(filtered)
    prominence = max(robust_scale(envelope) * 2.0, np.std(envelope) * 0.35)
    peaks, props = signal.find_peaks(
        envelope,
        distance=max(1, int(round(protocol.r_refractory_sec * fs))),
        prominence=prominence,
    )
    if len(peaks) >= 2:
        rr = np.diff(peaks) / fs
        plausible = (rr >= 0.28) & (rr <= 2.0)
        rr_plausible_fraction = float(plausible.mean())
        median_hr = float(60.0 / np.median(rr))
    else:
        rr_plausible_fraction = 0.0
        median_hr = float("nan")
    return peaks.astype(np.int64), {
        "r_count": int(len(peaks)),
        "r_prominence_median": float(np.median(props.get("prominences", [0.0]))),
        "rr_plausible_fraction": rr_plausible_fraction,
        "median_hr_bpm": median_hr,
        "detector": "zero_phase_bandpass_abs_prominence_v1",
    }


def orient_ppg(
    ppg_filtered: np.ndarray, fixed_polarity: int | None = None
) -> tuple[np.ndarray, int, str]:
    if fixed_polarity in {-1, 1}:
        return ppg_filtered * fixed_polarity, int(fixed_polarity), "dataset_channel_fixed"
    skew = float(stats.skew(ppg_filtered, nan_policy="omit"))
    polarity = 1 if not np.isfinite(skew) or skew >= 0 else -1
    return ppg_filtered * polarity, polarity, "window_skew_legacy_audit_only"


def tangent_foot(
    ppg_oriented: np.ndarray,
    r_idx: int,
    next_r_idx: int | None,
    fs: float,
    protocol: PATProtocol,
) -> tuple[float | None, dict]:
    lo = r_idx + int(round(protocol.search_ms[0] * fs / 1000.0))
    hi = r_idx + int(round(protocol.search_ms[1] * fs / 1000.0))
    if next_r_idx is not None:
        hi = min(hi, next_r_idx - int(round(0.04 * fs)))
    hi = min(hi, len(ppg_oriented) - 2)
    if hi - lo < max(5, int(round(0.08 * fs))):
        return None, {"reason": "short_search"}

    y = ppg_oriented
    derivative = np.gradient(y) * fs
    slope_idx = lo + int(np.argmax(derivative[lo:hi]))
    slope = float(derivative[slope_idx])
    slope_quality = slope / robust_scale(derivative[lo:hi])
    if not np.isfinite(slope) or slope <= 0 or slope_quality < 1.5:
        return None, {"reason": "weak_upstroke", "slope_quality": float(slope_quality)}

    baseline_lo = max(lo, slope_idx - int(round(0.30 * fs)))
    baseline_hi = slope_idx - max(1, int(round(0.02 * fs)))
    if baseline_hi <= baseline_lo:
        return None, {"reason": "short_baseline"}
    baseline_idx = baseline_lo + int(np.argmin(y[baseline_lo:baseline_hi]))
    baseline = float(y[baseline_idx])
    foot_float = slope_idx + (baseline - float(y[slope_idx])) * fs / slope
    foot_float = float(np.clip(foot_float, baseline_idx, slope_idx))
    pat_ms = (foot_float - r_idx) * 1000.0 / fs
    return pat_ms, {
        "reason": "ok",
        "foot_index": foot_float,
        "slope_index": int(slope_idx),
        "baseline_index": int(baseline_idx),
        "slope_quality": float(slope_quality),
    }


def second_derivative_foot(
    ppg_oriented: np.ndarray,
    r_idx: int,
    next_r_idx: int | None,
    fs: float,
    protocol: PATProtocol,
) -> float | None:
    lo = r_idx + int(round(protocol.search_ms[0] * fs / 1000.0))
    hi = r_idx + int(round(protocol.search_ms[1] * fs / 1000.0))
    if next_r_idx is not None:
        hi = min(hi, next_r_idx - int(round(0.04 * fs)))
    hi = min(hi, len(ppg_oriented) - 2)
    if hi - lo < 5:
        return None
    second = np.gradient(np.gradient(ppg_oriented)) * fs * fs
    idx = lo + int(np.argmax(second[lo:hi]))
    return float((idx - r_idx) * 1000.0 / fs)


def build_pat_window(
    ecg: np.ndarray,
    ppg: np.ndarray,
    fs: float,
    protocol: PATProtocol,
    r_peaks: Iterable[int] | None = None,
    r_source: str = "detected",
    apply_frozen_range: bool = True,
    ppg_polarity: int | None = None,
) -> dict:
    ecg = finite_interp(ecg)
    ppg = finite_interp(ppg)
    ppg_filtered = zero_phase_bandpass(
        ppg, fs, protocol.ppg_band_hz[0], protocol.ppg_band_hz[1]
    )
    ppg_oriented, polarity, polarity_source = orient_ppg(
        ppg_filtered, fixed_polarity=ppg_polarity
    )
    if r_peaks is None:
        r_peaks_arr, r_qc = detect_r_peaks(ecg, fs, protocol)
    else:
        r_peaks_arr = np.asarray(list(r_peaks), dtype=np.int64)
        r_qc = {
            "r_count": int(len(r_peaks_arr)),
            "r_prominence_median": float("nan"),
            "rr_plausible_fraction": float("nan"),
            "median_hr_bpm": float("nan"),
            "detector": r_source,
        }

    pats: list[float] = []
    pats_secondary: list[float] = []
    reject_reasons: dict[str, int] = {}
    slope_quality: list[float] = []
    for i, r_idx in enumerate(r_peaks_arr):
        next_r = int(r_peaks_arr[i + 1]) if i + 1 < len(r_peaks_arr) else None
        pat, detail = tangent_foot(ppg_oriented, int(r_idx), next_r, fs, protocol)
        if pat is None:
            reason = str(detail["reason"])
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
            continue
        if apply_frozen_range and not (
            protocol.frozen_accept_ms[0] <= pat <= protocol.frozen_accept_ms[1]
        ):
            reject_reasons["outside_frozen_range"] = (
                reject_reasons.get("outside_frozen_range", 0) + 1
            )
            continue
        pats.append(float(pat))
        slope_quality.append(float(detail.get("slope_quality", np.nan)))
        secondary = second_derivative_foot(
            ppg_oriented, int(r_idx), next_r, fs, protocol
        )
        if secondary is not None:
            pats_secondary.append(float(secondary))

    values = np.asarray(pats, dtype=np.float64)
    secondary_values = np.asarray(pats_secondary, dtype=np.float64)
    valid = len(values) >= protocol.min_valid_beats
    result = {
        "valid": bool(valid),
        "reject_reason": "" if valid else "fewer_than_min_valid_beats",
        "pat_median_ms": float(np.median(values)) if len(values) else float("nan"),
        "pat_mean_ms": float(np.mean(values)) if len(values) else float("nan"),
        "pat_iqr_ms": (
            float(np.percentile(values, 75) - np.percentile(values, 25))
            if len(values)
            else float("nan")
        ),
        "pat_secondary_median_ms": (
            float(np.median(secondary_values)) if len(secondary_values) else float("nan")
        ),
        "valid_beat_count": int(len(values)),
        "candidate_r_count": int(len(r_peaks_arr)),
        "valid_fraction": float(len(values) / max(1, len(r_peaks_arr))),
        "slope_quality_median": (
            float(np.nanmedian(slope_quality)) if slope_quality else float("nan")
        ),
        "ppg_polarity": int(polarity),
        "ppg_polarity_source": polarity_source,
        "reject_reasons": reject_reasons,
        "r_qc": r_qc,
        "label_kind": "derived_PAT_not_official_not_true_PTT",
        "protocol_version": protocol.version,
    }
    return result
