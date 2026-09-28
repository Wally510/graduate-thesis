# -*- coding: utf-8 -*-
"""
Permissive ECG+PPG beat extraction for the orthogonal Transformer variant.

The original preprocessor skips records when R-peak detection is uncertain.
This wrapper keeps the same public helpers but replaces
extract_pretrain_npz_arrays with a no-drop version:
  1) try lenient R-peak based beat boundaries;
  2) if no usable R peak exists, fall back to evenly spaced pseudo-beats.

The pat target in this permissive path is ECG R-peak to PPG maximum upslope,
not ECG R-peak to PPG foot.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy import signal

import prepare_ecg_ppg_pretrain_npz as base
from prepare_ecg_ppg_pretrain_npz import *  # re-export loaders/savers/CLI helpers


PSEUDO_BEAT_SEC = 0.5


def _estimate_pat_to_max_slope(
    ppg_f: np.ndarray,
    r_peak: int,
    fs: int,
    min_delay_sec: float = 0.08,
    max_delay_sec: float = 0.55,
) -> float:
    """Return ECG R-peak to PPG maximum upslope delay in seconds."""
    lo = int(r_peak) + int(round(min_delay_sec * fs))
    hi = int(r_peak) + int(round(max_delay_sec * fs))
    if lo < 0 or hi >= len(ppg_f) or hi <= lo + 2:
        return float("nan")

    win = np.asarray(ppg_f[lo:hi], dtype=np.float32)
    if not np.isfinite(win).all() or float(np.std(win)) < 1e-6:
        return float("nan")

    slope = np.diff(win, prepend=win[0])
    if not np.isfinite(slope).all():
        return float("nan")
    max_slope_idx = lo + int(np.argmax(slope))
    return (max_slope_idx - int(r_peak)) / float(max(fs, 1))


def _ensure_2d_signal(sig_tc: np.ndarray, min_cols: int) -> np.ndarray:
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


def _detect_r_peaks_lenient(ecg_f: np.ndarray, fs: int) -> np.ndarray:
    ecg = np.asarray(ecg_f, dtype=np.float32)
    if len(ecg) < 3:
        return np.array([], dtype=np.int64)

    der = np.diff(ecg, prepend=ecg[0])
    energy = der * der
    win = max(1, int(0.10 * max(fs, 1)))
    mwa = np.convolve(energy, np.ones(win, dtype=np.float32) / win, mode="same")

    finite = np.isfinite(mwa)
    if finite.sum() < 3:
        return np.array([], dtype=np.int64)
    med = float(np.median(mwa[finite]))
    mad = float(np.median(np.abs(mwa[finite] - med))) + 1e-8
    thr = med + 3.0 * 1.4826 * mad

    peaks, _ = signal.find_peaks(mwa, height=thr, distance=max(1, int(0.18 * max(fs, 1))))
    if len(peaks) == 0:
        peaks, _ = signal.find_peaks(ecg, distance=max(1, int(0.18 * max(fs, 1))))
    if len(peaks) == 0:
        return np.array([], dtype=np.int64)

    refine = []
    w = max(1, int(0.05 * max(fs, 1)))
    for p in peaks:
        left = max(0, int(p) - w)
        right = min(len(ecg), int(p) + w + 1)
        if right > left:
            refine.append(left + int(np.argmax(ecg[left:right])))
    return np.array(sorted(set(refine)), dtype=np.int64)


def _arrays_from_segments(
    processed: np.ndarray,
    ppg_f: np.ndarray,
    fs: int,
    beat_len: int,
    starts: Sequence[int],
    ends: Sequence[int],
    centers: Sequence[int],
    r_peaks_for_hrv: Optional[np.ndarray] = None,
) -> Optional[Dict[str, np.ndarray]]:
    beats = []
    time_sec = []
    pats = []
    morphs = []
    first_time = None
    n = int(processed.shape[0])

    for start, end, center in zip(starts, ends, centers):
        start = int(np.clip(start, 0, max(n - 1, 0)))
        end = int(np.clip(end, start + 1, n))
        center = int(np.clip(center, start, max(end - 1, start)))
        if end <= start:
            continue

        pat = _estimate_pat_to_max_slope(ppg_f, r_peak=center, fs=fs)
        _, foot, peak = base.estimate_pat_and_foot(ppg_f, r_peak=center, fs=fs)
        beat_ch = [base.resample_segment(processed[start:end, ch], beat_len) for ch in range(processed.shape[1])]
        beats.append(np.stack(beat_ch, axis=0))

        t = center / float(max(fs, 1))
        if first_time is None:
            first_time = t
        time_sec.append(t - first_time)
        pats.append(pat)
        morphs.append(base.estimate_morphology(ppg_f, start=start, end=end, foot=foot, peak=peak, fs=fs))

    if not beats:
        return None

    if r_peaks_for_hrv is not None and len(r_peaks_for_hrv) >= 3:
        hrv = base.compute_hrv(r_peaks_for_hrv, fs=fs)
    else:
        hrv = np.full(base.HRV_DIM, np.nan, dtype=np.float32)

    return {
        "beats": np.stack(beats, axis=0).astype(np.float32),
        "time_sec": np.asarray(time_sec, dtype=np.float32),
        "pat": np.asarray(pats, dtype=np.float32),
        "morph": np.stack(morphs, axis=0).astype(np.float32),
        "hrv": hrv,
    }


def _arrays_from_r_peaks(
    processed: np.ndarray,
    ppg_f: np.ndarray,
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
        rr_default = int(round(PSEUDO_BEAT_SEC * max(fs, 1)))
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

    return _arrays_from_segments(processed, ppg_f, fs, beat_len, starts, ends, centers, r_peaks)


def _arrays_from_even_windows(
    processed: np.ndarray,
    ppg_f: np.ndarray,
    fs: int,
    beat_len: int,
) -> Dict[str, np.ndarray]:
    n = int(processed.shape[0])
    duration = n / float(max(fs, 1))
    count = max(1, int(np.ceil(duration / PSEUDO_BEAT_SEC)))
    edges = np.linspace(0, n, num=count + 1)
    starts = np.floor(edges[:-1]).astype(np.int64)
    ends = np.ceil(edges[1:]).astype(np.int64)
    centers = ((starts + ends) // 2).astype(np.int64)
    arrays = _arrays_from_segments(processed, ppg_f, fs, beat_len, starts, ends, centers, None)
    if arrays is not None:
        return arrays

    zero = np.zeros((1, processed.shape[1], beat_len), dtype=np.float32)
    return {
        "beats": zero,
        "time_sec": np.zeros((1,), dtype=np.float32),
        "pat": np.full((1,), np.nan, dtype=np.float32),
        "morph": np.full((1, base.MORPH_DIM), np.nan, dtype=np.float32),
        "hrv": np.full(base.HRV_DIM, np.nan, dtype=np.float32),
    }


def extract_pretrain_npz_arrays(
    sig_tc: np.ndarray,
    fs: int,
    input_cols: Sequence[int],
    ecg_col: int,
    ppg_target_col: int,
    beat_len: int,
    min_beats: int = 6,
) -> Optional[Dict[str, np.ndarray]]:
    """Convert any loadable continuous signal into model-ready beat arrays."""
    del min_beats  # Kept for CLI compatibility; this permissive path does not drop by beat count.
    fs = int(fs) if int(fs) > 0 else 250
    input_cols = list(input_cols)
    min_cols = max(input_cols + [int(ecg_col), int(ppg_target_col)]) + 1
    sig = _ensure_2d_signal(sig_tc, min_cols=min_cols)

    processed = base.preprocess_channels(sig, fs=fs, ecg_col=ecg_col, input_cols=input_cols)
    ecg_idx = input_cols.index(ecg_col) if ecg_col in input_cols else 0
    ecg_f = processed[:, ecg_idx]

    if ppg_target_col in input_cols:
        ppg_f = processed[:, input_cols.index(ppg_target_col)]
    else:
        ppg_f = base.robust_zscore(base.safe_bandpass(sig[:, ppg_target_col], fs=fs, low=0.3, high=8.0, order=3))

    r_peaks = _detect_r_peaks_lenient(ecg_f, fs=fs)
    arrays = _arrays_from_r_peaks(processed, ppg_f, r_peaks, fs=fs, beat_len=beat_len)
    if arrays is not None:
        return arrays
    return _arrays_from_even_windows(processed, ppg_f, fs=fs, beat_len=beat_len)


def main(argv=None) -> None:
    base.extract_pretrain_npz_arrays = extract_pretrain_npz_arrays
    base.main(argv)


if __name__ == "__main__":
    main()
