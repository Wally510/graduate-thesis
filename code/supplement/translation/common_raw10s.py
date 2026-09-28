#!/usr/bin/env python3
"""Common four-dataset raw10s backend and Split/Compose (S-C) adapter."""

from __future__ import annotations

import bisect
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


SERVER_ROOT = Path("/data-ai/sl20200894")
WINDOW_SAMPLES = 2500
SAMPLING_RATE = 250
BEAT_LEN = 128
MAX_BEATS = 25
DETECTOR_THEORETICAL_MAX_BEATS = 50
DATASET_ORDER = ("pulsedb", "sysu", "tptcom", "uci")


def guard_server_path(path: str | Path, label: str) -> Path:
    resolved = Path(path).resolve()
    if SERVER_ROOT not in (resolved, *resolved.parents):
        raise ValueError(f"{label} outside server root: {resolved}")
    return resolved


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_indices(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def stable_hash(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def canonical_group_id(dataset: str, record_id: str) -> str:
    """Return the strongest identity available without reading target PPG.

    Complete PulseDB cache record IDs end in ``:<segment_index>`` even though
    all segments came from the same MAT file.  Removing only that final numeric
    suffix prevents windows from one MAT file crossing splits.  Other caches
    retain their record_id because their writer already stores one source
    record ID while start_sec identifies windows within it.
    """
    dataset = str(dataset).lower()
    record_id = str(record_id)
    if dataset == "pulsedb":
        record_id = re.sub(r":\d+$", "", record_id)
    return f"{dataset}:{record_id}"


def contiguous_boundaries_from_peaks(peaks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    peaks = np.asarray(peaks, dtype=np.int64)
    if (
        peaks.ndim != 1 or len(peaks) < 2
        or np.any(peaks[1:] <= peaks[:-1])
        or peaks[0] < 0 or peaks[-1] >= WINDOW_SAMPLES
    ):
        raise ValueError(f"invalid ECG peaks: {peaks}")
    mids = ((peaks[:-1] + peaks[1:]) // 2).astype(np.int64)
    starts = np.concatenate([np.array([0], np.int64), mids])
    ends = np.concatenate([mids, np.array([WINDOW_SAMPLES], np.int64)])
    if np.any(ends <= starts) or starts[0] != 0 or ends[-1] != WINDOW_SAMPLES:
        raise ValueError("ECG peak midpoint boundaries do not cover raw10s")
    return starts, ends


def resample_1d(values: np.ndarray, out_len: int = BEAT_LEN) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError(f"cannot resample shape={values.shape}")
    old = np.arange(len(values), dtype=np.float64) / float(len(values))
    new = np.arange(out_len, dtype=np.float64) / float(out_len)
    return np.interp(new, old, values).astype(np.float32)


class RawCache:
    def __init__(self, root: str | Path) -> None:
        self.root = guard_server_path(root, "raw cache")
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file() or not (self.root / "RUN_COMPLETE.txt").is_file():
            raise FileNotFoundError(f"incomplete raw10s cache: {self.root}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != "csfm_official_raw_continuous_window_pack_v1":
            raise ValueError(f"unsupported raw cache: {self.root}")
        if int(self.manifest.get("sampling_rate_hz", -1)) != SAMPLING_RATE:
            raise ValueError("raw cache is not 250 Hz")
        if int(self.manifest.get("window_samples", -1)) != WINDOW_SAMPLES:
            raise ValueError("raw cache is not 2500 samples")
        if str(self.manifest.get("raw_waveform_shape", "")) != "[N,2,2500]":
            raise ValueError("raw cache waveform declaration mismatch")
        if tuple(self.manifest.get("channels", [])) != ("ECG", "PPG"):
            raise ValueError("raw cache channel order is not ECG then PPG")
        self.shards = list(self.manifest.get("shards", []))
        if not self.shards:
            raise ValueError(f"raw cache has no shards: {self.root}")
        self.ends: list[int] = []
        total = 0
        for shard in self.shards:
            count = int(shard["n_samples"])
            if count <= 0:
                raise ValueError(f"bad shard count: {shard}")
            total += count
            self.ends.append(total)
        if total != int(self.manifest.get("total_samples", -1)):
            raise ValueError(f"raw cache total mismatch: {self.root}")
        self.total = total
        self._arrays: dict[int, dict[str, np.ndarray]] = {}

    def locate(self, raw_index: int) -> tuple[int, int]:
        shard_id = bisect.bisect_right(self.ends, int(raw_index))
        start = 0 if shard_id == 0 else self.ends[shard_id - 1]
        return shard_id, int(raw_index) - start

    def shard_arrays(self, shard_id: int, *, waveform: bool) -> dict[str, np.ndarray]:
        cached = self._arrays.get(shard_id)
        if cached is not None and (not waveform or "waveform" in cached):
            return cached
        directory = self.root / str(self.shards[shard_id]["dir"])
        arrays = dict(cached or {})
        arrays.update({
            "record_id": np.load(directory / "record_id.npy", mmap_mode="r", allow_pickle=False),
            "source_index": np.load(directory / "source_index.npy", mmap_mode="r", allow_pickle=False),
            "start_sec": np.load(directory / "start_sec.npy", mmap_mode="r", allow_pickle=False),
            "end_sec": np.load(directory / "end_sec.npy", mmap_mode="r", allow_pickle=False),
        })
        expected = int(self.shards[shard_id]["n_samples"])
        for name in ("record_id", "source_index", "start_sec", "end_sec"):
            if arrays[name].shape != (expected,):
                raise ValueError(f"bad {name} shape: {directory}")
        if waveform:
            arrays["waveform"] = np.load(directory / "waveform.npy", mmap_mode="r")
            if tuple(arrays["waveform"].shape[1:]) != (2, WINDOW_SAMPLES):
                raise ValueError(f"bad waveform shape: {directory}")
        self._arrays[shard_id] = arrays
        return arrays

    def item(self, raw_index: int) -> dict[str, Any]:
        shard_id, local = self.locate(raw_index)
        arrays = self.shard_arrays(shard_id, waveform=True)
        waveform = np.asarray(arrays["waveform"][local], dtype=np.float32).copy()
        if waveform.shape != (2, WINDOW_SAMPLES) or not np.isfinite(waveform).all():
            raise ValueError(f"invalid waveform raw_index={raw_index}")
        return {
            "waveform": waveform,
            "record_id": str(arrays["record_id"][local]),
            "source_index": int(arrays["source_index"][local]),
            "start_sec": float(arrays["start_sec"][local]),
            "end_sec": float(arrays["end_sec"][local]),
        }


class BeatCache:
    def __init__(self, root: str | Path, raw: RawCache) -> None:
        self.root = guard_server_path(root, "beat cache")
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file() or not (self.root / "RUN_COMPLETE.txt").is_file():
            raise FileNotFoundError(f"incomplete beat cache: {self.root}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != "csfm_raw10s_beat_manifest_v1":
            raise ValueError(f"unsupported beat cache: {self.root}")
        if self.manifest.get("raw10s_manifest_sha256") != sha256_file(raw.root / "manifest.json"):
            raise ValueError("beat cache was built from another raw cache")
        if int(self.manifest.get("total_samples", -1)) != raw.total:
            raise ValueError("raw/beat total mismatch")
        self.raw = raw
        self._arrays: dict[int, dict[str, np.ndarray]] = {}

    def shard_arrays(self, shard_id: int) -> dict[str, np.ndarray]:
        cached = self._arrays.get(shard_id)
        if cached is not None:
            return cached
        directory = self.root / str(self.raw.shards[shard_id]["dir"])
        arrays = {
            "offsets": np.load(directory / "beat_offsets.npy", mmap_mode="r"),
            "peaks": np.load(directory / "beat_peaks.npy", mmap_mode="r"),
            "counts": np.load(directory / "beat_counts.npy", mmap_mode="r"),
        }
        expected = int(self.raw.shards[shard_id]["n_samples"])
        if arrays["offsets"].shape != (expected + 1,) or arrays["counts"].shape != (expected,):
            raise ValueError(f"bad beat shard: {directory}")
        if int(arrays["offsets"][0]) != 0 or int(arrays["offsets"][-1]) != len(arrays["peaks"]):
            raise ValueError(f"bad beat flat offsets: {directory}")
        self._arrays[shard_id] = arrays
        return arrays

    def peaks_for(self, raw_index: int) -> np.ndarray:
        shard_id, local = self.raw.locate(raw_index)
        arrays = self.shard_arrays(shard_id)
        left, right = int(arrays["offsets"][local]), int(arrays["offsets"][local + 1])
        return np.asarray(arrays["peaks"][left:right], dtype=np.int64)


class CommonBackend:
    """Stable global eligible index shared by SplitEnc8, CSFM and baselines."""

    def __init__(self, index_path: str | Path) -> None:
        self.index_path = guard_server_path(index_path, "multi-cache index")
        self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        if self.index.get("format") != "csfm_raw10s_four_dataset_multicache_index_v1":
            raise ValueError("unsupported four-dataset multi-cache index")
        if int(self.index.get("sampling_rate_hz", -1)) != SAMPLING_RATE:
            raise ValueError("multi-cache index is not 250 Hz")
        if int(self.index.get("window_samples", -1)) != WINDOW_SAMPLES:
            raise ValueError("multi-cache index is not raw10s/2500")
        rows = list(self.index.get("dataset_rows", []))
        if tuple(self.index.get("dataset_order", [])) != DATASET_ORDER:
            raise ValueError("four-dataset order mismatch")
        if len(rows) != len(DATASET_ORDER):
            raise ValueError("four-dataset row count mismatch")
        self.entries: list[dict[str, Any]] = []
        self.ends: list[int] = []
        total = 0
        for dataset_id, row in enumerate(rows):
            if str(row.get("dataset", "")) != DATASET_ORDER[dataset_id]:
                raise ValueError("dataset row order mismatch")
            raw = RawCache(row["cache_root"])
            raw_manifest_path = raw.root / "manifest.json"
            if guard_server_path(row["raw_manifest"], "raw manifest") != raw_manifest_path:
                raise ValueError(f"raw manifest path mismatch: {row['dataset']}")
            if sha256_file(raw_manifest_path) != str(row["raw_manifest_sha256"]):
                raise ValueError(f"raw manifest hash mismatch: {row['dataset']}")
            eligible_path = guard_server_path(row["eligible_indices_path"], "eligible indices")
            eligible = np.load(eligible_path, mmap_mode="r")
            eligible = np.asarray(eligible, dtype=np.int64)
            if len(eligible) != int(row["eligible_samples"]):
                raise ValueError(f"eligible count mismatch: {row['dataset']}")
            if len(eligible) and (
                eligible[0] < 0 or eligible[-1] >= raw.total
                or np.any(eligible[1:] <= eligible[:-1])
            ):
                raise ValueError(f"eligible indices invalid: {row['dataset']}")
            if sha256_file(eligible_path) != str(row["eligible_indices_sha256_file"]):
                raise ValueError(f"eligible file hash mismatch: {row['dataset']}")
            if sha256_indices(eligible) != str(row["eligible_indices_sha256_int64"]):
                raise ValueError(f"eligible int64 hash mismatch: {row['dataset']}")
            beat = BeatCache(Path(row["beat_manifest"]).parent, raw)
            if guard_server_path(row["beat_manifest"], "beat manifest") != beat.root / "manifest.json":
                raise ValueError(f"beat manifest path mismatch: {row['dataset']}")
            if sha256_file(beat.root / "manifest.json") != str(row["beat_manifest_sha256"]):
                raise ValueError(f"beat manifest hash mismatch: {row['dataset']}")
            start, end = int(row["eligible_global_start"]), int(row["eligible_global_end"])
            if start != total or end != start + len(eligible):
                raise ValueError("eligible global range mismatch")
            self.entries.append({
                "dataset": str(row["dataset"]), "dataset_id": dataset_id,
                "raw": raw, "beat": beat, "eligible": eligible, "row": row,
            })
            total = end
            self.ends.append(total)
        if total != int(self.index["eligible_total"]):
            raise ValueError("eligible total mismatch")
        self.total = total

    def resolve(self, global_index: int) -> tuple[dict[str, Any], int]:
        global_index = int(global_index)
        if global_index < 0 or global_index >= self.total:
            raise IndexError(global_index)
        dataset_id = bisect.bisect_right(self.ends, global_index)
        start = 0 if dataset_id == 0 else self.ends[dataset_id - 1]
        entry = self.entries[dataset_id]
        raw_index = int(entry["eligible"][global_index - start])
        return entry, raw_index

    def item(self, global_index: int) -> dict[str, Any]:
        entry, raw_index = self.resolve(global_index)
        item = entry["raw"].item(raw_index)
        peaks = entry["beat"].peaks_for(raw_index)
        item.update({
            "global_index": int(global_index), "raw_index": raw_index,
            "dataset": entry["dataset"], "dataset_id": entry["dataset_id"], "peaks": peaks,
        })
        return item


class SCTranslationDataset(Dataset):
    """ECG-only beat input plus target-independent compose coordinates."""

    def __init__(
        self,
        backend: CommonBackend,
        global_indices: Iterable[int],
        *,
        max_beats: int = MAX_BEATS,
    ) -> None:
        self.backend = backend
        self.indices = np.asarray(list(global_indices) if not isinstance(global_indices, np.ndarray) else global_indices, dtype=np.int64)
        self.max_beats = int(max_beats)
        if self.max_beats < 2:
            raise ValueError("max_beats must be at least 2")

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, index: int) -> dict[str, Any]:
        global_index = int(self.indices[index])
        item = self.backend.item(global_index)
        waveform = item["waveform"]
        ecg_raw, target = waveform[0], waveform[1]
        peaks = np.asarray(item["peaks"], dtype=np.int64)
        if not 2 <= len(peaks) <= self.max_beats:
            raise RuntimeError(f"common protocol beat count violation: {global_index} count={len(peaks)}")
        starts, ends = contiguous_boundaries_from_peaks(peaks)
        count = len(peaks)

        ecg_beats = np.zeros((self.max_beats, 1, BEAT_LEN), dtype=np.float32)
        features = np.zeros((self.max_beats, 3), dtype=np.float32)
        time_sec = np.zeros(self.max_beats, dtype=np.float32)
        beat_mask = np.zeros(self.max_beats, dtype=np.bool_)
        raw_beat_index = np.zeros(WINDOW_SAMPLES, dtype=np.int64)
        raw_position = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        for beat_index, (left, right, peak) in enumerate(zip(starts, ends, peaks)):
            segment = np.asarray(ecg_raw[left:right], dtype=np.float32)
            mean = float(segment.mean())
            std = float(segment.std())
            normalized = (resample_1d(segment) - mean) / max(std, 1e-6)
            ecg_beats[beat_index, 0] = normalized.astype(np.float32)
            duration = (right - left) / float(SAMPLING_RATE)
            features[beat_index] = np.array([mean, np.log(max(std, 1e-6)), duration], np.float32)
            time_sec[beat_index] = (int(peak) - int(peaks[0])) / float(SAMPLING_RATE)
            beat_mask[beat_index] = True
            raw_beat_index[left:right] = beat_index
            raw_position[left:right] = np.minimum(
                np.arange(right - left, dtype=np.float32) * (BEAT_LEN / float(right - left)),
                BEAT_LEN - 1.0,
            )

        return {
            "global_index": torch.tensor(global_index, dtype=torch.long),
            "dataset_id": torch.tensor(int(item["dataset_id"]), dtype=torch.long),
            "dataset": str(item["dataset"]),
            "record_id": str(item["record_id"]),
            "group_id": canonical_group_id(item["dataset"], item["record_id"]),
            "ecg": torch.from_numpy(ecg_beats),
            "time_sec": torch.from_numpy(time_sec),
            "beat_mask": torch.from_numpy(beat_mask),
            "raw_features": torch.from_numpy(features),
            "raw_beat_index": torch.from_numpy(raw_beat_index),
            "raw_position": torch.from_numpy(raw_position),
            "peaks": torch.from_numpy(np.pad(peaks, (0, self.max_beats - count), constant_values=-1)),
            "target_ppg": torch.from_numpy(np.asarray(target, dtype=np.float32)[None, :].copy()),
        }


__all__ = [
    "BEAT_LEN", "CommonBackend", "DATASET_ORDER", "DETECTOR_THEORETICAL_MAX_BEATS",
    "MAX_BEATS", "SAMPLING_RATE",
    "SCTranslationDataset", "WINDOW_SAMPLES", "canonical_group_id",
    "contiguous_boundaries_from_peaks", "guard_server_path", "sha256_file",
    "sha256_indices", "stable_hash",
]
