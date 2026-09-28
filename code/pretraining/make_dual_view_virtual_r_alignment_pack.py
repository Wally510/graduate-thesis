from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def make_memmap(path: Path, shape: Tuple[int, ...], dtype: np.dtype) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def parse_dtype(name: str) -> np.dtype:
    if name == "float16":
        return np.dtype(np.float16)
    if name == "float32":
        return np.dtype(np.float32)
    raise ValueError(f"unsupported dtype: {name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pack dual-view virtual-R alignment data into memmap shards.")
    p.add_argument("--model-code-dir", default="/data-ai/sl20200894/Code/foundation_embedding_alignment_warmup_nosort_bundle_20260628")
    p.add_argument("--cache-root", default="/data-ai/sl20200894/prepared_pretrain_npz/ecg_ppg_foundation_raw_anchor")
    p.add_argument("--teacher-matrix", default="/data-ai/sl20200894/CSFM/raw_anchor_csfm_embeddings/emb_matrix.npy")
    p.add_argument("--teacher-rel-paths", default="/data-ai/sl20200894/CSFM/raw_anchor_csfm_embeddings/rel_paths.txt")
    p.add_argument("--out-dir", default="/data-ai/sl20200894/prepared_pretrain_npz/dual_view_virtual_r_alignment_packed")
    p.add_argument("--anchor-code-dir", default="/data-ai/sl20200894/Code/ppg_anchor_distance_split_bundle_20260627")
    p.add_argument("--anchor-ckpt", default="/data-ai/sl20200894/Code/ppg_anchor_distance_split_bundle_20260627/ppg_anchor_distance_net_subject_split.pt")
    p.add_argument(
        "--anchor-device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help=(
            "Device for PPGAnchorDistanceNet during packing. Default is cpu so "
            "Slurm array packing can run without occupying the only GPU."
        ),
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard-size", type=int, default=50000)
    p.add_argument("--part-id", type=int, default=0)
    p.add_argument("--num-parts", type=int, default=1)
    p.add_argument("--store-dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--teacher-dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress-every", type=int, default=1000)
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--max-beats", type=int, default=25)
    p.add_argument("--beat-len", type=int, default=128)
    p.add_argument("--anchor-target-fs", type=float, default=125.0)
    p.add_argument("--anchor-w-left-sec", type=float, default=0.6)
    p.add_argument("--anchor-w-right-sec", type=float, default=0.4)
    p.add_argument("--anchor-width", type=int, default=64)
    p.add_argument("--anchor-kernel-size", type=int, default=7)
    p.add_argument("--anchor-dilations", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--anchor-dropout", type=float, default=0.1)
    p.add_argument("--virtual-r-jitter-prob", type=float, default=0.8)
    p.add_argument("--virtual-r-jitter-std-sec", type=float, default=0.02)
    p.add_argument("--virtual-r-jitter-max-sec", type=float, default=0.06)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_code_dir = Path(args.model_code_dir).resolve()
    sys.path.insert(0, str(model_code_dir))
    from train_dual_view_virtual_r_alignment import (  # noqa: WPS433
        fallback_ppg_from_cached,
        load_anchor_net,
        load_teacher_memmap,
        normalize_cached_beats,
        pad_beats,
        predict_virtual_r_idx,
        segment_ppg_by_virtual_r,
    )

    if args.num_parts < 1:
        raise ValueError("--num-parts must be >= 1")
    if args.part_id < 0 or args.part_id >= args.num_parts:
        raise ValueError(f"--part-id must be in [0, {args.num_parts}), got {args.part_id}")

    root_out_dir = Path(args.out_dir)
    out_dir = root_out_dir if args.num_parts == 1 else root_out_dir / f"part_{args.part_id:03d}"
    if out_dir.exists() and args.overwrite:
        log(f"removing existing out_dir={out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log("loading teacher memmap")
    teacher_keys, emb_matrix = load_teacher_memmap(args.teacher_matrix, args.teacher_rel_paths)
    total_all = len(teacher_keys)
    if args.limit and args.limit > 0:
        total_all = min(total_all, int(args.limit))
    part_start = total_all * args.part_id // args.num_parts
    part_end = total_all * (args.part_id + 1) // args.num_parts
    total = part_end - part_start
    log(
        f"pack config total_all={total_all} part={args.part_id}/{args.num_parts} "
        f"rows={part_start}:{part_end} part_total={total} shard_size={args.shard_size} "
        f"store_dtype={args.store_dtype} teacher_dtype={args.teacher_dtype} out_dir={out_dir}"
    )

    if args.anchor_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.anchor_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--anchor-device cuda requested, but torch.cuda.is_available() is false")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log(f"loading anchor_net device={device} anchor_device_arg={args.anchor_device}")
    anchor_net = load_anchor_net(args, device)

    cache_root = Path(args.cache_root)
    store_dtype = parse_dtype(args.store_dtype)
    teacher_dtype = parse_dtype(args.teacher_dtype)
    manifest: Dict[str, object] = {
        "created_at": now(),
        "model_code_dir": str(model_code_dir),
        "cache_root": str(cache_root),
        "teacher_matrix": str(args.teacher_matrix),
        "teacher_rel_paths": str(args.teacher_rel_paths),
        "anchor_ckpt": str(args.anchor_ckpt),
        "root_out_dir": str(root_out_dir),
        "part_id": args.part_id,
        "num_parts": args.num_parts,
        "row_start": part_start,
        "row_end": part_end,
        "total_all": total_all,
        "total_samples": total,
        "shard_size": args.shard_size,
        "store_dtype": args.store_dtype,
        "teacher_dtype": args.teacher_dtype,
        "max_beats": args.max_beats,
        "beat_len": args.beat_len,
        "teacher_dim": int(emb_matrix.shape[1]),
        "shards": [],
    }

    global_i = part_start
    shard_id = 0
    pack_t0 = time.time()
    while global_i < part_end:
        n = min(args.shard_size, part_end - global_i)
        shard_name = f"shard_{shard_id:06d}"
        shard_dir = out_dir / shard_name
        shard_dir.mkdir(parents=True, exist_ok=True)
        log(f"[{shard_name}] start rows={global_i}:{global_i+n}")

        beats_both_arr = make_memmap(shard_dir / "beats_both.npy", (n, args.max_beats, 2, args.beat_len), store_dtype)
        beats_ppg_arr = make_memmap(shard_dir / "beats_ppg.npy", (n, args.max_beats, 2, args.beat_len), store_dtype)
        time_both_arr = make_memmap(shard_dir / "time_both.npy", (n, args.max_beats), np.float32)
        time_ppg_arr = make_memmap(shard_dir / "time_ppg.npy", (n, args.max_beats), np.float32)
        mask_both_arr = make_memmap(shard_dir / "mask_both.npy", (n, args.max_beats), np.bool_)
        mask_ppg_arr = make_memmap(shard_dir / "mask_ppg.npy", (n, args.max_beats), np.bool_)
        teacher_arr = make_memmap(shard_dir / "teacher_emb.npy", (n, int(emb_matrix.shape[1])), teacher_dtype)
        status_arr = make_memmap(shard_dir / "status.npy", (n,), np.uint8)

        rel_paths: List[str] = []
        fallback_count = 0
        shard_t0 = time.time()
        for local in range(n):
            row = global_i + local
            key = teacher_keys[row]
            path = Path(key)
            if not path.is_absolute():
                path = cache_root / key

            with np.load(path, allow_pickle=False) as data:
                beats = normalize_cached_beats(data["beats"])
                time_sec = np.asarray(data["time_sec"], dtype=np.float32)
                beats_both, time_both, mask_both = pad_beats(beats, time_sec, args.max_beats, args.beat_len)
                raw_ppg = np.asarray(data["raw_ppg"], dtype=np.float32)
                fs = float(np.asarray(data["fs"]).reshape(-1)[0])
                ppg_idx = np.asarray(data["ppg_maxslope_idx"], dtype=np.int64)

            virtual_r = predict_virtual_r_idx(
                anchor_net=anchor_net,
                raw_ppg=raw_ppg,
                fs=fs,
                ppg_idx=ppg_idx,
                device=device,
                target_fs=args.anchor_target_fs,
                w_left_sec=args.anchor_w_left_sec,
                w_right_sec=args.anchor_w_right_sec,
                jitter_prob=args.virtual_r_jitter_prob,
                jitter_std_sec=args.virtual_r_jitter_std_sec,
                jitter_max_sec=args.virtual_r_jitter_max_sec,
            )
            packed_ppg = segment_ppg_by_virtual_r(raw_ppg, fs, virtual_r, args.max_beats, args.beat_len)
            if packed_ppg is None:
                packed_ppg = fallback_ppg_from_cached(
                    torch.from_numpy(beats_both),
                    torch.from_numpy(time_both),
                    torch.from_numpy(mask_both),
                )
                fallback_count += 1
                status_arr[local] = 1
            else:
                status_arr[local] = 0
            beats_ppg, time_ppg, mask_ppg = packed_ppg

            beats_both_arr[local] = beats_both.astype(store_dtype, copy=False)
            beats_ppg_arr[local] = beats_ppg.astype(store_dtype, copy=False)
            time_both_arr[local] = time_both
            time_ppg_arr[local] = time_ppg
            mask_both_arr[local] = mask_both
            mask_ppg_arr[local] = mask_ppg
            teacher_arr[local] = emb_matrix[row].astype(teacher_dtype, copy=False)
            rel_paths.append(key)

            done_part = row - part_start + 1
            if args.progress_every > 0 and done_part % args.progress_every == 0:
                elapsed = time.time() - pack_t0
                speed = done_part / max(elapsed, 1e-6)
                eta = (total - done_part) / max(speed, 1e-6)
                log(
                    f"[pack part={args.part_id}/{args.num_parts}] {done_part}/{total} "
                    f"global_row={row + 1}/{total_all} shard={shard_id} "
                    f"fallback_shard={fallback_count} speed={speed:.1f}/s eta={eta/3600:.2f}h"
                )

        for arr in (beats_both_arr, beats_ppg_arr, time_both_arr, time_ppg_arr, mask_both_arr, mask_ppg_arr, teacher_arr, status_arr):
            arr.flush()
        with open(shard_dir / "rel_paths.txt", "w", encoding="utf-8") as f:
            for rel in rel_paths:
                f.write(rel + "\n")

        shard_info = {
            "dir": shard_name,
            "n_samples": n,
            "fallback_count": int(fallback_count),
        }
        manifest["shards"].append(shard_info)  # type: ignore[index]
        with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        log(f"[{shard_name}] done n={n} fallback={fallback_count} elapsed={(time.time()-shard_t0)/60:.1f}min")

        global_i += n
        shard_id += 1

    log(
        f"pack done part={args.part_id}/{args.num_parts} total={total} shards={shard_id} "
        f"elapsed={(time.time()-pack_t0)/3600:.2f}h out_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
