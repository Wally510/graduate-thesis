# -*- coding: utf-8 -*-
"""
新架构第二阶段后训练（packed，2-GPU DDP 版，自包含）。

从 dual-view checkpoint 热启动，只用 recon/pat/morph/hrv 续训。
通过 torchrun 启动，例如：
  torchrun --nproc_per_node=2 train_stage2_recon_modality_flexible_packed_ddp.py --init-from ...

与单卡版 train_stage2_recon_modality_flexible_packed.py 逻辑一致，仅增加 DDP 并行。
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import socket
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from ecg_ppg_multitask_pretrain_orthogonal_modality_flexible_merged import (  # noqa: E402
    ECGPPGMultiTaskTransformer,
    count_parameters,
    move_batch,
    multitask_loss,
    sample_pretrain_mask,
    zscore_per_beat,
)

PACKED_DIR_DEFAULT = "/data-ai/sl20200894/prepared_pretrain_npz/packed_shards"
SAVE_PATH_DEFAULT = (
    "/data-ai/sl20200894/Code/foundation_stage2_ppg2ecg_aux_bundle_20260701/"
    "stage2_recon_modality_flexible_ddp.pt"
)

IN_CHANNELS = 2
BEAT_LEN = 128
MAX_BEATS = 25
MASK_RATIO = 0.30
MORPH_DIM = 6
HRV_DIM = 3
D_MODEL = 768
NHEAD = 12
NUM_LAYERS = 9
DIM_FEEDFORWARD = 2048
DROPOUT = 0.1
PHASE_TOKENS = 8

W_RECON = 1.0
W_PAT = 0.5
W_MORPH = 0.2
W_HRV = 0.2

LR_DEFAULT = 2e-5
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
SEED = 42


# =============================================================================
# DDP 运行时
# =============================================================================
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


def safe_barrier() -> None:
    if not dist.is_initialized():
        return
    try:
        dist.barrier()
    except Exception as exc:
        main_print(f"[warn] dist.barrier failed during cleanup: {exc!r}", flush=True)


def cleanup_distributed() -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    safe_barrier()
    try:
        dist.destroy_process_group()
    except Exception as exc:
        main_print(f"[warn] destroy_process_group failed: {exc!r}", flush=True)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def all_reduce_sum(value: float, device: torch.device) -> float:
    if not dist.is_initialized():
        return value
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def sync_standardizer_buffers(standardizer: nn.Module | None) -> None:
    if standardizer is None or not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    for name in ("morph_mean", "morph_std", "hrv_mean", "hrv_std", "inited"):
        buf = getattr(standardizer, name)
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        buf.div_(world_size)


# =============================================================================
# 小工具
# =============================================================================
def pad_1d(x: np.ndarray, length: int, value: float = 0.0) -> np.ndarray:
    out = np.full(length, value, dtype=np.float32)
    n = min(len(x), length)
    out[:n] = x[:n]
    return out


def pad_2d(x: np.ndarray, rows: int, cols: int, value: float = 0.0) -> np.ndarray:
    out = np.full((rows, cols), value, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    r = min(x.shape[0], rows)
    c = min(x.shape[1], cols)
    out[:r, :c] = x[:r, :c]
    return out


def fmt_hms(sec: float) -> str:
    sec = int(max(sec, 0))
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_multiprocessing_tmp() -> None:
    if os.name == "nt":
        return
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "codex"
    tmp_dir = f"/tmp/{user}_torch_mp"
    try:
        os.makedirs(tmp_dir, exist_ok=True)
    except OSError:
        return
    for key in ("TMPDIR", "TMP", "TEMP"):
        os.environ[key] = tmp_dir
    tempfile.tempdir = tmp_dir


_LOG_STATE: dict[str, object] = {}


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def __getattr__(self, name):
        return getattr(self.streams[0], name)


def setup_file_logging(log_path: str) -> None:
    if not log_path:
        return
    path = Path(log_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(path, "a", encoding="utf-8", buffering=1)
    _LOG_STATE["file"] = log_file
    _LOG_STATE["stdout"] = sys.__stdout__
    _LOG_STATE["stderr"] = sys.__stderr__
    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)
    print(f"\n===== run started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")
    print(f"logging to: {path.resolve()}")


def cleanup_file_logging() -> None:
    stdout = _LOG_STATE.get("stdout")
    stderr = _LOG_STATE.get("stderr")
    if stdout is not None:
        sys.stdout = stdout
    if stderr is not None:
        sys.stderr = stderr
    log_file = _LOG_STATE.get("file")
    if log_file is not None and hasattr(log_file, "closed") and not log_file.closed:
        try:
            log_file.flush()
            log_file.close()
        except Exception:
            pass
    _LOG_STATE.clear()


def shutdown_dataloader(loader: DataLoader | None) -> None:
    if loader is None:
        return
    try:
        if hasattr(loader, "_iterator") and loader._iterator is not None:
            loader._iterator._shutdown_workers()
            del loader._iterator
            loader._iterator = None
    except Exception as exc:
        main_print(f"[warn] dataloader worker shutdown: {exc!r}", flush=True)


def release_training_resources(
    loader: DataLoader | None,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    standardizer: nn.Module | None = None,
    device: torch.device | None = None,
) -> None:
    shutdown_dataloader(loader)
    if model is not None:
        try:
            unwrap_model(model).cpu()
        except Exception:
            pass
    del model, optimizer, standardizer, loader
    gc.collect()
    if device is not None and device.type == "cuda":
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass


class RunningStandardizer(nn.Module):
    def __init__(self, morph_dim: int = 6, hrv_dim: int = 3, momentum: float = 0.05):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("morph_mean", torch.zeros(morph_dim))
        self.register_buffer("morph_std", torch.ones(morph_dim))
        self.register_buffer("hrv_mean", torch.zeros(hrv_dim))
        self.register_buffer("hrv_std", torch.ones(hrv_dim))
        self.register_buffer("inited", torch.zeros(1))

    @torch.no_grad()
    def _update(self, mean_buf, std_buf, flat):
        if flat.numel() == 0:
            return
        m = flat.mean(0)
        s = flat.std(0).clamp_min(1e-3)
        if self.inited.item() < 1:
            mean_buf.copy_(m)
            std_buf.copy_(s)
        else:
            mean_buf.mul_(1 - self.momentum).add_(self.momentum * m)
            std_buf.mul_(1 - self.momentum).add_(self.momentum * s)

    @torch.no_grad()
    def update_from_batch(self, batch):
        morph_flat = batch.morph.reshape(-1, batch.morph.shape[-1])
        m_ok = torch.isfinite(morph_flat).all(1)
        self._update(self.morph_mean, self.morph_std, morph_flat[m_ok])
        hrv_flat = batch.hrv.reshape(-1, batch.hrv.shape[-1])
        h_ok = torch.isfinite(hrv_flat).all(1)
        self._update(self.hrv_mean, self.hrv_std, hrv_flat[h_ok])
        self.inited.fill_(1.0)

    def norm_morph(self, x):
        return (x - self.morph_mean) / self.morph_std

    def norm_hrv(self, x):
        return (x - self.hrv_mean) / self.hrv_std


def periodic_checkpoint_path(args: argparse.Namespace, epoch: int, global_step: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(args.periodic_checkpoint_dir) / (
        f"model_only_epoch{epoch:03d}_step{global_step:09d}_{stamp}.pt"
    )


def save_model_checkpoint(
    model: nn.Module,
    args: argparse.Namespace,
    out_path: str | os.PathLike,
    epoch: int,
    global_step: int,
    train_loss: float,
    n_samples: int,
) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {k: v.detach().cpu() for k, v in unwrap_model(model).state_dict().items()},
            "args": vars(args),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "train_loss": float(train_loss),
            "n_cache_files": int(n_samples),
            "stage": "recon_pat_morph_hrv_posttrain",
            "architecture": "modality_flexible_merged",
            "optimizer_state_saved": False,
            "ddp_world_size": get_world_size(),
        },
        path,
    )


def load_init_weights(model: nn.Module, path: str, device: torch.device) -> dict:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"--init-from 找不到 checkpoint: {path}")
    main_print(f"loading init weights from: {path}", flush=True)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = {"epoch": None, "global_step": None, "stage": None, "train_loss": None}
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
        meta["epoch"] = ckpt.get("epoch")
        meta["global_step"] = ckpt.get("global_step")
        meta["stage"] = ckpt.get("stage")
        meta["train_loss"] = ckpt.get("loss", ckpt.get("train_loss"))
        main_print(
            f"  ckpt meta: epoch={meta['epoch']} step={meta['global_step']} "
            f"loss={meta['train_loss']} stage={meta.get('stage', 'n/a')}",
            flush=True,
        )
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        # warm-up (train_warmup_csfm_both_distill_ddp.py) 存的是 {"model": backbone_state_dict, ...}
        state = ckpt["model"]
        meta["epoch"] = ckpt.get("epoch")
        main_print(f"  loaded from warm-up ckpt key 'model' (epoch={meta['epoch']})", flush=True)
    else:
        state = ckpt

    if not isinstance(state, dict) or len(state) == 0:
        raise ValueError(f"--init-from checkpoint 里找不到有效的 state_dict: {path}")
    # 防呆:确认取到的确实是参数字典(key 含 '.'),而不是 {model,proj,epoch,...} 顶层键
    if not any("." in k for k in list(state.keys())[:8]):
        raise ValueError(
            f"--init-from 解析出的 state 不像 state_dict (keys 无 '.'): {list(state.keys())[:8]}"
        )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if is_main_process():
        if missing:
            print(
                f"  [warn] missing keys ({len(missing)}): "
                f"{missing[:8]}{' ...' if len(missing) > 8 else ''}",
                flush=True,
            )
        if unexpected:
            print(
                f"  [warn] unexpected keys ({len(unexpected)}): "
                f"{unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}",
                flush=True,
            )
        if not missing and not unexpected:
            print("  loaded cleanly (all keys matched).", flush=True)
    return meta


def resolve_start_epoch(args: argparse.Namespace, resume_meta: dict) -> int:
    start_epoch = int(args.start_epoch)
    if args.auto_continue and resume_meta.get("epoch") is not None:
        if resume_meta.get("stage") == "recon_pat_morph_hrv_posttrain":
            start_epoch = int(resume_meta["epoch"]) + 1
            main_print(
                f"auto-continue: resume posttrain ckpt epoch={resume_meta['epoch']} "
                f"-> start_epoch={start_epoch}",
                flush=True,
            )
        else:
            main_print(
                f"auto-continue skipped: ckpt stage={resume_meta.get('stage', 'n/a')} "
                f"(use --start-epoch to override)",
                flush=True,
            )
    if start_epoch < 1:
        raise ValueError(f"--start-epoch must be >= 1, got {start_epoch}")
    if start_epoch > args.epochs:
        raise ValueError(
            f"--start-epoch ({start_epoch}) > --epochs ({args.epochs}); "
            "nothing to train"
        )
    return start_epoch


class PackedShardDataset(Dataset):
    def __init__(
        self,
        packed_dir,
        max_beats,
        in_channels,
        beat_len,
        morph_dim,
        hrv_dim,
        mask_ratio,
        training=True,
        normalize=True,
        seed=0,
    ):
        self.packed_dir = str(packed_dir)
        self.max_beats = max_beats
        self.in_channels = in_channels
        self.beat_len = beat_len
        self.morph_dim = morph_dim
        self.hrv_dim = hrv_dim
        self.mask_ratio = mask_ratio
        self.training = training
        self.normalize = normalize
        self.rng = np.random.default_rng(seed)
        self._warned = 0

        with open(os.path.join(self.packed_dir, "manifest.json")) as f:
            self.manifest = json.load(f)
        self.shard_dirs = [os.path.join(self.packed_dir, s["dir"]) for s in self.manifest["shards"]]
        self.shard_nsamples = [s["n_samples"] for s in self.manifest["shards"]]
        self.cum = np.concatenate([[0], np.cumsum(self.shard_nsamples)]).astype(np.int64)
        self.total = int(self.cum[-1])

        self._mm = {}
        self._offsets = {}

    def __len__(self):
        return self.total

    def _ensure_shard(self, si):
        if si in self._mm:
            return
        sd = self.shard_dirs[si]
        self._mm[si] = {
            "beats": np.load(os.path.join(sd, "beats.npy"), mmap_mode="r"),
            "time": np.load(os.path.join(sd, "time.npy"), mmap_mode="r"),
            "pat": np.load(os.path.join(sd, "pat.npy"), mmap_mode="r"),
            "morph": np.load(os.path.join(sd, "morph.npy"), mmap_mode="r"),
            "hrv": np.load(os.path.join(sd, "hrv.npy"), mmap_mode="r"),
        }
        self._offsets[si] = np.load(os.path.join(sd, "offsets.npy"))

    def _locate(self, gidx):
        si = int(np.searchsorted(self.cum, gidx, side="right") - 1)
        local = gidx - int(self.cum[si])
        return si, local

    def _load_one(self, gidx):
        si, local = self._locate(gidx)
        self._ensure_shard(si)
        mm = self._mm[si]
        off = self._offsets[si]
        s, e = int(off[local]), int(off[local + 1])

        beats = np.asarray(mm["beats"][s:e], dtype=np.float32)
        time_sec = np.asarray(mm["time"][s:e], dtype=np.float32)
        pat = np.asarray(mm["pat"][s:e], dtype=np.float32)
        morph = np.asarray(mm["morph"][s:e], dtype=np.float32)
        hrv = np.asarray(mm["hrv"][local], dtype=np.float32)

        if beats.ndim != 3 or beats.shape[0] < 1:
            raise ValueError(f"invalid beats shape: {beats.shape}")
        beats = beats[:, : self.in_channels, : self.beat_len]
        if self.normalize:
            beats = zscore_per_beat(beats)

        n = int(beats.shape[0])
        if n >= self.max_beats:
            if self.training:
                idx = np.sort(self.rng.choice(n, size=self.max_beats, replace=False))
            else:
                idx = np.linspace(0, n - 1, num=self.max_beats).astype(np.int64)
            beats = beats[idx]
            time_sec = time_sec[idx]
            pat = pat[idx]
            morph = morph[idx]
            beat_mask = np.ones(self.max_beats, dtype=bool)
        else:
            beat_mask = np.zeros(self.max_beats, dtype=bool)
            beat_mask[:n] = True

        k = int(beats.shape[0])
        beats_pad = np.zeros((self.max_beats, self.in_channels, self.beat_len), dtype=np.float32)
        beats_pad[:k, : beats.shape[1], : beats.shape[2]] = beats
        time_pad = pad_1d(time_sec, self.max_beats)
        pat_pad = pad_1d(pat, self.max_beats, value=np.nan)
        morph_pad = pad_2d(morph, self.max_beats, self.morph_dim, value=np.nan)
        hrv_pad = pad_1d(hrv, self.hrv_dim, value=np.nan)

        pat_mask = beat_mask & np.isfinite(pat_pad)
        morph_mask = beat_mask[:, None] & np.isfinite(morph_pad)
        hrv_mask = np.isfinite(hrv_pad)

        pat_pad = np.nan_to_num(pat_pad, nan=0.0).astype(np.float32)
        morph_pad = np.nan_to_num(morph_pad, nan=0.0).astype(np.float32)
        hrv_pad = np.nan_to_num(hrv_pad, nan=0.0).astype(np.float32)
        pretrain_mask = sample_pretrain_mask(beat_mask, self.mask_ratio, self.rng)

        return {
            "beats": torch.from_numpy(beats_pad).float(),
            "time_sec": torch.from_numpy(time_pad).float(),
            "beat_mask": torch.from_numpy(beat_mask).bool(),
            "pretrain_mask": torch.from_numpy(pretrain_mask).bool(),
            "pat": torch.from_numpy(pat_pad[:, None]).float(),
            "pat_mask": torch.from_numpy(pat_mask[:, None]).bool(),
            "morph": torch.from_numpy(morph_pad).float(),
            "morph_mask": torch.from_numpy(morph_mask).bool(),
            "hrv": torch.from_numpy(hrv_pad).float(),
            "hrv_mask": torch.from_numpy(hrv_mask).bool(),
            "modality_mask": torch.tensor([True, True], dtype=torch.bool),
            "input_mode": torch.tensor(0, dtype=torch.long),
        }

    def __getitem__(self, gidx):
        last = None
        for off in range(min(10, self.total)):
            try:
                return self._load_one((gidx + off) % self.total)
            except Exception as exc:
                last = exc
        if self._warned < 20 and is_main_process():
            print(f"[skip packed] gidx={gidx} reason={last!r}", flush=True)
            self._warned += 1
        return None


def collate_skip_none(batch):
    good = [item for item in batch if item is not None]
    if not good:
        return None
    keys = list(good[0].keys())
    return {k: torch.stack([item[k] for item in good], dim=0) for k in keys}


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    args,
    epoch,
    ckpt_state,
    standardizer,
    steps_per_epoch,
    world_size,
    use_amp=False,
):
    model.train()
    sums = {"loss": 0.0, "recon": 0.0, "pat": 0.0, "morph": 0.0, "hrv": 0.0}
    n = 0
    step = 0
    t0 = time.time()
    t_last = t0

    for items in loader:
        if items is None:
            continue
        batch = move_batch(items, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = model(
                batch.beats,
                time_sec=batch.time_sec,
                beat_mask=batch.beat_mask,
                pretrain_mask=batch.pretrain_mask,
                modality_mask=batch.modality_mask,
                input_mode=batch.modality_status,
            )

            if standardizer is not None:
                standardizer.update_from_batch(batch)
                sync_standardizer_buffers(standardizer)
                if "morph" in out:
                    out["morph"] = standardizer.norm_morph(out["morph"])
                    batch.morph = standardizer.norm_morph(batch.morph)
                if "hrv" in out:
                    out["hrv"] = standardizer.norm_hrv(out["hrv"])
                    batch.hrv = standardizer.norm_hrv(batch.hrv)

            losses = multitask_loss(
                out,
                batch,
                w_recon=args.w_recon,
                w_pat=args.w_pat,
                w_morph=args.w_morph,
                w_hrv=args.w_hrv,
            )
            total = losses["loss"]

        total.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        bs = batch.beats.shape[0]
        sums["loss"] += float(total.detach().cpu()) * bs
        for key in ("recon", "pat", "morph", "hrv"):
            sums[key] += float(losses[key].detach().cpu()) * bs
        n += bs
        step += 1
        ckpt_state["global_step"] = int(ckpt_state.get("global_step", 0)) + 1

        if step % args.log_every == 0 and is_main_process():
            now = time.time()
            dt = now - t_last
            t_last = now
            sps = (args.log_every * bs * world_size) / max(dt, 1e-6)
            frac = step / max(steps_per_epoch, 1)
            elapsed = now - t0
            eta = elapsed / max(frac, 1e-9) - elapsed
            print(
                f"  [ep{epoch:03d}] step {step}/{steps_per_epoch} ({frac * 100:5.1f}%) "
                f"loss={total.item():.4f} recon={losses['recon'].item():.3f} "
                f"pat={losses['pat'].item():.3f} morph={losses['morph'].item():.3f} "
                f"hrv={losses['hrv'].item():.3f} "
                f"| {sps:.0f} smp/s | epoch_elapsed={fmt_hms(elapsed)} epoch_ETA={fmt_hms(eta)}",
                flush=True,
            )

    if n == 0:
        raise RuntimeError("no usable samples were loaded")

    metrics = {}
    for key, val in sums.items():
        total_val = all_reduce_sum(val, device)
        total_n = all_reduce_sum(float(n), device)
        metrics[key] = total_val / max(total_n, 1.0)
    return metrics


def parse_args():
    p = argparse.ArgumentParser(
        description="Modality-flexible stage-2 post-training (DDP, recon/pat/morph/hrv only)"
    )
    p.add_argument("--init-from", required=True, help="dual-view 阶段产出的 checkpoint（热启动）")
    p.add_argument("--packed-dir", default=PACKED_DIR_DEFAULT)
    p.add_argument("--save-path", default=SAVE_PATH_DEFAULT)
    p.add_argument("--log-path", default="")
    p.add_argument("--periodic-checkpoint-dir", default="stage2_recon_modality_flexible_ddp_periodic")
    p.add_argument("--save-every-epoch", action="store_true")
    p.add_argument("--epochs", type=int, default=10, help="训练结束时的目标 epoch（含）")
    p.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="从第几个 epoch 开始训练；续训时设为 6 可从 ep5 接着跑到 --epochs",
    )
    p.add_argument(
        "--auto-continue",
        action="store_true",
        help="若 --init-from 是后训练 periodic ckpt，自动从 ckpt.epoch+1 续训",
    )
    p.add_argument("--batch-size", type=int, default=32, help="每张 GPU 的 batch size")
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--lr", type=float, default=LR_DEFAULT)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--in-channels", type=int, default=IN_CHANNELS)
    p.add_argument("--beat-len", type=int, default=BEAT_LEN)
    p.add_argument("--max-beats", type=int, default=MAX_BEATS)
    p.add_argument("--mask-ratio", type=float, default=MASK_RATIO)
    p.add_argument("--morph-dim", type=int, default=MORPH_DIM)
    p.add_argument("--hrv-dim", type=int, default=HRV_DIM)
    p.add_argument("--d-model", type=int, default=D_MODEL)
    p.add_argument("--nhead", type=int, default=NHEAD)
    p.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    p.add_argument("--dim-feedforward", type=int, default=DIM_FEEDFORWARD)
    p.add_argument("--dropout", type=float, default=DROPOUT)
    p.add_argument("--phase-tokens", type=int, default=PHASE_TOKENS)
    p.add_argument("--w-recon", type=float, default=W_RECON)
    p.add_argument("--w-pat", type=float, default=W_PAT)
    p.add_argument("--w-morph", type=float, default=W_MORPH)
    p.add_argument("--w-hrv", type=float, default=W_HRV)
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()


def main_worker(local_rank: int, args: argparse.Namespace) -> None:
    world_size = get_world_size()
    set_seed(args.seed + get_rank())
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    train_loader: DataLoader | None = None
    model: nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    standardizer: RunningStandardizer | None = None

    try:
        train_ds = PackedShardDataset(
            args.packed_dir,
            max_beats=args.max_beats,
            in_channels=args.in_channels,
            beat_len=args.beat_len,
            morph_dim=args.morph_dim,
            hrv_dim=args.hrv_dim,
            mask_ratio=args.mask_ratio,
            training=True,
            seed=args.seed + get_rank(),
        )
        train_sampler = None
        if is_distributed():
            train_sampler = DistributedSampler(
                train_ds,
                num_replicas=world_size,
                rank=get_rank(),
                shuffle=True,
                drop_last=False,
            )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_skip_none,
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
        )
        n_samples = len(train_ds)
        steps_per_epoch = len(train_loader)
        global_batch = args.batch_size * world_size

        model = ECGPPGMultiTaskTransformer(
            in_channels=args.in_channels,
            beat_len=args.beat_len,
            morph_dim=args.morph_dim,
            hrv_dim=args.hrv_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            phase_tokens=args.phase_tokens,
            num_input_modes=2,
        ).to(device)
        resume_meta = load_init_weights(model, args.init_from, device)
        start_epoch = resolve_start_epoch(args, resume_meta)

        if is_distributed():
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        if is_main_process():
            pinfo = count_parameters(unwrap_model(model))
            print(f"device: {device}  world_size={world_size}", flush=True)
            print("post-training: distill OFF, loss = recon + pat + morph + hrv (DDP)", flush=True)
            print(f"mask_ratio={args.mask_ratio}", flush=True)
            print(f"model params: total={pinfo['total']:,} trainable={pinfo['trainable']:,}", flush=True)
            print(f"packed_dir={args.packed_dir}  total_samples={n_samples}", flush=True)
            print(
                f"steps_per_epoch(per_gpu)={steps_per_epoch}  "
                f"batch_size(per_gpu)={args.batch_size}  global_batch={global_batch}  "
                f"start_epoch={start_epoch}  epochs={args.epochs}  lr={args.lr}",
                flush=True,
            )
            print(
                f"AMP(bf16): {'ENABLED' if args.amp else 'DISABLED'}  "
                f"num_workers(per_gpu)={args.num_workers}  prefetch={args.prefetch_factor}",
                flush=True,
            )

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if not args.no_standardize:
            standardizer = RunningStandardizer(morph_dim=args.morph_dim, hrv_dim=args.hrv_dim).to(device)
            main_print("target standardization: ENABLED", flush=True)

        ckpt_state = {
            "global_step": int(resume_meta.get("global_step") or 0),
            "last_train_loss": float(resume_meta.get("train_loss") or float("nan")),
        }
        train_start = time.time()

        for epoch in range(start_epoch, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if is_distributed():
                dist.barrier()

            ep_t0 = time.time()
            tr = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                args,
                epoch,
                ckpt_state,
                standardizer,
                steps_per_epoch,
                world_size,
                use_amp=args.amp,
            )
            ckpt_state["last_train_loss"] = float(tr["loss"])

            if is_main_process():
                total_elapsed = time.time() - train_start
                epochs_done = epoch - start_epoch + 1
                epochs_left = args.epochs - epoch
                eta_all = (total_elapsed / max(epochs_done, 1)) * epochs_left
                print(
                    f"epoch={epoch:03d} DONE  train_loss={tr['loss']:.4f} recon={tr['recon']:.4f} "
                    f"pat={tr['pat']:.4f} morph={tr['morph']:.4f} hrv={tr['hrv']:.4f} "
                    f"| epoch_time={fmt_hms(time.time() - ep_t0)} "
                    f"total_elapsed={fmt_hms(total_elapsed)} train_ETA={fmt_hms(eta_all)}",
                    flush=True,
                )

            if args.save_every_epoch and is_main_process():
                cp = periodic_checkpoint_path(args, epoch=epoch, global_step=int(ckpt_state["global_step"]))
                save_model_checkpoint(
                    model,
                    args,
                    cp,
                    epoch=epoch,
                    global_step=int(ckpt_state["global_step"]),
                    train_loss=float(tr["loss"]),
                    n_samples=n_samples,
                )
                print(f"saved ckpt (epoch{epoch:03d}): {cp}", flush=True)

            if is_distributed():
                dist.barrier()

        if is_main_process():
            save_model_checkpoint(
                model,
                args,
                args.save_path,
                epoch=args.epochs,
                global_step=int(ckpt_state["global_step"]),
                train_loss=float(ckpt_state["last_train_loss"]),
                n_samples=n_samples,
            )
            print(f"saved final: {args.save_path}", flush=True)
            print(
                f"finished. final_train_loss={ckpt_state['last_train_loss']:.4f} "
                f"total_time={fmt_hms(time.time() - train_start)}",
                flush=True,
            )

        if is_distributed():
            safe_barrier()
    finally:
        release_training_resources(train_loader, model, optimizer, standardizer, device)
        if is_distributed():
            safe_barrier()


def main():
    args = parse_args()
    configure_multiprocessing_tmp()

    if not is_distributed():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(find_free_port()))
        raise RuntimeError(
            "请使用 torchrun 启动本脚本，例如：\n"
            "  torchrun --nproc_per_node=2 train_stage2_recon_modality_flexible_packed_ddp.py "
            "--init-from <ckpt> ..."
        )

    local_rank = setup_distributed()
    if is_main_process():
        setup_file_logging(args.log_path)

    try:
        main_worker(local_rank, args)
    finally:
        if is_main_process():
            cleanup_file_logging()
            print("exiting cleanly.", flush=True)
        cleanup_distributed()


if __name__ == "__main__":
    main()