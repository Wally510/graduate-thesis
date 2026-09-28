#!/usr/bin/env python3
"""两卡Token-Dense纯训练DDP核心：全量packed、无验证、每epoch保存。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


DDP_CORE_REVISION = "20260727_dataloader_batch_once_v2"


class DistInfo:
    def __init__(self) -> None:
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.is_main = self.rank == 0
        self.device = torch.device("cpu")

    def setup_required_two_gpu(self) -> torch.device:
        if self.world_size != 2:
            raise RuntimeError(
                "正式脚本固定使用两卡DDP，请用"
                "torchrun --standalone --nproc_per_node=2启动；"
                f"当前WORLD_SIZE={self.world_size}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("两卡DDP要求CUDA可用")
        visible = torch.cuda.device_count()
        if visible < 2 or self.local_rank >= visible:
            raise RuntimeError(
                f"可见GPU不足：visible={visible}, local_rank={self.local_rank}"
            )
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}")
        dist.init_process_group(backend="nccl", init_method="env://")
        return self.device

    def barrier(self) -> None:
        # 显式绑定本进程的local GPU，避免NCCL在首次barrier时猜测设备映射。
        dist.barrier(device_ids=[self.local_rank])

    def cleanup(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    def reduce_running(
        self, sums: dict[str, float], count: int
    ) -> tuple[dict[str, float], int]:
        keys = sorted(sums)
        vector = torch.tensor(
            [sums[key] for key in keys] + [float(count)],
            dtype=torch.float64,
            device=self.device,
        )
        dist.all_reduce(vector, op=dist.ReduceOp.SUM)
        return (
            {key: float(vector[index].item()) for index, key in enumerate(keys)},
            int(vector[-1].item()),
        )


class TrainSystem(nn.Module):
    def __init__(self, model: nn.Module, adapter: nn.Module, hooks) -> None:
        super().__init__()
        self.model = model
        self.adapter = adapter
        self.hooks = hooks

    def forward(
        self,
        beats: torch.Tensor,
        time_sec: torch.Tensor,
        beat_mask: torch.Tensor,
        control: torch.Tensor,
    ):
        return self.hooks.forward_train(
            self.model,
            self.adapter,
            beats,
            time_sec,
            beat_mask,
            control,
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--module-path", required=True)
    parser.add_argument("--packed-dir", required=True)
    parser.add_argument("--common-root", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="每张GPU的batch；两卡全局batch=2×该值",
    )
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--prefetch-factor", type=int, default=3)
    parser.add_argument("--head-lr", type=float, required=True)
    parser.add_argument("--backbone-lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-beats", type=int, default=25)
    parser.add_argument("--beat-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--nhead", type=int, default=12)
    parser.add_argument("--num-layers", type=int, default=9)
    parser.add_argument("--dim-feedforward", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--phase-tokens", type=int, default=8)
    parser.add_argument("--decoder-hidden", type=int, default=256)


def prepare_train_dataset(args, build):
    common = build["common"]
    model_module = build["model_module"]
    packed_manifest = json.loads(
        (Path(args.packed_dir) / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    eligible_indices, eligibility_report = (
        common.find_eligible_global_indices(
            Path(args.packed_dir),
            packed_manifest["shards"],
            min_beats=2,
        )
    )
    train_indices = eligible_indices
    if 0 < args.train_samples < train_indices.size:
        rng = np.random.default_rng(args.seed + 6101)
        train_indices = np.sort(
            rng.choice(
                train_indices, size=args.train_samples, replace=False
            )
        )
    train_sha = hashlib.sha256(
        train_indices.astype(np.int64, copy=False).tobytes()
    ).hexdigest()
    zscore = getattr(model_module, "zscore_per_beat", None)
    dataset_class = common.IndexedPackedBeatDataset
    train_dataset = dataset_class(
        Path(args.packed_dir),
        train_indices,
        max_beats=args.max_beats,
        beat_len=args.beat_len,
        zscore_function=zscore,
        subset_seed=args.seed,
        training=True,
    )
    return train_dataset, train_indices, train_sha, eligibility_report


def save_checkpoint(
    path: Path,
    raw_system: TrainSystem,
    *,
    args,
    hooks,
    build,
    local_step: int,
    local_epoch: int,
    kind: str,
    steps_per_epoch: int,
    global_batch_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    source_report = build["source_report"]
    payload = {
        "phase_adapter_state": {
            key: value.detach().cpu()
            for key, value in raw_system.adapter.state_dict().items()
        },
        "full_model_state": {
            key: value.detach().cpu()
            for key, value in raw_system.model.state_dict().items()
        },
        "step": int(local_step),
        "epoch": int(local_epoch),
        "validation": None,
        "validation_disabled": True,
        "kind": kind,
        "policy": hooks.policy,
        "task": hooks.task_name,
        "source_checkpoint": str(args.source_checkpoint),
        "source_step": int(source_report.get("step", 0)),
        "source_epoch": int(source_report.get("epoch", 0)),
        "parameter_warm_start": bool(
            source_report.get("parameter_warm_start", True)
        ),
        "optimizer_state_saved": False,
        "scheduler_state_saved": False,
        "full_model_state_saved": True,
        "ddp": {
            "world_size": 2,
            "per_gpu_batch_size": int(args.batch_size),
            "global_batch_size": int(global_batch_size),
            "steps_per_full_epoch": int(steps_per_epoch),
            "find_unused_parameters": False,
        },
        "active_losses": hooks.active_losses(args),
        "args": vars(args),
    }
    payload.update(hooks.checkpoint_metadata(args))
    torch.save(payload, path)


def train_ddp(args, hooks) -> None:
    dinfo = DistInfo()
    device = dinfo.setup_required_two_gpu()
    try:
        seed_everything(args.seed + dinfo.rank)
        if args.phase_tokens != 8 or args.beat_len != 128:
            raise ValueError("固定beat_len=128、phase_tokens=8")
        if args.epochs < 1:
            raise ValueError("epochs必须>=1")
        output = Path(args.output_dir)
        if dinfo.is_main:
            output.mkdir(parents=True, exist_ok=True)
        dinfo.barrier()

        build = hooks.build(args, device)
        raw_system = TrainSystem(
            build["model"], build["adapter"], hooks
        ).to(device)
        for parameter in raw_system.model.parameters():
            parameter.requires_grad_(True)
        hooks.configure_trainable(raw_system.adapter)

        (
            train_dataset,
            train_indices,
            train_sha,
            eligibility_report,
        ) = prepare_train_dataset(args, build)
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=dinfo.world_size,
            rank=dinfo.rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
        loader_kwargs: dict[str, Any] = {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "pin_memory": True,
            "persistent_workers": args.num_workers > 0,
        }
        if args.num_workers > 0:
            loader_kwargs["prefetch_factor"] = args.prefetch_factor
        train_loader = DataLoader(
            train_dataset,
            sampler=sampler,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )
        steps_per_epoch = len(train_loader)
        total_steps = args.epochs * steps_per_epoch
        global_batch = args.batch_size * dinfo.world_size

        if dinfo.is_main:
            split_report = {
                "method": "all_eligible_packed_no_validation",
                "train_count": int(train_indices.size),
                "validation_count": 0,
                "train_indices_sha256_int64": train_sha,
                "eligibility_report": eligibility_report,
            }
            (output / "split_manifest.json").write_text(
                json.dumps(split_report, ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
            config = vars(args).copy()
            config.update(
                {
                    "world_size": 2,
                    "per_gpu_batch_size": args.batch_size,
                    "global_batch_size": global_batch,
                    "steps_per_full_epoch": steps_per_epoch,
                    "total_optimizer_steps": total_steps,
                    "train_count": int(train_indices.size),
                    "validation_count": 0,
                    "checkpoint_policy": "epoch_end_only",
                    "active_losses": hooks.active_losses(args),
                }
            )
            (output / "train_config.json").write_text(
                json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (output / "source_checkpoint_report.json").write_text(
                json.dumps(
                    build["source_report"],
                    ensure_ascii=False,
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )

        ddp_system = DDP(
            raw_system,
            device_ids=[dinfo.local_rank],
            output_device=dinfo.local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
        parameter_groups = hooks.optimizer_groups(
            raw_system, args
        )
        optimizer = torch.optim.AdamW(
            parameter_groups, weight_decay=args.weight_decay
        )

        def lr_factor(step_index: int) -> float:
            if step_index < args.warmup_steps:
                return max(
                    (step_index + 1) / max(args.warmup_steps, 1),
                    1e-3,
                )
            progress = (step_index - args.warmup_steps) / max(
                total_steps - args.warmup_steps, 1
            )
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return args.min_lr_ratio + (
                1.0 - args.min_lr_ratio
            ) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_factor
        )
        use_amp = bool(
            args.amp
            and getattr(
                torch.cuda, "is_bf16_supported", lambda: False
            )()
        )
        args.amp = use_amp
        rng = np.random.default_rng(
            args.seed + 4001 + 100003 * dinfo.rank
        )
        checkpoint_rows: list[dict[str, Any]] = []
        global_step = 0
        running = {name: 0.0 for name in hooks.loss_names}
        running_count = 0
        start_time = time.time()

        for epoch in range(1, args.epochs + 1):
            sampler.set_epoch(epoch)
            ddp_system.train()
            epoch_start_time = time.time()
            for epoch_step, batch in enumerate(train_loader, start=1):
                beats = batch["beats"].to(device, non_blocking=True)
                time_sec = batch["time_sec"].to(
                    device, non_blocking=True
                )
                beat_mask = batch["beat_mask"].to(
                    device, non_blocking=True
                )
                control = hooks.make_train_control(
                    batch, args, rng, device
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=use_amp,
                ):
                    prediction, aux = ddp_system(
                        beats, time_sec, beat_mask, control
                    )
                    loss, parts = hooks.loss(
                        prediction,
                        beats,
                        aux,
                        control,
                        beat_mask,
                        args,
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"rank={dinfo.rank} step={global_step} loss非有限"
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in raw_system.parameters()
                        if parameter.requires_grad
                    ],
                    args.grad_clip,
                )
                optimizer.step()
                scheduler.step()
                global_step += 1
                for name in hooks.loss_names:
                    running[name] += float(parts[name])
                running_count += 1

                if global_step % args.log_every == 0:
                    reduced, reduced_count = dinfo.reduce_running(
                        running, running_count
                    )
                    if dinfo.is_main:
                        message = " ".join(
                            f"{name}={reduced[name]/max(reduced_count,1):.5f}"
                            for name in hooks.loss_names
                        )
                        elapsed_epoch = time.time() - epoch_start_time
                        seconds_per_step = elapsed_epoch / max(epoch_step, 1)
                        remaining_seconds = seconds_per_step * (
                            steps_per_epoch - epoch_step
                        )
                        print(
                            f"step={global_step}/{total_steps} "
                            f"epoch={epoch}/{args.epochs} "
                            f"epoch_step={epoch_step}/{steps_per_epoch} "
                            f"{message} "
                            f"epoch_elapsed_min={elapsed_epoch/60:.1f} "
                            f"epoch_eta_min={remaining_seconds/60:.1f}",
                            flush=True,
                        )
                    running = {
                        name: 0.0 for name in hooks.loss_names
                    }
                    running_count = 0

            dinfo.barrier()
            if dinfo.is_main:
                path = (
                    output
                    / "checkpoints"
                    / f"model_epoch{epoch:03d}_step{global_step:09d}.pt"
                )
                save_checkpoint(
                    path,
                    raw_system,
                    args=args,
                    hooks=hooks,
                    build=build,
                    local_step=global_step,
                    local_epoch=epoch,
                    kind="epoch",
                    steps_per_epoch=steps_per_epoch,
                    global_batch_size=global_batch,
                )
                checkpoint_rows.append(
                    {
                        "epoch": epoch,
                        "step": global_step,
                        "checkpoint": str(path),
                        "epoch_elapsed_seconds": (
                            time.time() - epoch_start_time
                        ),
                    }
                )
                (output / "checkpoint_index.json").write_text(
                    json.dumps(
                        checkpoint_rows, ensure_ascii=False, indent=2
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(
                    "epoch_complete="
                    + json.dumps(checkpoint_rows[-1]),
                    flush=True,
                )
            dinfo.barrier()

        if dinfo.is_main:
            last_path = Path(checkpoint_rows[-1]["checkpoint"])
            summary = {
                "status": "complete",
                "task": hooks.task_name,
                "policy": hooks.policy,
                "world_size": 2,
                "per_gpu_batch_size": args.batch_size,
                "global_batch_size": global_batch,
                "train_samples": len(train_dataset),
                "validation_samples": 0,
                "validation_disabled": True,
                "steps_per_full_epoch": steps_per_epoch,
                "global_step": global_step,
                "completed_full_epochs": args.epochs,
                "checkpoint_policy": "one_checkpoint_per_epoch",
                "last_checkpoint": str(last_path),
                "active_losses": hooks.active_losses(args),
                "elapsed_seconds": time.time() - start_time,
            }
            summary.update(hooks.summary_metadata(args, output))
            (output / "training_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                "training_complete=" + json.dumps(summary),
                flush=True,
            )
        dinfo.barrier()
    finally:
        dinfo.cleanup()
