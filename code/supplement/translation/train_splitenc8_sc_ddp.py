#!/usr/bin/env python3
"""Full-finetune SplitEnc8 S-C on the frozen common raw10s protocol."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from common_raw10s import CommonBackend, SCTranslationDataset, guard_server_path, sha256_file, sha256_indices
from build_common_protocol import verify as verify_protocol
from splitenc8_sc_model import SplitEnc8SC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multi-cache-index", required=True)
    parser.add_argument("--protocol-dir", required=True)
    parser.add_argument("--legacy-train-package", required=True)
    parser.add_argument("--module-path", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backbone-mode", choices=("frozen", "fullfinetune"), default="fullfinetune")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=368, help="per GPU")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-beats", type=int, default=25)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--eval-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=3)
    parser.add_argument("--train-samples", type=int, default=0)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--derivative-weight", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=831)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--expected-world-size", type=int, default=4)
    parser.add_argument("--ddp-timeout-seconds", type=int, default=6000)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class StridedEvalSampler(Sampler[int]):
    def __init__(self, size: int, rank: int, world_size: int) -> None:
        self.values = range(rank, size, world_size)

    def __iter__(self):
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)


def loader_kwargs(batch_size: int, workers: int, prefetch: int) -> dict[str, Any]:
    output: dict[str, Any] = {
        "batch_size": batch_size, "num_workers": workers, "pin_memory": True,
        "drop_last": False, "persistent_workers": workers > 0,
    }
    if workers > 0:
        output["prefetch_factor"] = prefetch
    return output


def raw_loss_terms(prediction: torch.Tensor, target: torch.Tensor, derivative_weight: float):
    # Accumulate losses and metrics in FP32 even when the forward pass uses BF16.
    prediction = prediction.float()
    target = target.float()
    smooth_sum = F.smooth_l1_loss(prediction, target, reduction="sum")
    difference = prediction - target
    abs_sum = difference.abs().sum()
    sq_sum = difference.square().sum()
    pred_diff = prediction[..., 1:] - prediction[..., :-1]
    target_diff = target[..., 1:] - target[..., :-1]
    derivative_sum = (pred_diff - target_diff).abs().sum()
    count = prediction.new_tensor(prediction.numel(), dtype=torch.float64)
    derivative_count = prediction.new_tensor(pred_diff.numel(), dtype=torch.float64)
    loss = smooth_sum / count.to(prediction.dtype) + derivative_weight * derivative_sum / derivative_count.to(prediction.dtype)
    terms = torch.stack([
        smooth_sum.double(), derivative_sum.double(), abs_sum.double(), sq_sum.double(), count, derivative_count,
    ])
    return loss, terms


def terms_to_metrics(terms: torch.Tensor, derivative_weight: float) -> dict[str, float]:
    smooth_sum, derivative_sum, abs_sum, sq_sum, count, derivative_count = [float(v) for v in terms.cpu()]
    smooth = smooth_sum / max(count, 1.0)
    derivative = derivative_sum / max(derivative_count, 1.0)
    return {
        "loss": smooth + derivative_weight * derivative,
        "smooth_l1": smooth,
        "derivative_l1": derivative,
        "mae": abs_sum / max(count, 1.0),
        "rmse": math.sqrt(sq_sum / max(count, 1.0)),
        "valid_values": int(count),
    }


def forward_batch(model, batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = model(
        batch["ecg"].to(device, non_blocking=True),
        batch["time_sec"].to(device, non_blocking=True),
        batch["beat_mask"].to(device, non_blocking=True),
        batch["raw_features"].to(device, non_blocking=True),
        batch["raw_beat_index"].to(device, non_blocking=True),
        batch["raw_position"].to(device, non_blocking=True),
    )
    target = batch["target_ppg"].to(device, non_blocking=True)
    return prediction, target


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, derivative_weight: float, amp: bool) -> dict[str, float]:
    model.eval()
    sums = torch.zeros(6, dtype=torch.float64, device=device)
    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            prediction, target = forward_batch(model, batch, device)
            _loss, terms = raw_loss_terms(prediction, target, derivative_weight)
        sums += terms
    dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    return terms_to_metrics(sums, derivative_weight)


def cosine_schedule(step: int, total_steps: int, warmup_steps: int, min_ratio: float) -> float:
    if step < warmup_steps:
        return max((step + 1) / max(warmup_steps, 1), 1e-8)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def append_csv(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    if args.max_beats < 2 or args.grad_accum_steps < 1:
        raise ValueError("max-beats must be >=2 and grad-accum-steps must be >=1")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != args.expected_world_size or not torch.cuda.is_available():
        raise RuntimeError(f"expected {args.expected_world_size} CUDA ranks, got {world_size}")
    dist.init_process_group("nccl", timeout=timedelta(seconds=args.ddp_timeout_seconds))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed_everything(args.seed + rank)

    output = guard_server_path(args.output_dir, "training output")
    protocol_dir = guard_server_path(args.protocol_dir, "protocol")
    output_exists = torch.tensor(int(output.exists()), dtype=torch.int32, device=device)
    dist.all_reduce(output_exists, op=dist.ReduceOp.MAX)
    if int(output_exists.item()):
        raise FileExistsError(f"refusing to overwrite: {output}")
    dist.barrier()
    if rank == 0:
        (output / "checkpoints").mkdir(parents=True)
    dist.barrier()
    split_manifest = verify_protocol(protocol_dir)
    protocol_max_beats = int(split_manifest.get("model_max_beats", 25))
    if protocol_max_beats != args.max_beats:
        raise ValueError(f"protocol max beats {protocol_max_beats} != training max beats {args.max_beats}")
    with np.load(protocol_dir / "split_indices.npz") as split_file:
        splits = {name: np.asarray(split_file[name], dtype=np.int64) for name in ("train", "val", "test")}
    train_indices = splits["train"]
    if args.train_samples:
        if args.train_samples > len(train_indices):
            raise ValueError("train-samples exceeds train split")
        rng = np.random.default_rng(args.seed + 199)
        train_indices = np.sort(rng.choice(train_indices, args.train_samples, replace=False))

    backend = CommonBackend(args.multi_cache_index)
    train_dataset = SCTranslationDataset(backend, train_indices, max_beats=args.max_beats)
    val_dataset = SCTranslationDataset(backend, splits["val"], max_beats=args.max_beats)
    train_sampler = DistributedSampler(train_dataset, world_size, rank, shuffle=True, seed=args.seed, drop_last=False)
    val_sampler = StridedEvalSampler(len(val_dataset), rank, world_size)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs(args.batch_size, args.num_workers, args.prefetch_factor))
    val_loader = DataLoader(val_dataset, sampler=val_sampler, **loader_kwargs(args.eval_batch_size, args.eval_workers, args.prefetch_factor))

    legacy_root = guard_server_path(args.legacy_train_package, "legacy train package")
    sys.path.insert(0, str(legacy_root))
    import trainer_core_ecg2ppg as legacy
    model_module = load_module(guard_server_path(args.module_path, "model module"), "common_raw10s_splitenc8_backbone")
    architecture = SimpleNamespace(
        beat_len=128, d_model=768, nhead=12, num_layers=9,
        dim_feedforward=2048, dropout=0.1, phase_tokens=8,
    )
    backbone = legacy.build_backbone(model_module, architecture)
    checkpoint_report = legacy.load_checkpoint_strict(backbone, guard_server_path(args.base_checkpoint, "base checkpoint"))
    system = SplitEnc8SC(
        backbone,
        d_model=768,
        phase_tokens=8,
        beat_len=128,
        gradient_checkpointing=args.gradient_checkpointing,
        backbone_mode=args.backbone_mode,
    ).to(device)
    head_parameters = [parameter for name, parameter in system.named_parameters() if not name.startswith("backbone.") and parameter.requires_grad]
    backbone_parameters = [parameter for name, parameter in system.named_parameters() if name.startswith("backbone.") and parameter.requires_grad]
    groups = [{"params": head_parameters, "lr": args.head_lr, "weight_decay": args.weight_decay, "name": "sc_and_head"}]
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": args.backbone_lr, "weight_decay": args.weight_decay, "name": "active_backbone"})
    optimizer = AdamW(groups)
    microbatches_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = math.ceil(microbatches_per_epoch / args.grad_accum_steps)
    total_optimizer_steps = optimizer_steps_per_epoch * args.epochs
    scheduler = LambdaLR(
        optimizer,
        lambda step: cosine_schedule(step, total_optimizer_steps, args.warmup_steps, args.min_lr_ratio),
    )
    ddp = DistributedDataParallel(system, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    amp = bool(args.amp)

    if rank == 0:
        config = {
            "task": "ecg_to_ppg_common_raw10s_splitenc8_sc",
            "training_type": f"{args.backbone_mode}_new_sc_head",
            "source": "ECG only",
            "target_in_forward": False,
            "target": "per-window-zscore PPG [1,2500]",
            "segmentation": "ECG peak midpoint only",
            "compose": "parameter-free differentiable linear interpolation",
            "protocol_manifest": split_manifest,
            "effective_train_count": len(train_indices),
            "val_count": len(splits["val"]),
            "test_count": len(splits["test"]),
            "train_indices_sha256": sha256_indices(train_indices),
            "distributed_sampler_padding_max_per_epoch": world_size - 1,
            "max_beats": args.max_beats,
            "microbatches_per_epoch": microbatches_per_epoch,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "total_optimizer_steps": total_optimizer_steps,
            "steps_per_epoch": optimizer_steps_per_epoch,
            "total_steps": total_optimizer_steps,
            "world_size": world_size,
            "micro_global_batch_size": args.batch_size * world_size,
            "global_batch_size": args.batch_size * world_size * args.grad_accum_steps,
            "checkpoint_report": checkpoint_report,
            "args": vars(args),
        }
        (output / "training_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output / "split_manifest.json").write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    best_loss = float("inf")
    best_epoch = 0
    global_step = 0
    global_micro_step = 0
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        ddp.train()
        epoch_terms = torch.zeros(6, dtype=torch.float64, device=device)
        started = time.time()
        for local_step, batch in enumerate(train_loader, start=1):
            micro_in_group = (local_step - 1) % args.grad_accum_steps
            if micro_in_group == 0:
                optimizer.zero_grad(set_to_none=True)
                current_group_size = min(
                    args.grad_accum_steps,
                    microbatches_per_epoch - local_step + 1,
                )
                current_group_samples = sum(
                    min(
                        args.batch_size,
                        len(train_sampler) - (local_step - 1 + offset) * args.batch_size,
                    )
                    for offset in range(current_group_size)
                )
            should_step = micro_in_group + 1 == current_group_size
            sync_context = nullcontext() if should_step else ddp.no_sync()
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                    prediction, target = forward_batch(ddp, batch, device)
                    loss, terms = raw_loss_terms(prediction, target, args.derivative_weight)
                batch_weight = int(batch["ecg"].shape[0]) / max(current_group_samples, 1)
                (loss * batch_weight).backward()
            if should_step:
                clip_grad_norm_([parameter for parameter in system.parameters() if parameter.requires_grad], args.grad_clip)
                optimizer.step()
                scheduler.step()
                global_step += 1
            global_micro_step += 1
            epoch_terms += terms.detach()
            if rank == 0 and (local_step == 1 or local_step % args.log_every == 0 or local_step == microbatches_per_epoch):
                elapsed = max(time.time() - started, 1e-6)
                eta = elapsed / local_step * (microbatches_per_epoch - local_step) / 60.0
                print(
                    f"epoch={epoch}/{args.epochs} micro={local_step}/{microbatches_per_epoch} "
                    f"optimizer_step={global_step} loss={float(loss):.6f} eta_min={eta:.1f}",
                    flush=True,
                )
        dist.all_reduce(epoch_terms, op=dist.ReduceOp.SUM)
        train_metrics = terms_to_metrics(epoch_terms, args.derivative_weight)
        val_metrics = evaluate(ddp, val_loader, device, args.derivative_weight, amp)
        if rank == 0:
            row = {"epoch": epoch, "global_step": global_step, "global_micro_step": global_micro_step, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
            append_csv(output / "epoch_metrics.csv", row)
            payload = {
                "format": "splitenc8_sc_common_raw10s_configurable_v3",
                "epoch": epoch,
                "global_step": global_step,
                "global_micro_step": global_micro_step,
                "model_state": system.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "base_checkpoint_report": checkpoint_report,
                "protocol_manifest": split_manifest,
                "args": vars(args),
            }
            epoch_path = output / "checkpoints" / f"splitenc8_sc_epoch{epoch:03d}_step{global_step:09d}.pt"
            torch.save(payload, epoch_path)
            if val_metrics["loss"] < best_loss:
                best_loss, best_epoch = val_metrics["loss"], epoch
                torch.save(payload, output / "checkpoints" / "splitenc8_sc_best.pt")
            print(f"epoch_complete={epoch} train={train_metrics} val={val_metrics} best_epoch={best_epoch}", flush=True)
        dist.barrier()

    if rank == 0:
        complete = {
            "status": "complete", "best_epoch": best_epoch, "best_val_loss": best_loss,
            "selection": "validation_loss_only", "test_evaluated_during_training": False,
            "best_checkpoint": str(output / "checkpoints" / "splitenc8_sc_best.pt"),
        }
        (output / "RUN_COMPLETE.txt").write_text(json.dumps(complete, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
