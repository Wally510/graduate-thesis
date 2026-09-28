from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def fmt_hms(sec: float) -> str:
    sec = int(max(sec, 0))
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def epoch_checkpoint_path(save_path: Path, epoch: int) -> Path:
    return save_path.with_name(f"{save_path.stem}_epoch{epoch:03d}{save_path.suffix}")


def load_resume_checkpoint(
    resume_path: Path,
    model: nn.Module,
    proj: nn.Module,
    device: torch.device,
) -> Tuple[int, float]:
    """Load weights from an epoch/best checkpoint; return (start_epoch, best_loss)."""
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError(f"checkpoint missing model_state: {resume_path}")
    model.load_state_dict(ckpt["model_state"], strict=True)
    if "proj_state" in ckpt:
        proj.load_state_dict(ckpt["proj_state"], strict=True)
    else:
        log(f"[resume] warning: no proj_state in {resume_path}, projection head stays random-init")
    done_epoch = int(ckpt.get("epoch", 0))
    best_loss = float(ckpt.get("loss", math.inf))
    log(
        f"[resume] loaded {resume_path} done_epoch={done_epoch:03d} "
        f"checkpoint_loss={best_loss:.5f}"
    )
    return done_epoch + 1, best_loss


class PackedDualViewDataset(Dataset):
    def __init__(self, packed_dir: str) -> None:
        self.packed_dir = Path(packed_dir)
        self.manifests: List[Dict[str, object]] = []
        self.shards: List[Dict[str, object]] = []
        root_manifest = self.packed_dir / "manifest.json"
        if root_manifest.exists():
            self._load_manifest(root_manifest)
        else:
            part_manifests = sorted(self.packed_dir.glob("part_*/manifest.json"))
            if not part_manifests:
                raise FileNotFoundError(
                    f"no manifest found in {self.packed_dir}; expected manifest.json or part_*/manifest.json"
                )
            for manifest_path in part_manifests:
                self._load_manifest(manifest_path)
        self.ns = [int(s["n_samples"]) for s in self.shards]
        self.cum = np.concatenate([[0], np.cumsum(self.ns)]).astype(np.int64)
        self.total = int(self.cum[-1])
        self._mm: Dict[int, Dict[str, np.ndarray]] = {}

    def _load_manifest(self, manifest_path: Path) -> None:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.manifests.append(manifest)
        base_dir = manifest_path.parent
        for shard in manifest["shards"]:
            item = dict(shard)
            item["dir_path"] = str(base_dir / str(shard["dir"]))
            self.shards.append(item)

    def __len__(self) -> int:
        return self.total

    def _locate(self, idx: int) -> Tuple[int, int]:
        shard_id = int(np.searchsorted(self.cum, idx, side="right") - 1)
        local = int(idx - self.cum[shard_id])
        return shard_id, local

    def _ensure_shard(self, shard_id: int) -> None:
        if shard_id in self._mm:
            return
        shard_dir = Path(str(self.shards[shard_id]["dir_path"]))
        self._mm[shard_id] = {
            "beats_both": np.load(shard_dir / "beats_both.npy", mmap_mode="r"),
            "beats_ppg": np.load(shard_dir / "beats_ppg.npy", mmap_mode="r"),
            "time_both": np.load(shard_dir / "time_both.npy", mmap_mode="r"),
            "time_ppg": np.load(shard_dir / "time_ppg.npy", mmap_mode="r"),
            "mask_both": np.load(shard_dir / "mask_both.npy", mmap_mode="r"),
            "mask_ppg": np.load(shard_dir / "mask_ppg.npy", mmap_mode="r"),
            "teacher_emb": np.load(shard_dir / "teacher_emb.npy", mmap_mode="r"),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        shard_id, local = self._locate(idx)
        self._ensure_shard(shard_id)
        mm = self._mm[shard_id]
        return {
            "beats_both": torch.from_numpy(np.asarray(mm["beats_both"][local], dtype=np.float32)),
            "beats_ppg": torch.from_numpy(np.asarray(mm["beats_ppg"][local], dtype=np.float32)),
            "time_both": torch.from_numpy(np.asarray(mm["time_both"][local], dtype=np.float32)),
            "time_ppg": torch.from_numpy(np.asarray(mm["time_ppg"][local], dtype=np.float32)),
            "mask_both": torch.from_numpy(np.asarray(mm["mask_both"][local], dtype=np.bool_)),
            "mask_ppg": torch.from_numpy(np.asarray(mm["mask_ppg"][local], dtype=np.bool_)),
            "teacher_emb": torch.from_numpy(np.asarray(mm["teacher_emb"][local], dtype=np.float32)),
        }


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def cosine_distill_loss(student_z: torch.Tensor, teacher_z: torch.Tensor) -> torch.Tensor:
    student_z = F.normalize(student_z.float(), dim=-1)
    teacher_z = F.normalize(teacher_z.float(), dim=-1)
    return (1.0 - (student_z * teacher_z).sum(dim=-1)).mean()


def masked_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    beat_mask: torch.Tensor,
    *,
    beta: float = 0.05,
) -> torch.Tensor:
    """Huber loss over valid beats only.

    `pred`/`target` are typically [B,K,C,L]; `beat_mask` is [B,K].
    """
    pred_f = pred.float()
    target_f = target.float()
    valid = beat_mask.to(device=pred.device, dtype=torch.bool)
    while valid.ndim < pred_f.ndim:
        valid = valid.unsqueeze(-1)
    valid = valid.expand_as(pred_f)
    if not bool(valid.any()):
        return pred_f.sum() * 0.0
    elem = F.smooth_l1_loss(pred_f, target_f, reduction="none", beta=beta)
    return elem.masked_select(valid).mean()


def first_diff(x: torch.Tensor) -> torch.Tensor:
    return x[..., 1:] - x[..., :-1]


def ppg2ecg_aux_loss(
    ecg_hat: torch.Tensor,
    ecg_target: torch.Tensor,
    beat_mask: torch.Tensor,
    *,
    beta: float = 0.05,
    diff_weight: float = 0.2,
) -> torch.Tensor:
    """PPG-only latent -> ECG waveform translation loss.

    This is auxiliary supervision only. The generated ECG is not fed back into
    the fusion path as a real ECG modality.
    """
    waveform = masked_huber_loss(ecg_hat, ecg_target, beat_mask, beta=beta)
    slope = masked_huber_loss(first_diff(ecg_hat), first_diff(ecg_target), beat_mask, beta=beta)
    return waveform + diff_weight * slope


def count_parameters(module: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage-2 dual-view virtual-R alignment with PPG-to-ECG auxiliary translation."
    )
    p.add_argument("--model-code-dir", default="")
    p.add_argument("--packed-dir", default="/data-ai/sl20200894/prepared_pretrain_npz/dual_view_virtual_r_alignment_packed")
    p.add_argument("--save-path", default="/data-ai/sl20200894/Code/foundation_stage2_ppg2ecg_aux_bundle_20260701/stage2_ppg2ecg_aux_packed_amp_merged.pt")
    p.add_argument("--epochs", type=int, default=15, help="Train through this epoch (inclusive).")
    p.add_argument(
        "--resume",
        default="",
        help="Path to epoch/best checkpoint (.pt). Training continues from the next epoch.",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--w-both", type=float, default=0.5)
    p.add_argument("--w-ppg", type=float, default=1.0)
    p.add_argument("--w-cons", type=float, default=0.1)
    p.add_argument("--w-ppg2ecg", type=float, default=0.1)
    p.add_argument("--w-ppg-recon", type=float, default=0.05)
    p.add_argument("--w-recon-both", type=float, default=0.0)
    p.add_argument("--ppg2ecg-diff-weight", type=float, default=0.2)
    p.add_argument("--huber-beta", type=float, default=0.05)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--beat-len", type=int, default=128)
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--nhead", type=int, default=12)
    p.add_argument("--num-layers", type=int, default=9)
    p.add_argument("--dim-feedforward", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--phase-tokens", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_code_dir = Path(args.model_code_dir).resolve() if args.model_code_dir else Path(__file__).resolve().parent
    sys.path.insert(0, str(model_code_dir))
    from ecg_ppg_multitask_pretrain_orthogonal_modality_flexible_merged import ECGPPGMultiTaskTransformer  # noqa: WPS433

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    log(f"device={device} amp_bf16={int(use_amp)} packed_dir={args.packed_dir}")

    ds = PackedDualViewDataset(args.packed_dir)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
    )
    steps_per_epoch = len(loader)
    if args.max_steps and args.max_steps > 0:
        steps_per_epoch = min(steps_per_epoch, int(args.max_steps))
    log(f"samples={len(ds)} batch_size={args.batch_size} steps_per_epoch={steps_per_epoch} workers={args.num_workers} prefetch={args.prefetch_factor}")

    model = ECGPPGMultiTaskTransformer(
        in_channels=2,
        beat_len=args.beat_len,
        morph_dim=6,
        hrv_dim=3,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        phase_tokens=args.phase_tokens,
        num_input_modes=2,
    ).to(device)
    proj = ProjectionHead(args.d_model, 768).to(device)
    mt, mtr = count_parameters(model)
    pt, ptr = count_parameters(proj)
    log(f"student parameters total={mt:,} trainable={mtr:,}; projection total={pt:,} trainable={ptr:,}")

    opt = torch.optim.AdamW(list(model.parameters()) + list(proj.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    best_loss = math.inf
    start_epoch = 1
    if args.resume:
        resume_path = Path(args.resume).resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        start_epoch, best_loss = load_resume_checkpoint(resume_path, model, proj, device)
        if start_epoch > args.epochs:
            raise ValueError(
                f"resume would start at epoch {start_epoch:03d}, but --epochs={args.epochs}. "
                "Set --epochs higher than the checkpoint epoch."
            )
        log(f"[resume] will train epochs {start_epoch:03d}..{args.epochs:03d} (best_loss seed={best_loss:.5f})")
    else:
        log(f"[train] fresh start, epochs 001..{args.epochs:03d}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        proj.train()
        total_loss = 0.0
        component_sums = {
            "teacher_both": 0.0,
            "teacher_ppg": 0.0,
            "cons": 0.0,
            "ppg2ecg": 0.0,
            "ppg_recon": 0.0,
            "recon_both": 0.0,
        }
        n_steps = 0
        epoch_t0 = time.time()
        last_log_t = epoch_t0
        log(f"[epoch {epoch:03d}] start")
        for step, batch in enumerate(loader, start=1):
            if args.max_steps and args.max_steps > 0 and step > args.max_steps:
                break
            beats_both = batch["beats_both"].to(device, non_blocking=True)
            beats_ppg = batch["beats_ppg"].to(device, non_blocking=True)
            time_both = batch["time_both"].to(device, non_blocking=True)
            time_ppg = batch["time_ppg"].to(device, non_blocking=True)
            mask_both = batch["mask_both"].to(device, non_blocking=True)
            mask_ppg = batch["mask_ppg"].to(device, non_blocking=True)
            teacher = batch["teacher_emb"].to(device, non_blocking=True)

            bsz = beats_both.shape[0]
            both_modality_mask = torch.ones((bsz, 2), dtype=torch.bool, device=device)
            ppg_modality_mask = torch.tensor([[False, True]], dtype=torch.bool, device=device).repeat(bsz, 1)
            both_input_mode = torch.zeros((bsz,), dtype=torch.long, device=device)
            ppg_input_mode = torch.ones((bsz,), dtype=torch.long, device=device)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                # Merge the both-view and the ppg-view into ONE forward of size 2B.
                # The model has no cross-sample ops except BatchNorm, and the ECG
                # skip inside the model keeps that BN clean, so this is equivalent
                # to two separate forwards but avoids half the kernel-launch /
                # Python overhead and raises GPU occupancy. Order is [both ; ppg]:
                # the first bsz rows are the both-view, the last bsz are the ppg-view.
                beats_cat = torch.cat([beats_both, beats_ppg], dim=0)
                time_cat = torch.cat([time_both, time_ppg], dim=0)
                mask_cat = torch.cat([mask_both, mask_ppg], dim=0)
                modality_mask_cat = torch.cat([both_modality_mask, ppg_modality_mask], dim=0)
                input_mode_cat = torch.cat([both_input_mode, ppg_input_mode], dim=0)

                out = model(
                    beats_cat,
                    beat_mask=mask_cat,
                    time_sec=time_cat,
                    modality_mask=modality_mask_cat,
                    input_mode=input_mode_cat,
                )
                cls_all = out["cls"]
                recon_all = out["recon"]
                cls_both, cls_ppg = cls_all[:bsz], cls_all[bsz:]
                recon_both, recon_ppg = recon_all[:bsz], recon_all[bsz:]

                z_all = proj(cls_all)
                loss_both = cosine_distill_loss(z_all[:bsz], teacher)
                loss_ppg = cosine_distill_loss(z_all[bsz:], teacher)
                loss_cons = cosine_distill_loss(cls_ppg, cls_both.detach())

                # PPG-only view uses zero ECG as input, but the ECG target must
                # come from paired real ECG in beats_both, not from beats_ppg.
                ecg_hat = recon_ppg[:, :, 0:1, :]
                ppg_hat = recon_ppg[:, :, 1:2, :]
                ecg_target = beats_both[:, :, 0:1, :]
                ppg_target = beats_ppg[:, :, 1:2, :]
                valid_ppg2ecg = mask_ppg & mask_both
                loss_ppg2ecg = ppg2ecg_aux_loss(
                    ecg_hat,
                    ecg_target,
                    valid_ppg2ecg,
                    beta=args.huber_beta,
                    diff_weight=args.ppg2ecg_diff_weight,
                )
                loss_ppg_recon = masked_huber_loss(ppg_hat, ppg_target, mask_ppg, beta=args.huber_beta)
                loss_recon_both = masked_huber_loss(recon_both, beats_both, mask_both, beta=args.huber_beta)

                loss = (
                    args.w_both * loss_both
                    + args.w_ppg * loss_ppg
                    + args.w_cons * loss_cons
                    + args.w_ppg2ecg * loss_ppg2ecg
                    + args.w_ppg_recon * loss_ppg_recon
                    + args.w_recon_both * loss_recon_both
                )
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(proj.parameters()), args.grad_clip)
            opt.step()

            total_loss += float(loss.detach().cpu())
            component_sums["teacher_both"] += float(loss_both.detach().cpu())
            component_sums["teacher_ppg"] += float(loss_ppg.detach().cpu())
            component_sums["cons"] += float(loss_cons.detach().cpu())
            component_sums["ppg2ecg"] += float(loss_ppg2ecg.detach().cpu())
            component_sums["ppg_recon"] += float(loss_ppg_recon.detach().cpu())
            component_sums["recon_both"] += float(loss_recon_both.detach().cpu())
            n_steps += 1
            if step % args.log_every == 0:
                now_t = time.time()
                dt = now_t - last_log_t
                last_log_t = now_t
                elapsed = now_t - epoch_t0
                pct = 100.0 * step / max(steps_per_epoch, 1)
                eta = elapsed / max(step, 1) * max(steps_per_epoch - step, 0)
                # 2*bsz samples/step: the both-view and the ppg-view are processed
                # together in the merged forward, so real throughput counts both.
                sps = (args.log_every * 2 * bsz) / max(dt, 1e-6)
                print(
                    f"epoch={epoch:03d} step={step:06d}/{steps_per_epoch:06d} "
                    f"pct={pct:5.1f}% {sps:6.0f} smp/s "
                    f"elapsed={fmt_hms(elapsed)} eta={fmt_hms(eta)} "
                    f"loss={total_loss / max(n_steps, 1):.5f} "
                    f"both={float(loss_both.detach().cpu()):.5f} "
                    f"ppg={float(loss_ppg.detach().cpu()):.5f} "
                    f"cons={float(loss_cons.detach().cpu()):.5f} "
                    f"ppg2ecg={float(loss_ppg2ecg.detach().cpu()):.5f} "
                    f"ppg_recon={float(loss_ppg_recon.detach().cpu()):.5f} "
                    f"recon_both={float(loss_recon_both.detach().cpu()):.5f}",
                    flush=True,
                )

        epoch_loss = total_loss / max(n_steps, 1)
        epoch_components = {key: val / max(n_steps, 1) for key, val in component_sums.items()}
        log(
            f"[epoch {epoch:03d}] done loss={epoch_loss:.5f} "
            f"ppg2ecg={epoch_components['ppg2ecg']:.5f} "
            f"cons={epoch_components['cons']:.5f} "
            f"elapsed={fmt_hms(time.time() - epoch_t0)}"
        )
        save_obj = {
            "model_state": model.state_dict(),
            "proj_state": proj.state_dict(),
            "epoch": epoch,
            "loss": epoch_loss,
            "loss_components": epoch_components,
            "args": vars(args),
        }
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        epoch_path = epoch_checkpoint_path(save_path, epoch)
        torch.save(save_obj, epoch_path)
        log(f"[epoch {epoch:03d}] saved epoch checkpoint {epoch_path}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = save_path.with_name(save_path.stem + "_best.pt")
            torch.save(save_obj, best_path)
            log(f"[epoch {epoch:03d}] saved best {best_path}")


if __name__ == "__main__":
    main()
