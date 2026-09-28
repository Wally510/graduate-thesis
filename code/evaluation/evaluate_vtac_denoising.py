#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, os, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

POLICIES={
    "frozen":"epoch014_frozen_random_token_dense_targetbeat_denoising_ddp2_v1",
    "fullfinetune":(
        "epoch014_fullfinetune_random_token_dense_targetbeat_denoising_"
        "ddp2_v1"
    ),
}
CURRENT_VARIANT=os.environ.get("CURRENT_VARIANT","")
if CURRENT_VARIANT not in POLICIES:
    raise RuntimeError(f"CURRENT_VARIANT错误：{CURRENT_VARIANT}")
POLICY=POLICIES[CURRENT_VARIANT]
FAMILIES=("gaussian","baseline_wander","narrowband_sinusoid","motion_burst","mixed")
SEVERITIES=("mild","moderate","severe")
CORRUPTION_IMPLEMENTATION="embedded_exact_training_v1"

def deterministic_corrupt_target_beats(
    beats: torch.Tensor,
    target_beat: torch.Tensor,
    *,
    family_name: str,
    severity_name: str,
    generator: torch.Generator,
) -> torch.Tensor:
    """复刻训练加噪公式，避免依赖服务器训练代码的函数签名。"""
    if family_name not in FAMILIES:
        raise ValueError(f"未知噪声类型：{family_name}")
    if severity_name not in SEVERITIES:
        raise ValueError(f"未知噪声强度：{severity_name}")
    b, _, channels, length = beats.shape
    if target_beat.shape != beats.shape[:2]:
        raise ValueError(
            f"target_beat形状错误：{tuple(target_beat.shape)}，"
            f"预期{tuple(beats.shape[:2])}"
        )
    device, dtype = beats.device, beats.dtype
    family_index = {
        "gaussian": 1,
        "baseline_wander": 2,
        "narrowband_sinusoid": 3,
        "motion_burst": 4,
        "mixed": 5,
    }[family_name]
    severity_index = SEVERITIES.index(severity_name)
    family = torch.full(
        (b,), family_index, device=device, dtype=torch.long
    )
    severity = torch.full(
        (b,), severity_index, device=device, dtype=torch.long
    )
    scale = torch.tensor(
        [0.65, 1.0, 1.55], device=device, dtype=dtype
    )[severity].view(b, 1, 1, 1)
    position = torch.linspace(
        0.0, 1.0, length, device=device, dtype=dtype
    ).view(1, 1, 1, length)
    noise = torch.zeros_like(beats)

    gaussian_on = (family == 1) | (family == 5)
    gaussian_sigma = 0.16 * scale * (
        0.75
        + 0.50
        * torch.rand(
            b, 1, channels, 1,
            device=device, dtype=dtype, generator=generator,
        )
    )
    noise = noise + (
        torch.randn(
            beats.shape, device=device, dtype=dtype, generator=generator
        )
        * gaussian_sigma
        * gaussian_on.view(b, 1, 1, 1)
    )

    baseline_on = (family == 2) | (family == 5)
    baseline_amp = 0.30 * scale * (
        0.70
        + 0.60
        * torch.rand(
            b, 1, channels, 1,
            device=device, dtype=dtype, generator=generator,
        )
    )
    baseline_cycles = 0.20 + 1.30 * torch.rand(
        b, 1, channels, 1,
        device=device, dtype=dtype, generator=generator,
    )
    baseline_phase = 2.0 * math.pi * torch.rand(
        b, 1, channels, 1,
        device=device, dtype=dtype, generator=generator,
    )
    baseline = baseline_amp * torch.sin(
        2.0 * math.pi * baseline_cycles * position + baseline_phase
    )
    noise = noise + baseline * baseline_on.view(b, 1, 1, 1)

    narrow_on = (family == 3) | (family == 5)
    narrow_amp = 0.10 * scale * (
        0.70
        + 0.60
        * torch.rand(
            b, 1, channels, 1,
            device=device, dtype=dtype, generator=generator,
        )
    )
    narrow_cycles = torch.randint(
        4, 19, (b, 1, channels, 1),
        device=device, generator=generator,
    ).to(dtype)
    narrow_phase = 2.0 * math.pi * torch.rand(
        b, 1, channels, 1,
        device=device, dtype=dtype, generator=generator,
    )
    narrow = narrow_amp * torch.sin(
        2.0 * math.pi * narrow_cycles * position + narrow_phase
    )
    noise = noise + narrow * narrow_on.view(b, 1, 1, 1)

    motion_on = (family == 4) | (family == 5)
    center = 0.10 + 0.80 * torch.rand(
        b, 1, channels, 1,
        device=device, dtype=dtype, generator=generator,
    )
    width = 0.035 + 0.12 * torch.rand(
        b, 1, channels, 1,
        device=device, dtype=dtype, generator=generator,
    )
    envelope = torch.exp(-0.5 * ((position - center) / width).square())
    motion_amp = 0.65 * scale * (
        0.60
        + 0.80
        * torch.rand(
            b, 1, channels, 1,
            device=device, dtype=dtype, generator=generator,
        )
    )
    motion_texture = 0.55 * torch.randn(
        beats.shape, device=device, dtype=dtype, generator=generator
    ) + torch.empty(
        b, 1, channels, 1, device=device, dtype=dtype
    ).uniform_(-1.0, 1.0, generator=generator)
    noise = noise + (
        motion_amp
        * envelope
        * motion_texture
        * motion_on.view(b, 1, 1, 1)
    )

    noise = torch.where(
        (family == 5).view(b, 1, 1, 1), 0.55 * noise, noise
    )
    return torch.where(
        target_beat[:, :, None, None], beats + noise, beats
    )

def load(path,name):
    s=importlib.util.spec_from_file_location(name,path)
    if s is None or s.loader is None: raise ImportError(path)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def stable(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str,parts)).encode()).digest()[:8],"little")
class Metric:
    def __init__(self): self.n=0; self.a=0.; self.q=0.; self.x=0.; self.y=0.; self.xx=0.; self.yy=0.; self.xy=0.
    def add(self,x,y):
        x=np.asarray(x,dtype=np.float64).ravel(); y=np.asarray(y,dtype=np.float64).ravel(); d=x-y
        self.n+=x.size; self.a+=np.abs(d).sum(); self.q+=(d*d).sum(); self.x+=x.sum(); self.y+=y.sum(); self.xx+=(x*x).sum(); self.yy+=(y*y).sum(); self.xy+=(x*y).sum()
    def report(self):
        den=(self.n*self.xx-self.x*self.x)*(self.n*self.yy-self.y*self.y)
        return {"n":self.n,"mae":self.a/self.n,"rmse":math.sqrt(self.q/self.n),"pearson":(self.n*self.xy-self.x*self.y)/math.sqrt(den) if den>1e-20 else float("nan")}
def scalar(x,y):
    x=np.asarray(x,dtype=np.float64).ravel(); y=np.asarray(y,dtype=np.float64).ravel(); d=x-y
    return float(np.abs(d).mean()),float(np.sqrt((d*d).mean())),float(np.corrcoef(x,y)[0,1]) if x.std()>1e-10 and y.std()>1e-10 else float("nan")
def write_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    p=argparse.ArgumentParser()
    for x in ("checkpoint","module-path","preprocess-module","vtac-root","vtac-manifest","reference-mask-manifest","denoise-checkpoint","denoise-train-root","base-eval-root","beat-eval-root","common-root","dense-root","parent-root","output-dir"): p.add_argument("--"+x,required=True)
    p.add_argument("--batch-size",type=int,default=32); p.add_argument("--seed",type=int,default=7142); p.add_argument("--examples-per-condition",type=int,default=1); p.add_argument("--device",default="cuda:0")
    a=p.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[a.base_eval_root,a.beat_eval_root,a.common_root,a.dense_root,a.denoise_train_root]
    base=load(Path(a.base_eval_root)/"evaluate_phase8_vtac.py","denoise_vtac_base")
    den=load(Path(a.denoise_train_root)/"train_denoising.py","denoise_train_for_vtac")
    raw=torch.load(a.denoise_checkpoint,map_location="cpu",weights_only=False)
    if raw.get("policy")!=POLICY or not raw.get("full_model_state"): raise RuntimeError("去噪checkpoint policy或完整backbone错误")
    if raw.get("validation_disabled") is not True: raise RuntimeError("去噪checkpoint未关闭validation")
    if raw.get("backbone_mode")!=CURRENT_VARIANT: raise RuntimeError("去噪checkpoint backbone_mode错误")
    if raw.get("phase_token_replacement") is not False or raw.get("phase_mask_token_trainable") is not False: raise RuntimeError("去噪checkpoint错误使用phase mask token")
    expected_losses={"target_beat_smoothl1":1.0,"target_beat_derivative":0.1,"clean_context_anchor":0.005}
    if raw.get("active_losses")!=expected_losses: raise RuntimeError("去噪checkpoint loss协议错误")
    cfg=SimpleNamespace(beat_len=128,d_model=768,nhead=12,num_layers=9,dim_feedforward=2048,dropout=.1,phase_tokens=8)
    audit,_=den.BASE.load_common(Path(a.common_root)); mm=audit.load_module(Path(a.module_path)); model=audit.build_model(mm,cfg).to(a.device)
    base_report=audit.load_checkpoint_exact(model,Path(a.checkpoint)); model.load_state_dict(raw["full_model_state"],strict=True)
    adapter=den.DENSE.PhaseMaskAdapterTokenDense(d_model=768,phase_tokens=8,beat_len=128,initial_mask_token=model.mask_token,hidden_dim=256).to(a.device)
    adapter.load_state_dict(raw["phase_adapter_state"],strict=True); model.eval(); adapter.eval()
    manifest_audit=base.audit_canonical_test_manifest(Path(a.vtac_manifest))
    prep=base.load_module(Path(a.preprocess_module))
    samples=base.load_vtac_samples(Path(a.vtac_root),Path(a.vtac_manifest),"test",0,a.seed,25,128,250,240.0,10.0,prep)
    candidates=[s for s in samples if int(s.beat_mask.sum())>=2]
    authoritative_ids=[]; authoritative_targets={}
    with Path(a.reference_mask_manifest).open(encoding="utf-8") as f:
        for line in f:
            row=json.loads(line); sid=str(row["sample_id"])
            if sid not in authoritative_targets:
                targets=row.get("target_beats",[])
                if len(targets)!=1: raise RuntimeError(f"权威manifest target错误：{sid}")
                authoritative_ids.append(sid);authoritative_targets[sid]=int(targets[0])
            elif row.get("target_beats") != [authoritative_targets[sid]]:
                raise RuntimeError(f"权威manifest同一样本target不一致：{sid}")
    if len(authoritative_ids)!=2654: raise RuntimeError(f"权威manifest唯一sample={len(authoritative_ids)}，预期2654")
    candidate_by_id={s.sample_id:s for s in candidates}
    if len(candidate_by_id)!=len(candidates): raise RuntimeError("当前VTaC加载结果含重复sample_id")
    missing=[sid for sid in authoritative_ids if sid not in candidate_by_id]
    if missing: raise RuntimeError(f"无法加载权威2654集合中的样本：{missing[:10]}")
    eligible=[candidate_by_id[sid] for sid in authoritative_ids]
    conditions=[("clean_identity","clean")]+[(f,s) for f in FAMILIES for s in SEVERITIES]
    rows=[]; per=[]; manifest=[]; example_root=out/"waveform_examples"; example_root.mkdir()
    device=torch.device(a.device); use_amp=device.type=="cuda" and torch.cuda.is_bf16_supported()
    for family,severity in conditions:
        cid=f"{family}__{severity}"; recm={c:Metric() for c in ("both","ecg","ppg")}; corm={c:Metric() for c in ("both","ecg","ppg")}; examples=0
        for start in range(0,len(eligible),a.batch_size):
            group=eligible[start:start+a.batch_size]
            beats=torch.from_numpy(np.stack([s.beats for s in group])).float().to(device); times=torch.from_numpy(np.stack([s.time_sec for s in group])).float().to(device); bm=torch.from_numpy(np.stack([s.beat_mask for s in group])).bool().to(device)
            corrupted=[]; targets=[]
            for i,sample in enumerate(group):
                target=authoritative_targets[sample.sample_id]
                if target<0 or target>=len(sample.beat_mask) or not sample.beat_mask[target]: raise RuntimeError(f"权威target在当前样本无效：{sample.sample_id}:{target}")
                targets.append(target)
                if family=="clean_identity": ci=beats[i:i+1].clone(); noise_seed=stable(a.seed,sample.sample_id,cid,"noise")%(2**63-1)
                else:
                    noise_seed=stable(a.seed,sample.sample_id,cid,"noise")%(2**63-1); g=torch.Generator(device=device).manual_seed(noise_seed)
                    tm=torch.zeros((1,beats.shape[1]),dtype=torch.bool,device=device); tm[0,target]=True
                    ci=deterministic_corrupt_target_beats(
                        beats[i:i+1],tm,
                        family_name=family,
                        severity_name=severity,
                        generator=g,
                    )
                corrupted.append(ci); manifest.append({"sample_id":sample.sample_id,"record_id":sample.metadata.get("record_id",sample.sample_id),"condition_id":cid,"target_beat_index":target,"noise_seed":int(noise_seed)})
            corrupted=torch.cat(corrupted)
            with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=use_amp):
                phase_z=den.encode_backbone(model,adapter,corrupted,times,bm,gradient_checkpointing=False)
                pred=adapter.decode(phase_z)
            clean=beats.cpu().numpy(); corr=corrupted.cpu().numpy(); recovered=pred.float().cpu().numpy()
            for i,(sample,target) in enumerate(zip(group,targets)):
                for ch,name in ((None,"both"),(0,"ecg"),(1,"ppg")):
                    y=clean[i,target] if ch is None else clean[i,target,ch]; x=corr[i,target] if ch is None else corr[i,target,ch]; z=recovered[i,target] if ch is None else recovered[i,target,ch]
                    corm[name].add(x,y); recm[name].add(z,y)
                cmae,crmse,ccorr=scalar(corr[i,target],clean[i,target]); rmae,rrmse,rcorr=scalar(recovered[i,target],clean[i,target])
                per.append({"sample_id":sample.sample_id,"condition_id":cid,"family":family,"severity":severity,"target_beat_index":target,"corrupted_mae":cmae,"corrupted_rmse":crmse,"corrupted_pearson":ccorr,"recovered_mae":rmae,"recovered_rmse":rrmse,"recovered_pearson":rcorr,"rmse_improvement":crmse-rrmse})
                if examples<a.examples_per_condition:
                    fig,ax=plt.subplots(2,1,figsize=(12,6),sharex=True)
                    for ch,label in enumerate(("ECG","PPG")):
                        ax[ch].plot(clean[i,target,ch],color="black",label="Original",lw=1.5); ax[ch].plot(corr[i,target,ch],color="tomato",label="Corrupted",alpha=.8); ax[ch].plot(recovered[i,target,ch],color="royalblue",label="Recovered",alpha=.9); ax[ch].set_title(f"{label} | {cid} | {sample.sample_id}"); ax[ch].legend(fontsize=8); ax[ch].grid(alpha=.2)
                    fig.tight_layout(); fig.savefig(example_root/f"{cid}_example{examples+1:02d}.png",dpi=160); plt.close(fig); examples+=1
        for ch in ("both","ecg","ppg"):
            cr=corm[ch].report(); rr=recm[ch].report()
            rows.append({"condition_id":cid,"family":family,"severity":severity,"channel":ch,"evaluated_sample_count":len(eligible),"corrupted_mae":cr["mae"],"corrupted_rmse":cr["rmse"],"corrupted_pearson":cr["pearson"],"recovered_mae":rr["mae"],"recovered_rmse":rr["rmse"],"recovered_pearson":rr["pearson"],"rmse_improvement":cr["rmse"]-rr["rmse"],"relative_rmse_improvement":(cr["rmse"]-rr["rmse"])/cr["rmse"] if cr["rmse"]>0 else None})
    write_csv(out/"metrics_by_condition.csv",rows); write_csv(out/"per_sample_metrics.csv",per)
    with (out/"fixed_vtac_denoise_manifest.jsonl").open("w",encoding="utf-8") as f:
        for r in manifest:f.write(json.dumps(r,ensure_ascii=False)+"\n")
    both=[r for r in rows if r["channel"]=="both" and r["family"]!="clean_identity"]
    fig,ax=plt.subplots(figsize=(14,6)); x=np.arange(len(both)); ax.bar(x-.2,[r["corrupted_rmse"] for r in both],.4,label="Corrupted"); ax.bar(x+.2,[r["recovered_rmse"] for r in both],.4,label="Recovered"); ax.set_xticks(x,[r["condition_id"] for r in both],rotation=60,ha="right",fontsize=8); ax.legend(); ax.set_ylabel("RMSE"); ax.grid(axis="y",alpha=.2); fig.tight_layout(); fig.savefig(out/"vtac_denoise_dashboard.png",dpi=180); plt.close(fig)
    report={"status":"complete","training_variant":CURRENT_VARIANT,"canonical_unique_record_ids":manifest_audit["unique_record_id_count"],"loaded_sample_count":len(samples),"candidate_ge2_count_before_authoritative_filter":len(candidates),"eligible_sample_count":len(eligible),"authoritative_sample_count":len(authoritative_ids),"excluded_non_authoritative_candidates":len(candidates)-len(eligible),"reference_mask_manifest":a.reference_mask_manifest,"reference_mask_manifest_sha256":hashlib.sha256(Path(a.reference_mask_manifest).read_bytes()).hexdigest(),"target_beat_source":"job9996_authoritative_manifest","condition_count":len(conditions),"noisy_condition_count":15,"corruption_implementation":CORRUPTION_IMPLEMENTATION,"checkpoint":a.denoise_checkpoint,"checkpoint_epoch":raw.get("epoch"),"checkpoint_step":raw.get("step"),"policy":raw.get("policy"),"active_losses":raw.get("active_losses"),"base_checkpoint_report":base_report,"full_model_state_loaded_strict":True,"phase_adapter_state_loaded_strict":True,"fixed_manifest_rows":len(manifest),"png_count":len(list(example_root.glob("*.png")))}
    (out/"evaluation_config.json").write_text(json.dumps(vars(a),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"checkpoint_report.json").write_text(json.dumps({"checkpoint":a.denoise_checkpoint,"epoch":raw.get("epoch"),"step":raw.get("step"),"policy":raw.get("policy"),"variant":CURRENT_VARIANT,"full_model_state_key_count":len(raw["full_model_state"]),"phase_adapter_state_key_count":len(raw["phase_adapter_state"]),"strict":True},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"canonical_manifest_audit.json").write_text(json.dumps(manifest_audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); (out/"RUN_COMPLETE.txt").write_text("status=complete\n",encoding="utf-8")
    print("vtac_denoise_complete="+json.dumps(report,ensure_ascii=False))
if __name__=="__main__": main()
