#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
from statistics import mean
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read(p,delimiter=","):
    with p.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f,delimiter=delimiter))
def write(p,rows):
    with p.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument("--run-root",required=True);p.add_argument("--table",required=True);p.add_argument("--epochs",type=int,default=10);a=p.parse_args();root=Path(a.run_root);table={int(r["epoch"]):r for r in read(Path(a.table),"\t")}
    summaries=[];details=[];hashes=set()
    for ep in range(1,a.epochs+1):
        d=root/f"epoch{ep:02d}"
        if not (d/"RUN_COMPLETE.txt").is_file():raise FileNotFoundError(d/"RUN_COMPLETE.txt")
        audit=json.loads((d/"protocol_audit.json").read_text(encoding="utf-8"))
        if audit.get("status")!="passed":raise RuntimeError(f"epoch{ep}协议审计失败")
        hashes.add(sha(d/"fixed_vtac_denoise_manifest.jsonl"))
        rows=[r for r in read(d/"metrics_by_condition.csv") if r["channel"]=="both"]
        noisy=[r for r in rows if r["family"]!="clean_identity"]
        if len(rows)!=16 or len(noisy)!=15:raise RuntimeError(f"epoch{ep}条件数错误")
        for r in rows:details.append({"epoch":ep,"step":table[ep]["step"],**r})
        summaries.append({"epoch":ep,"step":int(table[ep]["step"]),"corrupted_macro_rmse":mean(float(r["corrupted_rmse"]) for r in noisy),"recovered_macro_mae":mean(float(r["recovered_mae"]) for r in noisy),"recovered_macro_rmse":mean(float(r["recovered_rmse"]) for r in noisy),"recovered_macro_pearson":mean(float(r["recovered_pearson"]) for r in noisy),"macro_relative_rmse_improvement":mean(float(r["relative_rmse_improvement"]) for r in noisy),"clean_identity_rmse":float(next(r for r in rows if r["family"]=="clean_identity")["recovered_rmse"]),"checkpoint":table[ep]["checkpoint"]})
    if len(hashes)!=1:raise RuntimeError("各epoch噪声manifest不一致")
    write(root/"all_epochs_summary.csv",summaries);write(root/"all_epochs_condition_metrics.csv",details)
    diagnostic_min=min(summaries,key=lambda r:r["recovered_macro_rmse"]);eps=[r["epoch"] for r in summaries]
    fig,ax=plt.subplots(2,2,figsize=(12,9))
    ax[0,0].plot(eps,[r["recovered_macro_rmse"] for r in summaries],marker="o",label="Recovered");ax[0,0].plot(eps,[r["corrupted_macro_rmse"] for r in summaries],ls="--",label="Corrupted");ax[0,0].legend();ax[0,0].set_title("Noisy 15-condition macro RMSE")
    ax[0,1].plot(eps,[r["recovered_macro_mae"] for r in summaries],marker="o");ax[0,1].set_title("Recovered macro MAE")
    ax[1,0].plot(eps,[r["recovered_macro_pearson"] for r in summaries],marker="o");ax[1,0].set_title("Recovered macro Pearson")
    ax[1,1].plot(eps,[100*r["macro_relative_rmse_improvement"] for r in summaries],marker="o");ax[1,1].set_title("RMSE improvement (%)")
    for q in ax.flat:q.set_xticks(eps);q.set_xlabel("Denoising epoch");q.grid(alpha=.25)
    fig.suptitle("VTaC denoising epoch sweep");fig.tight_layout();fig.savefig(root/"vtac_denoise_all_epochs_convergence.png",dpi=180);plt.close(fig)
    report={"status":"complete","diagnostic_min_rmse_epoch":diagnostic_min["epoch"],"test_sweep_not_for_checkpoint_selection":True,"fixed_manifest_sha256":next(iter(hashes)),"epoch_count":a.epochs,"noisy_condition_count":15}
    (root/"convergence_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");(root/"ALL_EPOCHS_COMPLETE.txt").write_text(f"status=complete\nepoch_count={a.epochs}\nselection_warning=VTaC_test_not_for_checkpoint_selection\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False))
if __name__=="__main__":main()
