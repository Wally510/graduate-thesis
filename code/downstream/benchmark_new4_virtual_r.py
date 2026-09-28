#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, io, json, math, os, random, re, sys, time, zipfile
from pathlib import Path
from collections import defaultdict, Counter
from types import SimpleNamespace
from typing import Any

os.environ.setdefault('PYTHONHASHSEED','42')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score

NEW_CODE=Path('/data-ai/sl20200894/Code')
BENCH_ROOT=NEW_CODE/'downstream_benchmarks'
DEFAULT_TASK_DIR=Path(__file__).resolve().parents[1]
TASK_DIR=Path(os.environ.get('CODE_DIR', str(DEFAULT_TASK_DIR))).resolve()
RESULTS_ROOT=Path(os.environ.get(
    'RESULTS_DIR',
    str(BENCH_ROOT/'virtual_r_new4_public_results_20260704'),
)).resolve()
MANIFEST=Path(os.environ.get(
    'NEW4_MANIFEST',
    str(TASK_DIR/'manifests_v2/all_new4_manifest_seed42.csv'),
)).resolve()
DATA_ROOT=Path('/data-ai/sl20200894/downstream_datasets')
for p in [Path(__file__).resolve().parent]:
    if str(p) not in sys.path: sys.path.insert(0,str(p))
import benchmark_butppg_foundation_models as base

ZIPS={
 'scientisst_move': DATA_ROOT/'scientisst-move-annotated-wearable-multimodal-biosignals-recorded-during-everyday-life-activities-in-naturalistic-environments-1.0.1.zip',
 'senssmarttech': DATA_ROOT/'senssmarttech-database-of-cardiovascular-signals-synchronously-recorded-by-an-electrocardiograph-phonocardiograph-photoplethysmograph-and-accelerometer-1.0.0.zip',
 'nback_music': DATA_ROOT/'a-multimodal-dataset-for-investigating-working-memory-in-presence-of-music-1.0.0.zip',
}
PAP_SIGMA_MODELS=set()
MODEL_ALIASES={}
TASK_KIND={
 'scientisst_activity_loso':'multiclass', 'scientisst_hr_loso':'regression',
 'simultaneous_load_loso':'multiclass', 'simultaneous_hr_loso':'regression',
 'senssmarttech_ba_5fold':'binary', 'senssmarttech_hr_5fold':'regression', 'senssmarttech_recovery_delta_hr_5fold':'regression',
 'nback_music_workload_loso':'binary',
}


def set_all_seeds(seed:int):
    os.environ['PYTHONHASHSEED']=str(seed); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    try: torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception: pass


def zscore(x):
    x=np.asarray(x,dtype=np.float32); m=np.nanmean(x); s=np.nanstd(x); s=s if np.isfinite(s) and s>1e-6 else 1.0
    return np.nan_to_num((x-m)/s, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def resample_to_len(x, target_len):
    x=np.asarray(x,dtype=np.float32)
    if len(x)==target_len: return x
    if len(x)<2: return np.zeros(target_len,dtype=np.float32)
    # simple linear interpolation is robust enough for input adaptation
    xp=np.linspace(0,1,len(x)); xnew=np.linspace(0,1,target_len)
    return np.interp(xnew,xp,x).astype(np.float32)


def find_in_zip(zf, name):
    if not name: raise FileNotFoundError('empty name')
    if '#' in name: name=name.split('#',1)[0]
    names=zf.namelist()
    if name in names: return name
    m=[n for n in names if n.endswith('/'+name) or n.endswith(name)]
    if not m: raise FileNotFoundError(name)
    return m[0]


def edf_header(zf,name):
    name=find_in_zip(zf,name)
    with zf.open(name) as f:
        fixed=f.read(256); hb=int(fixed[184:192].decode('latin1').strip() or 0); rest=f.read(hb-256)
    head=fixed+rest
    def field(s,l): return head[s:s+l].decode('latin1',errors='replace').strip()
    nrec=int(float(field(236,8))); dur=float(field(244,8)); nsig=int(field(252,4))
    widths=[16,80,8,8,8,8,8,80,8,32]; names=['label','transducer','phys_dim','phys_min','phys_max','dig_min','dig_max','prefilter','spr','reserved']
    pos=256; sec={}
    for nm,w in zip(names,widths):
        vals=[]
        for _ in range(nsig): vals.append(head[pos:pos+w].decode('latin1',errors='replace').strip()); pos+=w
        sec[nm]=vals
    ch=[]
    for i in range(nsig):
        def fl(k,default=0.0):
            try: return float(sec[k][i])
            except Exception: return default
        spr=int(float(sec['spr'][i] or 0))
        ch.append({'index':i,'label':sec['label'][i],'spr':spr,'fs':spr/dur,'phys_min':fl('phys_min'),'phys_max':fl('phys_max',1.0),'dig_min':fl('dig_min'),'dig_max':fl('dig_max',1.0)})
    return {'zip_name':name,'header_bytes':hb,'n_records':nrec,'channels':ch}


def read_edf_segment(zf, name, channel_label, start_sec, end_sec):
    h=edf_header(zf,name); ch=next((c for c in h['channels'] if c['label']==channel_label), None)
    if ch is None: ch=h['channels'][0]
    spr=[c['spr'] for c in h['channels']]; rec_bytes=sum(s*2 for s in spr); offsets=[]; off=0
    for s in spr: offsets.append(off); off+=s*2
    idx=ch['index']; data=[]
    start_samp=int(float(start_sec)*ch['fs']); end_samp=int(float(end_sec)*ch['fs'])
    rec0=max(0,start_samp//max(1,ch['spr'])); rec1=min(h['n_records'], int(math.ceil(end_samp/max(1,ch['spr']))))
    with zf.open(h['zip_name']) as f:
        f.read(h['header_bytes']+rec0*rec_bytes)
        for _ in range(rec0, rec1):
            rec=f.read(rec_bytes)
            if len(rec)<rec_bytes: break
            arr=np.frombuffer(rec[offsets[idx]:offsets[idx]+spr[idx]*2], dtype='<i2').astype(np.float32); data.append(arr)
    x=np.concatenate(data) if data else np.zeros(0,dtype=np.float32)
    rel0=start_samp-rec0*ch['spr']; rel1=rel0+(end_samp-start_samp); x=x[max(0,rel0):max(0,rel1)]
    denom=(ch['dig_max']-ch['dig_min']) or 1.0
    x=(x-ch['dig_min'])*(ch['phys_max']-ch['phys_min'])/denom+ch['phys_min']
    return x.astype(np.float32), float(ch['fs'])


def read_csv_from_zip(zf, path):
    name=find_in_zip(zf,path)
    return pd.read_csv(zf.open(name))


def select_numeric_column(df, preferred=None):
    if preferred and preferred in df.columns: return df[preferred].to_numpy(dtype=np.float32)
    cols=[c for c in df.columns if c.lower() not in {'t','time','timestamp'}]
    for c in cols:
        try: return df[c].to_numpy(dtype=np.float32)
        except Exception: pass
    return df.select_dtypes(include=[np.number]).iloc[:, -1].to_numpy(dtype=np.float32)


def read_signal_pair(row):
    ds=row['dataset']
    if ds=='scientisst_move':
        with zipfile.ZipFile(ZIPS[ds]) as zf:
            ecg,_=read_edf_segment(zf,row['ecg_file'],row['ecg_channel'],float(row['start_sec']),float(row['end_sec']))
            ppg,_=read_edf_segment(zf,row['ppg_file'],row['ppg_channel'],float(row['start_sec']),float(row['end_sec']))
        return ecg, ppg
    if ds=='simultaneous':
        import wfdb
        hea=Path(row['source_file']); rec=str(hea.with_suffix('')); fs=float(row['ecg_fs']); s0=int(float(row['start_sec'])*fs); s1=int(float(row['end_sec'])*fs)
        record=wfdb.rdrecord(rec, sampfrom=s0, sampto=s1, physical=True)
        names=list(record.sig_name)
        ecg_i=names.index(row['ecg_channel']) if row['ecg_channel'] in names else 0
        ppg_i=names.index(row['ppg_channel']) if row['ppg_channel'] in names else min(1, len(names)-1)
        return record.p_signal[:,ecg_i].astype(np.float32), record.p_signal[:,ppg_i].astype(np.float32)
    if ds=='senssmarttech':
        with zipfile.ZipFile(ZIPS[ds]) as zf:
            ecg_df=read_csv_from_zip(zf,row['ecg_file']); ppg_df=read_csv_from_zip(zf,row['ppg_file'])
        ecg=select_numeric_column(ecg_df)
        # use first PPG channel listed, usually carotid_880nm
        ppg_col=str(row.get('ppg_channel','')).split(';')[0]
        ppg=select_numeric_column(ppg_df, ppg_col)
        return ecg, ppg
    if ds=='nback_music':
        with zipfile.ZipFile(ZIPS[ds]) as zf:
            ecg_df=read_csv_from_zip(zf,row['ecg_file']); ppg_df=read_csv_from_zip(zf,row['ppg_file'])
        fs=float(row['ecg_fs']); s0=int(float(row['start_sec'])*fs); s1=int(float(row['end_sec'])*fs)
        ecg=select_numeric_column(ecg_df)[s0:s1]; ppg=select_numeric_column(ppg_df)[s0:s1]
        return ecg, ppg
    raise ValueError(ds)


def load_unique_samples(task, max_unique=0):
    rows=[]
    with MANIFEST.open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            if r['task']==task: rows.append(r)
    uniq={}
    for r in rows:
        key=(r['dataset'],r['subject_id'],r['record_id'],r['window_id'])
        if key not in uniq:
            try:
                y=float(r['label']) if TASK_KIND[task]=='regression' else str(r['label'])
                if TASK_KIND[task]=='regression' and not math.isfinite(float(y)): continue
            except Exception: continue
            rr=dict(r); rr['_key']='|'.join(key); uniq[key]=rr
    samples=list(uniq.values())
    if max_unique and len(samples)>max_unique:
        rng=random.Random(42); rng.shuffle(samples); samples=samples[:max_unique]
    key_to_i={r['_key']:i for i,r in enumerate(samples)}
    folds=defaultdict(lambda: {'train':[], 'val':[], 'test':[]})
    for r in rows:
        key='|'.join((r['dataset'],r['subject_id'],r['record_id'],r['window_id']))
        if key in key_to_i:
            folds[int(r['fold'])][r['split']].append(key_to_i[key])
    return samples, dict(sorted(folds.items()))


def prepare_model_windows(samples, model, args):
    alias=MODEL_ALIASES.get(model, model)
    target_fs={'anyppg':125,'pulseppg':50,'csfm':250,'orthogonal':args.orthogonal_fs}[alias]
    target_len=int(round(args.window_sec*target_fs))
    xs=[]; preprocess=None; project_root=None; channel_ids=None
    if alias=='csfm':
        project_root=base._resolve_csfm_project_root(); preprocess=base._load_preprocess_signal(project_root); channel_ids=base.parse_int_list(args.csfm_channel_ids)
    for i,r in enumerate(samples,1):
        ecg,ppg=read_signal_pair(r)
        if alias=='csfm':
            ecg=resample_to_len(zscore(ecg), target_len); ppg=resample_to_len(zscore(ppg), target_len)
            x=preprocess(signal=np.stack([ecg,ppg],0), channels=channel_ids, fs=target_fs, project_root=project_root)
            xs.append(np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))
        elif alias=='orthogonal':
            ecg=resample_to_len(zscore(ecg), target_len); ppg=resample_to_len(zscore(ppg), target_len)
            xs.append(np.stack([ecg, ppg], axis=0).astype(np.float32))
        else:
            ppg=resample_to_len(zscore(ppg), target_len); xs.append(ppg[None,:].astype(np.float32))
        if i%500==0: print(f'[windows] {model} prepared {i}/{len(samples)}', flush=True)
    return np.stack(xs).astype(np.float32)


def extract_embeddings_for(model, windows, args, device):
    alias=MODEL_ALIASES.get(model, model)
    return base.extract_embeddings(windows, alias, args, device).astype(np.float32)


class Head(nn.Module):
    def __init__(self, d, out_dim, hidden, dropout, kind):
        super().__init__(); self.kind=kind
        self.net=nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden,out_dim))
    def forward(self,x): return self.net(x)


def metrics_reg(y,p):
    y=np.asarray(y,dtype=float); p=np.asarray(p,dtype=float); e=p-y
    out={'mae':float(np.mean(np.abs(e))), 'me':float(np.mean(e)), 'sd':float(np.std(e)), 'rmse':float(np.sqrt(np.mean(e*e)))}
    if len(y)>1 and np.std(y)>1e-8 and np.std(p)>1e-8: out['pearson']=float(np.corrcoef(y,p)[0,1])
    else: out['pearson']=float('nan')
    return out


def metrics_cls(y, logits, kind):
    y=np.asarray(y,dtype=int); pred=np.argmax(logits,axis=1)
    out={'accuracy':float(accuracy_score(y,pred)), 'balanced_accuracy':float(balanced_accuracy_score(y,pred)), 'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0))}
    if kind=='binary' and logits.shape[1]==2 and len(set(y))==2:
        prob=torch.softmax(torch.tensor(logits),dim=1).numpy()[:,1]
        out['auroc']=float(roc_auc_score(y,prob)); out['auprc']=float(average_precision_score(y,prob)); out['f1']=float(f1_score(y,pred,zero_division=0))
    return out


def choose_score(rep, kind):
    if kind=='regression': return rep.get('mae', float('inf')), False, 'mae'
    for k in ['auroc','balanced_accuracy','accuracy','macro_f1']:
        if k in rep and np.isfinite(rep[k]): return rep[k], True, k
    return -float('inf'), True, 'missing'


def train_one(emb, labels, folds, kind, args, device):
    x_all=np.nan_to_num(emb.astype(np.float32)); labels=np.asarray(labels)
    reports=[]
    classes=None
    if kind in {'binary','multiclass'}:
        vals=sorted(set(map(str,labels.tolist()))); classes={v:i for i,v in enumerate(vals)}; y_all=np.asarray([classes[str(v)] for v in labels], dtype=np.int64); out_dim=len(vals)
    else:
        y_all=labels.astype(np.float32); out_dim=1
    for fold,sp in folds.items():
        train_idx=np.asarray(sorted(set(sp['train'])),dtype=int); val_idx=np.asarray(sorted(set(sp['val'])),dtype=int); test_idx=np.asarray(sorted(set(sp['test'])),dtype=int)
        if len(train_idx)==0 or len(val_idx)==0 or len(test_idx)==0: continue
        mu=x_all[train_idx].mean(0,keepdims=True); sd=np.maximum(x_all[train_idx].std(0,keepdims=True),1e-6); x=(x_all-mu)/sd
        set_all_seeds(args.seed+int(fold))
        head=Head(x.shape[1], out_dim, args.head_hidden, args.dropout, kind).to(device)
        opt=torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if kind=='regression': loss_fn=nn.MSELoss(); y_tensor=torch.from_numpy(y_all.astype(np.float32)[:,None])
        else: loss_fn=nn.CrossEntropyLoss(); y_tensor=torch.from_numpy(y_all.astype(np.int64))
        gen=torch.Generator(); gen.manual_seed(args.seed+int(fold))
        loader=DataLoader(TensorDataset(torch.from_numpy(x[train_idx]).float(), y_tensor[train_idx]), batch_size=args.batch_size, shuffle=True, generator=gen)
        best=None; best_state=None; best_metric=''
        for ep in range(1,args.epochs+1):
            head.train()
            for xb,yb in loader:
                xb=xb.to(device); yb=yb.to(device); opt.zero_grad(set_to_none=True); out=head(xb); loss=loss_fn(out, yb); loss.backward(); opt.step()
            head.eval(); outs=[]
            with torch.no_grad():
                for st in range(0,len(val_idx),args.eval_batch_size): outs.append(head(torch.from_numpy(x[val_idx[st:st+args.eval_batch_size]]).float().to(device)).cpu().numpy())
            val_out=np.concatenate(outs,0)
            val_rep=metrics_reg(y_all[val_idx], val_out[:,0]) if kind=='regression' else metrics_cls(y_all[val_idx], val_out, kind)
            score, higher, metric=choose_score(val_rep, kind)
            if best is None or (score>best if higher else score<best): best=score; best_metric=metric; best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}; best_ep=ep
        head.load_state_dict(best_state); head.eval(); outs=[]
        with torch.no_grad():
            for st in range(0,len(test_idx),args.eval_batch_size): outs.append(head(torch.from_numpy(x[test_idx[st:st+args.eval_batch_size]]).float().to(device)).cpu().numpy())
        test_out=np.concatenate(outs,0)
        rep=metrics_reg(y_all[test_idx], test_out[:,0]) if kind=='regression' else metrics_cls(y_all[test_idx], test_out, kind)
        rep.update({'fold':float(fold),'best_epoch':float(best_ep),'best_val_score':float(best),'best_val_metric':best_metric,'train_n':float(len(train_idx)),'val_n':float(len(val_idx)),'test_n':float(len(test_idx))})
        reports.append(rep); print('[fold]',fold,rep,flush=True)
    keys=sorted({k for r in reports for k in r if isinstance(r.get(k), (int,float,np.floating))})
    summary={}
    for k in keys:
        vals=np.asarray([r.get(k,np.nan) for r in reports],dtype=float); summary[k+'_mean']=float(np.nanmean(vals)); summary[k+'_std']=float(np.nanstd(vals))
    return {'classes':classes, 'fold_reports':reports, 'summary':summary}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir', default=str(RESULTS_ROOT/'virtual_r_latest'))
    ap.add_argument('--models', default='orthogonal')
    ap.add_argument('--tasks', default=','.join(TASK_KIND))
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--window-sec', type=float, default=10.0)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--eval-batch-size', type=int, default=512)
    ap.add_argument('--embed-batch-size', type=int, default=128)
    ap.add_argument('--head-hidden', type=int, default=128)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--csfm-channel-ids', default='1,12')
    ap.add_argument('--orthogonal-module-path', default=str(Path(__file__).resolve().with_name('direct_bp_orthogonal_foundation_singlefile.py')))
    ap.add_argument('--orthogonal-ckpt-path', default='/data-ai/sl20200894/Code/foundation_dual_view_alignment_packed_bundle_20260628/dual_view_virtual_r_alignment_packed_amp.pt')
    ap.add_argument('--orthogonal-fs', type=int, default=250)
    ap.add_argument('--orthogonal-beat-len', type=int, default=128)
    ap.add_argument('--orthogonal-max-beats', type=int, default=25)
    ap.add_argument('--orthogonal-d-model', type=int, default=768)
    ap.add_argument('--orthogonal-nhead', type=int, default=12)
    ap.add_argument('--orthogonal-num-layers', type=int, default=9)
    ap.add_argument('--orthogonal-dim-feedforward', type=int, default=2048)
    ap.add_argument('--orthogonal-dropout', type=float, default=0.1)
    ap.add_argument('--orthogonal-phase-tokens', type=int, default=8)
    ap.add_argument('--log-every-batches', type=int, default=20)
    ap.add_argument('--max-unique', type=int, default=0)
    ap.add_argument('--reuse-embeddings', action='store_true')
    args=ap.parse_args(); set_all_seeds(args.seed)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('[start]',time.strftime('%F %T'),'device',device,'args',vars(args),flush=True)
    all_summary={'task':'new4_ecgppg_public_frozen_probe','manifest':str(MANIFEST),'models':{},'config':vars(args)}
    for task in [t.strip() for t in args.tasks.split(',') if t.strip()]:
        kind=TASK_KIND[task]; samples,folds=load_unique_samples(task,args.max_unique)
        labels=[s['label'] for s in samples]
        print('\n[TASK]',task,'kind',kind,'unique',len(samples),'folds',len(folds),'labels',Counter(map(str,labels)).most_common(20),flush=True)
        task_dir=out/task; task_dir.mkdir(parents=True,exist_ok=True)
        all_summary.setdefault('tasks',{})[task]={'unique_samples':len(samples),'folds':len(folds),'kind':kind,'models':{}}
        for model in [m.strip() for m in args.models.split(',') if m.strip()]:
            model_dir=task_dir/model; model_dir.mkdir(parents=True,exist_ok=True)
            emb_path=model_dir/'embeddings.npy'
            if args.reuse_embeddings and emb_path.exists(): emb=np.load(emb_path)
            else:
                windows=prepare_model_windows(samples, model, args)
                print('[embed]',task,model,'windows',windows.shape,flush=True)
                emb=extract_embeddings_for(model, windows, args, device); np.save(emb_path, emb)
                del windows
            res=train_one(emb, labels, folds, kind, args, device)
            with (model_dir/'summary.json').open('w',encoding='utf-8') as f: json.dump(res,f,ensure_ascii=False,indent=2)
            all_summary['tasks'][task]['models'][model]=res['summary']
            with (out/'all_models_summary.json').open('w',encoding='utf-8') as f: json.dump(all_summary,f,ensure_ascii=False,indent=2)
            print('[done-model]',task,model,res['summary'],flush=True)
    print('[done]',time.strftime('%F %T'),flush=True)

if __name__=='__main__': main()
