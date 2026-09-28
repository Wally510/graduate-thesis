"""Verify or restore result files compacted by FP-90. No third-party dependencies."""
import argparse,gzip,hashlib,json,os,shutil
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=Path(__file__).resolve().parent.parent);p.add_argument('--restore',action='store_true');a=p.parse_args();root=a.project.resolve()
    def safe(rel):
        q=(root/rel).resolve()
        if not q.is_relative_to(root) or q==root:raise ValueError('Unsafe path')
        return q
    def stream_hash(s):
        h=hashlib.sha256()
        while b:=s.read(2*1024*1024):h.update(b)
        return h.hexdigest()
    def hash_file(q):
        with q.open('rb') as s:return stream_hash(s)
    manifest=json.loads((root/'project_control/FP-90_COMPACTION_MANIFEST_20260927.json').read_text(encoding='utf-8'))
    cmap={r['original']:r for r in manifest['compressed_files']}
    verified=set()
    for n,r in enumerate(manifest['compressed_files'],1):
        with gzip.open(safe(r['compressed']),'rb') as s:
            if stream_hash(s)!=r['sha256']:raise RuntimeError('Compressed content mismatch: '+r['original'])
        verified.add((r['original'],r['sha256']))
        if n%250==0:print(f'Verified {n} compressed files',flush=True)
    for r in manifest['duplicate_paths']:
        key=(r['canonical'],r['sha256'])
        if key in verified:continue
        if hash_file(safe(r['canonical']))!=r['sha256']:raise RuntimeError('Canonical content mismatch')
        verified.add(key)
    print('All retained content verified.',flush=True)
    if not a.restore:return
    for r in manifest['compressed_files']:
        dst=safe(r['original'])
        if dst.exists():
            if hash_file(dst)!=r['sha256']:raise RuntimeError('Refusing to overwrite changed original')
            continue
        tmp=safe(r['original']+'.fp90-restoring')
        with gzip.open(safe(r['compressed']),'rb') as src,tmp.open('xb') as out:shutil.copyfileobj(src,out,2*1024*1024)
        if hash_file(tmp)!=r['sha256']:raise RuntimeError('Restored text hash mismatch')
        os.replace(tmp,dst)
    for r in manifest['duplicate_paths']:
        dst=safe(r['original'])
        if dst.exists():
            if hash_file(dst)!=r['sha256']:raise RuntimeError('Refusing to overwrite changed duplicate path')
        else:
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(safe(r['canonical']),dst)
            if hash_file(dst)!=r['sha256']:raise RuntimeError('Restored duplicate hash mismatch')
    print('Restored original result paths. GZip backups are retained; disk usage will increase.',flush=True)
if __name__=='__main__':main()
