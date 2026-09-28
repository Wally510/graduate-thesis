"""Restore separated weights to their original paths; preserves ZIP member bytes."""
import argparse,csv,hashlib,json,os,shutil,zipfile
from pathlib import Path

def safe(root,rel):
    p=(root/rel).resolve()
    if not p.is_relative_to(root.resolve()) or p==root.resolve():raise ValueError('Unsafe path: '+rel)
    return p
def digest_stream(s):
    h=hashlib.sha256()
    while b:=s.read(8*1024*1024):h.update(b)
    return h.hexdigest()
def digest(p):
    with p.open('rb') as s:return digest_stream(s)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--archive',type=Path,default=Path(__file__).resolve().parent);ap.add_argument('--restore',action='store_true');a=ap.parse_args()
    root=a.project.resolve();arc=a.archive.resolve()
    rows=list(csv.DictReader((arc/'weight_manifest.csv').open(encoding='utf-8-sig',newline='')))
    unique=json.loads((arc/'unique_weights.json').read_text(encoding='utf-8'))
    for r in unique:
        p=safe(arc,r['archived_path'])
        if p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:raise RuntimeError('Archive verification failed: '+str(p))
    print('All unique weight hashes verified.',flush=True)
    if not a.restore:return
    if not root.is_dir():raise RuntimeError('Project folder does not exist')
    for r in rows:
        if r['kind']!='file':continue
        dst=safe(root,r['source_path']);src=safe(arc,r['archived_path'])
        if dst.exists():
            if digest(dst)!=r['sha256']:raise RuntimeError('Refusing overwrite: '+str(dst))
        else:
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        if digest(dst)!=r['sha256']:raise RuntimeError('Restored file hash mismatch')
    mapping={(r['source_path'],r['entry']):r for r in rows if r['kind']=='zip_member'}
    for recipe in json.loads((arc/'archive_recipes.json').read_text(encoding='utf-8')):
        dst=safe(root,recipe['source_path']);tmp=dst.with_name(dst.name+'.restore-staged')
        if tmp.exists():raise RuntimeError('Staging file already exists: '+str(tmp))
        old=zipfile.ZipFile(dst) if dst.exists() else None
        try:
            expected={e['name']:e for e in recipe['entries']}
            if old:
                for info in old.infolist():
                    if info.filename not in expected:raise RuntimeError('Unexpected existing ZIP member')
                    e=expected[info.filename]
                    if not info.is_dir():
                        with old.open(info) as s:
                            if digest_stream(s)!=e['sha256']:raise RuntimeError('Existing ZIP member has changed')
            dst.parent.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(tmp,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as out:
                for e in recipe['entries']:
                    if e['directory']:out.writestr(e['name'],b'');continue
                    if e['weight']:
                        r=mapping[(recipe['source_path'],e['name'])]
                        out.write(safe(arc,r['archived_path']),e['name'])
                    else:
                        if not old:raise RuntimeError('Missing non-weight result ZIP: '+str(dst))
                        with old.open(e['name']) as src,out.open(e['name'],'w',force_zip64=True) as target:shutil.copyfileobj(src,target,8*1024*1024)
        finally:
            if old:old.close()
        with zipfile.ZipFile(tmp) as check:
            for e in recipe['entries']:
                if not e['directory']:
                    with check.open(e['name']) as s:
                        if digest_stream(s)!=e['sha256']:raise RuntimeError('Restored ZIP verification failed')
        os.replace(tmp,dst)
    print('Restored all original file paths and ZIP member contents. ZIP container bytes/metadata may differ.',flush=True)
if __name__=='__main__':main()
